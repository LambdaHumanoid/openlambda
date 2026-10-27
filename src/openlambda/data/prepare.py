"""Create a new split/statistics bundle; never edits source datasets.

Usage: python -m lambda0 prepare --source-config data.json --data-root /data --output /bundles/run
Optional --normalization-groups maps metadata dataset names to logical dataset names.
"""
import argparse
import copy
import json
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lambda0.runtime.paths import resolve_path, metadata_paths
from lambda0.data.episode_index import EpisodeIndex
from lambda0.data.release_data import CONTRACT, quantile_parameters, split_episodes, valid_observations
from lambda0.models.action_specs import get_action_space_spec
from lambda0.data.camera_timeline import load_camera_episode, camera_valid_observations


@lru_cache(maxsize=1)
def _raw_table(path, columns):
    # SIMPLE stores many episodes in one parquet: keep only the last table,
    # and slice before materializing Python rows (not the whole corpus per episode).
    return pq.read_table(path, columns=list(columns), memory_map=True)


def raw_episode(meta, item):
    if meta["handler"].endswith(":CameraEpisodeHandler"):
        return load_camera_episode(meta, item)[:4]
    root = Path(meta["root_path"])
    episode = int(item["episode_index"])
    template = meta["data_path"]
    path = root / template.format(episode_index=episode, episode_chunk=episode // int(meta.get("chunks_size", 1000)))
    simple = meta["handler"].endswith(":PsiSimpleHandler")
    pico = meta["handler"].endswith(":ParquetEpisodeHandler")
    hand_state = meta["handler"].endswith(":HandStateEpisodeHandler")
    if not simple and not pico and not hand_state:
        raise ValueError("Release preparation supports explicit Pico/G1 and SIMPLE contracts only; convert other adapters first.")
    state_key = "states" if simple else "observation.state"
    columns = ([state_key, "observation.valid_mask"] if hand_state else
               [state_key, "action"] + ([] if simple else ["observation.valid_mask", "action.valid_mask"]))
    table = _raw_table(str(path), tuple(columns))
    if simple and meta.get("storage_layout", "merged_v3") == "merged_v3":
        start, stop = int(item["dataset_from_index"]), int(item["dataset_to_index"])
        if start < 0 or stop < start or stop > table.num_rows:
            raise ValueError(f"Invalid episode range [{start},{stop}) in {path}")
        table = table.slice(start, stop - start)
    table = table.to_pydict()
    state = np.asarray(table[state_key], dtype=np.float32)
    action = state.copy() if hand_state else np.asarray(table["action"], dtype=np.float32)
    sm = np.ones_like(state, dtype=bool) if simple else np.asarray(table["observation.valid_mask"], dtype=bool)
    am = sm.copy() if hand_state else (np.ones_like(action, dtype=bool) if simple else np.asarray(table["action.valid_mask"], dtype=bool))
    spec = get_action_space_spec(meta["action_space"])
    if state.shape != (item["length"], spec.proprio_dim) or action.shape != (item["length"], spec.action_dim):
        raise ValueError(f"Invalid raw shape: {path}")
    if sm.shape != state.shape or am.shape != action.shape:
        raise ValueError(f"Invalid validity-mask shape: {path}")
    if not np.isfinite(state[sm]).all() or not np.isfinite(action[am]).all():
        raise ValueError(f"Nonfinite valid state/action values: {path}")
    return state, action, sm, am



def _write_bundle(output, *, args, source, data, metas, groups, stage):
    for subdir in ("train", "validation", "stats"):
        (output / subdir).mkdir()
    manifest = {"contract": CONTRACT, "bundle_version": 2, "seed": args.seed, "validation_fraction": .01,
                "sampling": copy.deepcopy(data.get("sampling")), "datasets": {}}
    grouped = {}
    for meta in metas:
        name = meta["dataset_name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Dataset names must be plain file names.")
        if not isinstance(meta.get("fps"), (float, int)) or meta["fps"] <= 0:
            raise ValueError(f"Dataset must declare its actual positive fps: {name}")
        if "datalist" not in meta:
            path = Path(meta["episode_index_path"])
            if not path.is_absolute():
                path = Path(meta["root_path"]) / path
            meta["datalist"] = list(EpisodeIndex(path))
            meta.pop("episode_index_path", None)
        train, val = split_episodes(meta["datalist"], args.seed)
        if not train:
            raise ValueError(f"No training episodes: {name}")
        # Midtraining is one logical corpus; dates are sampling sources, not
        # independent normalization datasets. Still separate action contracts.
        default_group = (f"midtrain/{meta['action_space']}/{meta['coordinate_contract']}"
                         if source.get("stage") == "midtrain" else name)
        logical = groups.get(name, default_group)
        grouped.setdefault(logical, []).append((meta, train, val))
    for logical, entries in grouped.items():
        signatures = {(m["action_space"], m["coordinate_contract"]) for m, _, _ in entries}
        if len(signatures) != 1:
            raise ValueError("A statistics group cannot mix action spaces/coordinate contracts.")
        spec = get_action_space_spec(entries[0][0]["action_space"])
        state_rows = sum(int(item["length"]) for _, train, _ in entries for item in train)
        action_rows = sum(int(item.get("action_length", item["length"])) for _, train, _ in entries for item in train)
        rows = max(state_rows, action_rows)
        width = spec.proprio_dim + spec.action_dim
        with tempfile.TemporaryDirectory(prefix="stats-", dir=output) as scratch:
            values = np.memmap(Path(scratch) / "values", mode="w+", dtype="float32", shape=(rows, width))
            masks = np.memmap(Path(scratch) / "masks", mode="w+", dtype="bool", shape=(rows, width))
            masks[:] = False
            values[:] = 0
            state_offset = action_offset = 0
            for meta, train, val in entries:
                for split_name, items in (("train", train), ("validation", val)):
                    for item in items:
                        if meta["handler"].endswith(":CameraEpisodeHandler"):
                            state, action, sm, am, ct, at, _ = load_camera_episode(meta, item)
                            item["valid_sample_indices"] = camera_valid_observations(sm, am, ct, at)
                        else:
                            state, action, sm, am = raw_episode(meta, item)
                            item["valid_sample_indices"] = valid_observations(sm, am)
                        if not item["valid_sample_indices"]:
                            raise ValueError(f"Episode has no valid supervision: {meta['dataset_name']} {item['episode_index']}")
                        if split_name == "train":
                            ns, na = len(state), len(action)
                            values[state_offset:state_offset+ns, :spec.proprio_dim] = state
                            masks[state_offset:state_offset+ns, :spec.proprio_dim] = sm
                            values[action_offset:action_offset+na, spec.proprio_dim:] = action
                            masks[action_offset:action_offset+na, spec.proprio_dim:] = am
                            state_offset += ns
                            action_offset += na
            proprio = quantile_parameters(values[:, :spec.proprio_dim], masks[:, :spec.proprio_dim])
            action = quantile_parameters(values[:, spec.proprio_dim:], masks[:, spec.proprio_dim:])
            del values, masks
        group_train_ids = {m["dataset_name"]: [i["episode_index"] for i in train] for m, train, _ in entries}
        for meta, train, val in entries:
            name = meta["dataset_name"]
            # One alias file per source, identical group parameters and provenance.
            file_key = name
            stats = {"schema_version": 1, "dataset_name": name, "action_space": spec.name,
                     "coordinate_contract": meta["coordinate_contract"],
                     "normalization_contract": meta.get("normalization_contract", "quantile_q01_q99_v1"),
                     "release_contract": CONTRACT, "method": "q01_q99", "clip": None,
                     "normalization_group": logical, "training_episodes": group_train_ids,
                     "action": action, "proprio": proprio}
            stats_path = output / "stats" / f"{file_key}.json"
            stats_path.write_text(json.dumps(stats, indent=2))
            manifest["datasets"][name] = {"normalization_group": logical,
                                        "train": [i["episode_index"] for i in train],
                                        "validation": [i["episode_index"] for i in val]}
            for split_name, items in (("train", train), ("validation", val)):
                if not items:
                    continue
                selected = copy.deepcopy(meta)
                selected.update(datalist=items, episode_count=len(items), release_contract=CONTRACT,
                                root_path=Path(meta["root_path"]).relative_to(args.data_root).as_posix(),
                                normalization_path=f"../stats/{file_key}.json", normalization_contract=stats["normalization_contract"])
                (output / split_name / f"{name}.json").write_text(json.dumps(selected, indent=2))
    if stage == "pretrain" and data.get("sampling") == {"strategy": "sqrt_frames"}:
        weights = {}
        for path in sorted((output / "train").glob("*.json")):
            prepared = json.loads(path.read_text())
            usable = sum(len(item["valid_sample_indices"]) for item in prepared["datalist"])
            if usable <= 0:
                raise ValueError(f"No usable Stage-I frames in {path.stem}.")
            weights[path.stem] = math.sqrt(usable)
        manifest["sampling"] = {
            "strategy": "hierarchical_iid",
            "groups": [{
                "name": "egocentric_hand", "action_space": "human_hand_geometry", "weight": 1.0,
                "datasets": {"mode": "explicit", "weights": weights},
                "task": {"mode": "none", "key": None},
                "episode": {"mode": "uniform"}, "timestep": {"mode": "uniform"},
            }],
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

def main(argv=None):
    _raw_table.cache_clear()
    parser = argparse.ArgumentParser(prog="lambda0 prepare", description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalization-groups", type=Path)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)
    args.source_config = resolve_path(args.source_config)
    args.data_root = resolve_path(args.data_root)
    args.output = resolve_path(args.output)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    source = json.loads(args.source_config.read_text())
    stage = source.get("stage")
    if stage not in (None, "pretrain", "midtrain", "posttrain"):
        raise ValueError("Source stage must be pretrain, midtrain, or posttrain.")
    data = source["data"]
    metas_path = resolve_path(data["metas_path"], args.source_config.parent)
    requested = data.get("datasets")
    if requested and not all(isinstance(n, str) for n in requested):
        raise ValueError("data.datasets must contain metadata names.")
    paths = sorted(metas_path.glob("*.json")) if metas_path.is_dir() else [metas_path]
    metas = []
    aliases = {
        "datasets.domain_handler.pico_gopro_midtrain:PicoGoProMidtrainHandler": "lambda0.data.domain_handler.parquet_episode:ParquetEpisodeHandler",
        "lambda0.data.domain_handler.pico_gopro_midtrain:PicoGoProMidtrainHandler": "lambda0.data.domain_handler.parquet_episode:ParquetEpisodeHandler",
        "datasets.domain_handler.psi_simple:PsiSimpleHandler": "lambda0.data.domain_handler.psi_simple:PsiSimpleHandler",
    }
    for path in paths:
        if requested and path.stem not in requested:
            continue
        meta = metadata_paths(json.loads(path.read_text()), path, args.data_root)
        meta["handler"] = aliases.get(meta["handler"], meta["handler"])
        if stage == "pretrain" and (
            meta["handler"] != "lambda0.data.domain_handler.hand_state_episode:HandStateEpisodeHandler"
            or meta["action_space"] != "human_hand_geometry"
        ):
            raise ValueError("Stage I requires the 138D HandStateEpisodeHandler for every source.")
        dataset_root = Path(meta["root_path"])
        if not dataset_root.is_relative_to(args.data_root):
            raise ValueError(f"Dataset root must be inside --data-root: {dataset_root}")
        metas.append(meta)
    if not metas or (requested and set(requested) != {m["dataset_name"] for m in metas}):
        raise ValueError("Source dataset selection could not be resolved exactly.")
    groups = json.loads(resolve_path(args.normalization_groups).read_text()) if args.normalization_groups else {}
    if set(groups) - {m["dataset_name"] for m in metas}:
        raise ValueError("Unknown dataset in normalization groups.")
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temp_dir:
        _write_bundle(Path(temp_dir), args=args, source=source, data=data,
                      metas=metas, groups=groups, stage=stage)
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temp_dir, destination)
    print(f"Prepared {destination}; no training started.")


if __name__ == "__main__":
    main()
