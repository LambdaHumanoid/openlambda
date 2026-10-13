"""Portable data-bundle validation, shared by training and diagnostics."""
import json
from pathlib import Path
from ..runtime.paths import metadata_paths
from .release_data import CONTRACT
from .hierarchical_sampling import HierarchicalSampler


def verify_bundle(bundle, data_root, sampling):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("contract") != CONTRACT or manifest.get("bundle_version") != 2:
        raise ValueError("Prepare a version-2 portable data bundle with lambda0 prepare.")
    train = {}
    for split in ("train", "validation"):
        seen = set()
        for path in sorted((bundle / split).glob("*.json")):
            raw = json.loads(path.read_text())
            name = raw["dataset_name"]
            if name in seen or raw.get("release_contract") != CONTRACT:
                raise ValueError(f"Duplicate or incompatible metadata: {path}")
            seen.add(name)
            expected = manifest["datasets"].get(name)
            if expected is None:
                raise ValueError(f"Unknown dataset in metadata: {path}")
            if set(expected["train"]) & set(expected["validation"]):
                raise ValueError(f"Episode leakage: {name}")
            if [i["episode_index"] for i in raw["datalist"]] != expected[split]:
                raise ValueError(f"Episode split changed: {path}")
            if Path(raw["root_path"]).is_absolute() or Path(raw["normalization_path"]).is_absolute():
                raise ValueError("Bundle paths must be relative.")
            meta = metadata_paths(raw, path, data_root)
            root = Path(meta["root_path"])
            if not root.is_relative_to(Path(data_root).resolve()) or not root.is_dir():
                raise ValueError(f"Dataset root missing/outside data root: {root}")
            stats_path = Path(meta["normalization_path"])
            if not stats_path.is_relative_to(bundle.resolve()):
                raise ValueError("Statistics must live inside the bundle.")
            stats = json.loads(stats_path.read_text())
            if stats.get("dataset_name") != name or stats.get("action_space") != meta["action_space"]:
                raise ValueError(f"Statistics dataset/action-space mismatch: {stats_path}")
            if split == "train":
                meta["_episodes"] = meta["datalist"]
                train[name] = meta
        if seen != {name for name, entry in manifest["datasets"].items() if entry[split]}:
            raise ValueError(f"Missing/extra datasets in {split}.")
    HierarchicalSampler(train, sampling)
    return manifest
