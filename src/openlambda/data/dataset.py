# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import io
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
from mmengine import fileio
from torch.utils.data import IterableDataset, get_worker_info
from .image_augmentation import ImageAugmentation

from lambda0.models.action_specs import get_action_space_spec

from .action_packing import slice_action_trajectory
from .domain_handler.registry import get_handler_cls
from .episode_index import EpisodeIndex
from .hierarchical_sampling import HierarchicalSampler, SampleRequest


logger = logging.getLogger(__name__)


class InfiniteDataReader(IterableDataset):
    """
    Output sample:
      {
        'embodiment_id': LongTensor[], # embodiment id for conditioning
        'action_space_id': LongTensor[], # action-space contract id
        'language_instruction': str,
        'image_input': FloatTensor[V, C, H, W],
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action]
      }
    """

    def __init__(
        self,
        metas_path: str,
        num_actions: int = 50,
        num_views: int = 3,
        training: bool = True,
        dataset_names: list[str] | None = None,
        sample_weights: dict[str, float] | None = None,
        sampling_plan: Mapping[str, Any] | None = None,
        sampling_seed: int = 0,
        validation_samples_per_episode: int | None = None,
        shard_index: int = 0,
        num_shards: int = 1,
        data_root: str | None = None,
        image_max_pixels: int | None = None,
    ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.sampling_seed = int(sampling_seed)
        if validation_samples_per_episode is not None and training:
            raise ValueError(
                "validation_samples_per_episode requires training=False."
            )
        if (
            validation_samples_per_episode is not None
            and validation_samples_per_episode <= 0
        ):
            raise ValueError(
                "validation_samples_per_episode must be positive."
            )
        self.validation_samples_per_episode = validation_samples_per_episode
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError("shard_index must be in [0,num_shards).")
        self.shard_index = int(shard_index)
        self.num_shards = int(num_shards)
        self.metas: Dict[str, dict] = {}
        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(
                metas_path, suffix=".json", recursive=True, list_dir=False
            )
            root = metas_path
        else:
            meta_files, root = [metas_path], ""
        if not meta_files:
            raise FileNotFoundError(f"No metadata JSON files found at {metas_path}.")

        requested_names = set(dataset_names or ())
        if len(requested_names) != len(dataset_names or ()):
            raise ValueError("dataset_names must not contain duplicates.")
        if requested_names:
            meta_files = [
                file for file in meta_files if Path(str(file)).stem in requested_names
            ]
            if not meta_files:
                raise FileNotFoundError(
                    f"No requested metadata files found at {metas_path}."
                )
        meta_files = sorted(meta_files)
        for file in meta_files:
            file_path = fileio.join_path(root, file)
            with io.BytesIO(fileio.get(file_path)) as f:
                meta = json.load(f)
            from lambda0.runtime.paths import metadata_paths
            meta = metadata_paths(meta, file_path, data_root)
            self._register_meta(meta, file)

        missing_names = requested_names.difference(self.metas)
        if missing_names:
            raise KeyError(
                "Requested datasets were not registered: "
                + ", ".join(sorted(missing_names))
            )
        weight_overrides = sample_weights or {}
        unknown_weights = set(weight_overrides).difference(self.metas)
        if unknown_weights:
            raise KeyError(
                "Sampling weights reference unloaded datasets: "
                + ", ".join(sorted(unknown_weights))
            )
        for dataset_name, weight in weight_overrides.items():
            weight = float(weight)
            if weight <= 0:
                raise ValueError(
                    f"Sampling weight for {dataset_name!r} must be positive."
                )
            self.metas[dataset_name]["sample_weight"] = weight
        self.sampler = HierarchicalSampler(self.metas, sampling_plan)

        self.image_aug = ImageAugmentation(training=self.training, max_pixels=image_max_pixels)
        logger.info("Image augmentation: %s", "pi05-style" if self.training else "disabled (validation)")

    def _register_meta(self, meta: dict, source: str) -> None:
        required = {
            "dataset_name",
            "handler",
            "embodiment_id",
            "action_space",
            "root_path",
        }
        missing = required.difference(meta)
        if missing:
            raise KeyError(
                f"Metadata {source} is missing: {', '.join(sorted(missing))}."
            )
        dataset_name = str(meta["dataset_name"])
        if dataset_name in self.metas:
            raise ValueError(f"Duplicate dataset_name {dataset_name!r}.")
        has_inline = "datalist" in meta
        has_compact = "episode_index_path" in meta
        if has_inline == has_compact:
            raise ValueError(
                f"Dataset {dataset_name!r} must define exactly one of "
                "datalist or episode_index_path."
            )
        if has_inline:
            episodes = meta["datalist"]
            if not isinstance(episodes, list) or not episodes:
                raise ValueError(
                    f"Dataset {dataset_name!r} must contain a non-empty datalist."
                )
        else:
            index_path = Path(str(meta["episode_index_path"]))
            if not index_path.is_absolute():
                index_path = Path(str(meta["root_path"])) / index_path
            episodes = EpisodeIndex(index_path)
            expected_count = int(meta.get("episode_count", len(episodes)))
            if len(episodes) != expected_count:
                raise ValueError(
                    f"Dataset {dataset_name!r} episode index contains "
                    f"{len(episodes)} rows, expected {expected_count}."
                )
        meta["_episodes"] = episodes
        get_handler_cls(meta["handler"])
        meta["embodiment_id"] = int(meta["embodiment_id"])
        if meta["embodiment_id"] < 0:
            raise ValueError("embodiment_id must be non-negative.")
        meta["action_space_id"] = get_action_space_spec(
            meta["action_space"]
        ).id
        meta["sample_weight"] = float(meta.get("sample_weight", 1.0))
        if meta["sample_weight"] <= 0:
            raise ValueError("sample_weight must be positive.")
        self.metas[dataset_name] = meta
        logger.info(
            "Dataset %s: %d episodes, action_space_id=%d, embodiment_id=%d",
            dataset_name,
            len(meta["_episodes"]),
            meta["action_space_id"],
            meta["embodiment_id"],
        )

    def _finalize_sample(
        self,
        dataset_name: str,
        sample: dict,
        request: SampleRequest | None = None,
    ) -> dict:
        meta = self.metas[dataset_name]
        if "embodiment_id" not in sample:
            sample["embodiment_id"] = torch.tensor(
                meta["embodiment_id"],
                dtype=torch.long,
            )
        else:
            sample["embodiment_id"] = torch.as_tensor(
                sample["embodiment_id"], dtype=torch.long
            )
        if "dataset_id" not in sample and "dataset_id" in meta:
            sample["dataset_id"] = torch.as_tensor(
                meta["dataset_id"], dtype=torch.long
            )
        if "action_space_id" not in sample:
            sample["action_space_id"] = torch.tensor(
                meta["action_space_id"],
                dtype=torch.long,
            )
        else:
            sample_space = get_action_space_spec(
                int(torch.as_tensor(sample["action_space_id"]).item())
            )
            if sample_space.id != meta["action_space_id"]:
                raise ValueError(
                    f"Dataset {dataset_name!r} emitted action_space_id "
                    f"{sample_space.id}, expected {meta['action_space_id']}."
                )
            sample["action_space_id"] = torch.tensor(
                sample_space.id,
                dtype=torch.long,
            )
        idx_for_delta = sample.pop("idx_for_delta", [])
        idx_for_mask_proprio = sample.pop("idx_for_mask_proprio", [])
        abs_trajectory = sample.pop("abs_trajectory", None)
        if abs_trajectory is not None:
            sample.update(
                slice_action_trajectory(
                    abs_trajectory,
                    idx_for_delta,
                    idx_for_mask_proprio,
                )
            )
        elif "action" not in sample or "proprio" not in sample:
            raise KeyError(
                "A sample must provide abs_trajectory or both action and proprio tensors."
            )
        if request is not None:
            if request.timestep_index is not None:
                actual_index = int(sample.get("sample_index", -1))
                if actual_index != request.timestep_index:
                    raise RuntimeError(
                        f"Handler returned timestep {actual_index}, expected "
                        f"{request.timestep_index}."
                    )
            sample["sampling_group"] = request.group_name
            sample["dataset_name"] = request.dataset_name
            sample["task_name"] = request.task_name
            sample["episode_index"] = request.episode_index
        sample["dataset_name"] = dataset_name
        return sample

    def _iter_dataset_epoch(self, dataset_name: str) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        episode_count = len(meta["_episodes"])

        handler_cls = get_handler_cls(meta["handler"])
        handler = handler_cls(meta=meta, num_views=self.num_views)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        combined_shard = self.shard_index + self.num_shards * worker_id
        combined_shards = self.num_shards * worker_count
        for traj_idx in range(episode_count):
            if traj_idx % combined_shards != combined_shard:
                continue
            if self.validation_samples_per_episode is None:
                samples = handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),
                )
            else:
                samples = handler.iter_validation_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    samples_per_episode=self.validation_samples_per_episode,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),
                )
            for sample in samples:
                yield self._finalize_sample(dataset_name, sample)

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        while True:
            yield from self._iter_dataset_epoch(dataset_name)
            if not self.training:
                return

    def __iter__(self):
        names = list(self.metas.keys())
        if not names:
            raise RuntimeError("No datasets were loaded.")
        if not self.training:
            for name in names:
                yield from self._iter_one_dataset(name)
        else:
            handlers = {
                name: get_handler_cls(meta["handler"])(
                    meta=meta,
                    num_views=self.num_views,
                )
                for name, meta in self.metas.items()
            }
            worker = get_worker_info()
            worker_id = 0 if worker is None else worker.id
            worker_count = 1 if worker is None else worker.num_workers
            combined_shard = self.shard_index + self.num_shards * worker_id
            combined_shards = self.num_shards * worker_count
            empty_requests = 0
            if self.sampler.uses_epoch_sampling:
                epoch = 0
                while True:
                    requests = self.sampler.epoch_requests(
                        lambda dataset_name, episode_index: handlers[
                            dataset_name
                        ].candidate_timestep_count(
                            episode_index,
                            num_actions=self.num_actions,
                        ),
                        seed=self.sampling_seed,
                        epoch=epoch,
                    )
                    if len(requests) < combined_shards:
                        raise RuntimeError(
                            f"Epoch contains {len(requests)} samples for "
                            f"{combined_shards} rank-worker shards."
                        )
                    emitted = 0
                    for position in range(
                        combined_shard,
                        len(requests),
                        combined_shards,
                    ):
                        request = requests[position]
                        meta = self.metas[request.dataset_name]
                        sample = handlers[request.dataset_name].sample_episode(
                            request.episode_index,
                            num_actions=self.num_actions,
                            image_aug=self.image_aug,
                            lang_aug_map=meta.get("lang_aug_map"),
                            sample_index=request.timestep_index,
                        )
                        if sample is None:
                            empty_requests += 1
                            if empty_requests >= 1000:
                                raise RuntimeError(
                                    "Epoch sampling produced 1000 consecutive empty "
                                    "requests. Validate action masks."
                                )
                            continue
                        empty_requests = 0
                        emitted += 1
                        yield self._finalize_sample(
                            request.dataset_name,
                            sample,
                            request,
                        )
                    if emitted == 0:
                        raise RuntimeError(
                            f"Rank-worker shard {combined_shard} emitted no samples "
                            f"for epoch {epoch}."
                        )
                    epoch += 1

            while True:
                request = self.sampler.sample()
                meta = self.metas[request.dataset_name]
                sample = handlers[request.dataset_name].sample_episode(
                    request.episode_index,
                    num_actions=self.num_actions,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),
                    sample_index=request.timestep_index,
                )
                if sample is None:
                    empty_requests += 1
                    if empty_requests >= 1000:
                        raise RuntimeError(
                            "Hierarchical sampling produced 1000 consecutive empty "
                            "episode requests. Validate episode lengths and action masks."
                        )
                    continue
                empty_requests = 0
                yield self._finalize_sample(
                    request.dataset_name,
                    sample,
                    request,
                )
