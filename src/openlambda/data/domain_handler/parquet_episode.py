"""Random-access reader for canonical Parquet state/action episode rows."""

from __future__ import annotations

import random
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from lambda0.data.normalization import DatasetNormalizer
from lambda0.models.action_specs import get_action_space_spec

from .base import DomainHandler
from ..video import decode


class ParquetEpisodeHandler(DomainHandler):
    """Emit one strict Pico-GoPro action/state view from LeRobot rows."""

    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)
        self.spec = get_action_space_spec(meta["action_space"])
        normalization_path = meta.get("normalization_path")
        normalization_contract = meta.get("normalization_contract")
        if (normalization_path is None) != (normalization_contract is None):
            raise ValueError(
                "normalization_path and normalization_contract must be declared "
                "together."
            )
        self.normalizer = None
        if normalization_path is not None:
            path = Path(str(normalization_path))
            if not path.is_absolute():
                path = Path(str(meta["root_path"])) / path
            self.normalizer = DatasetNormalizer(
                path,
                str(meta["dataset_name"]),
                expected_action_space=self.spec.name,
                expected_coordinate_contract=str(meta["coordinate_contract"]),
                expected_normalization_contract=str(normalization_contract),
                action_dim=self.spec.action_dim,
                proprio_dim=self.spec.proprio_dim,
            )

    def _load_episode(self, traj_idx: int):
        item = self.meta["datalist"][traj_idx]
        episode_index = int(item["episode_index"])
        root = Path(self.meta["root_path"])
        chunk = episode_index // int(self.meta.get("chunks_size", 1000))
        parquet_path = root / self.meta["data_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
        )
        video_path = root / self.meta["video_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key="observation.images.front",
        )
        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state",
                "observation.valid_mask",
                "action",
                "action.valid_mask",
            ],
        ).to_pydict()
        state = np.asarray(table["observation.state"], dtype=np.float32)
        state_valid = np.asarray(table["observation.valid_mask"], dtype=bool)
        action = np.asarray(table["action"], dtype=np.float32)
        action_valid = np.asarray(table["action.valid_mask"], dtype=bool)
        frame_count = len(state)
        if state.ndim != 2 or state.shape[1] != self.spec.proprio_dim:
            raise ValueError(
                f"{self.spec.name} expects {self.spec.proprio_dim}D proprio, "
                f"got {state.shape}."
            )
        if action.ndim != 2 or action.shape[1] != self.spec.action_dim:
            raise ValueError(
                f"{self.spec.name} expects {self.spec.action_dim}D action, "
                f"got {action.shape}."
            )
        if state_valid.shape != state.shape or action_valid.shape != action.shape:
            raise ValueError("Pico-GoPro values and validity masks have different shapes.")
        if len(state_valid) != frame_count or len(action) != frame_count:
            raise ValueError("Pico-GoPro state and action row counts differ.")
        if len(action_valid) != frame_count:
            raise ValueError("Pico-GoPro action and validity-mask row counts differ.")
        return item, video_path, state, state_valid, action, action_valid

    def _instruction(
        self,
        item: dict,
        lang_aug_map: dict[str, list[str]] | None,
    ) -> str:
        task = str(
            item.get("tasks", ["Perform the observed full-body activity."])[0]
        )
        return (
            random.choice(lang_aug_map[task])
            if lang_aug_map and task in lang_aug_map
            else task
        )

    def candidate_timestep_count(
        self,
        episode_index: int,
        *,
        num_actions: int,
    ) -> int:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        length = int(self.meta["datalist"][episode_index]["length"])
        return max(0, length)

    def _build_sample(
        self,
        *,
        index: int,
        frame: np.ndarray,
        state: np.ndarray,
        state_valid: np.ndarray,
        action: np.ndarray,
        action_valid: np.ndarray,
        instruction: str,
        num_actions: int,
        image_aug,
    ) -> dict | None:
        frame_count = len(state)
        future_indices = np.arange(
            index + 1,
            index + num_actions + 1,
            dtype=np.int64,
        )
        in_episode = future_indices < frame_count
        future_indices = np.minimum(future_indices, frame_count - 1)
        target_valid = action_valid[future_indices].copy()
        if not target_valid.any() or not state_valid[index].any():
            return None
        image = image_aug(Image.fromarray(frame))
        views = [image] + [
            torch.zeros_like(image) for _ in range(max(0, self.num_views - 1))
        ]
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[0] = True
        proprio = state[index].copy()
        target_action = action[future_indices].copy()
        proprio_valid = state_valid[index].copy()
        if self.normalizer is not None:
            proprio = self.normalizer.normalize_proprio(
                proprio,
                proprio_valid,
            )
            target_action = self.normalizer.normalize_action(
                target_action,
                target_valid,
            )
        return {
            "language_instruction": instruction,
            "image_input": torch.stack(views),
            "image_mask": image_mask,
            "proprio": torch.from_numpy(proprio),
            "proprio_valid_mask": torch.from_numpy(proprio_valid),
            "action": torch.from_numpy(target_action),
            "action_valid_mask": torch.from_numpy(target_valid),
            "sample_index": index,
        }

    def sample_episode(
        self,
        episode_index: int,
        *,
        num_actions: int,
        image_aug,
        lang_aug_map: dict[str, list[str]] | None,
        sample_index: int | None = None,
    ) -> dict | None:
        loaded = self._load_episode(episode_index)
        item, video_path, state, state_valid, action, action_valid = loaded
        if len(state) < 1:
            return None
        instruction = self._instruction(item, lang_aug_map)
        if sample_index is None:
            indices = list(range(len(state)))
            random.shuffle(indices)
        else:
            index = int(sample_index)
            if not 0 <= index < len(state):
                raise IndexError(
                    f"Sample index {index} is outside episode range "
                    f"[0,{len(state) - 1})."
                )
            indices = [index]
        for index in indices:
            future = np.arange(index + 1, index + num_actions + 1)
            in_episode = future < len(state)
            future = np.minimum(future, len(state) - 1)
            valid = action_valid[future].copy()
            if valid.any() and state_valid[index].any():
                frame = decode(video_path, [index])[0]
                return self._build_sample(
                    index=index,
                    frame=frame,
                    state=state,
                    state_valid=state_valid,
                    action=action,
                    action_valid=action_valid,
                    instruction=instruction,
                    num_actions=num_actions,
                    image_aug=image_aug,
                )
        return None

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict[str, list[str]] | None,
        **kwargs,
    ):
        loaded = self._load_episode(traj_idx)
        item, video_path, state, state_valid, action, action_valid = loaded
        frame_count = len(state)
        if frame_count < 1:
            return
        instruction = self._instruction(item, lang_aug_map)
        container = av.open(str(video_path), options={"threads": "2"})
        decoded = 0
        try:
            for index, frame in enumerate(container.decode(video=0)):
                decoded = index + 1
                if index >= frame_count:
                    break
                sample = self._build_sample(
                    index=index,
                    frame=frame.to_ndarray(format="rgb24"),
                    state=state,
                    state_valid=state_valid,
                    action=action,
                    action_valid=action_valid,
                    instruction=instruction,
                    num_actions=num_actions,
                    image_aug=image_aug,
                )
                if sample is not None:
                    yield sample
        finally:
            container.close()
        if decoded < frame_count:
            raise ValueError(
                f"Video ended at frame {decoded}, expected {frame_count}: {video_path}"
            )

    def iter_validation_episode(
        self,
        episode_index: int,
        *,
        num_actions: int,
        samples_per_episode: int,
        image_aug,
        lang_aug_map: dict[str, list[str]] | None,
    ):
        """Yield evenly spaced valid timesteps without validation randomness."""
        if samples_per_episode <= 0:
            raise ValueError("samples_per_episode must be positive.")
        loaded = self._load_episode(episode_index)
        item, video_path, state, state_valid, action, action_valid = loaded
        candidates = []
        for index in range(len(state)):
            future = np.arange(index + 1, index + num_actions + 1)
            in_episode = future < len(state)
            future = np.minimum(future, len(state) - 1)
            valid = action_valid[future].copy()
            if valid.any() and state_valid[index].any():
                candidates.append(index)
        if not candidates:
            return
        sample_count = min(samples_per_episode, len(candidates))
        offsets = np.minimum(
            (
                (np.arange(sample_count, dtype=np.float64) + 0.5)
                * len(candidates)
                / sample_count
            ).astype(np.int64),
            len(candidates) - 1,
        )
        indices = [candidates[int(offset)] for offset in offsets]
        frames = decode(video_path, indices)
        instruction = str(
            item.get("tasks", ["Perform the observed full-body activity."])[0]
        )
        for index, frame in zip(indices, frames, strict=True):
            sample = self._build_sample(
                index=index,
                frame=frame,
                state=state,
                state_valid=state_valid,
                action=action,
                action_valid=action_valid,
                instruction=instruction,
                num_actions=num_actions,
                image_aug=image_aug,
            )
            if sample is None:
                raise RuntimeError(
                    f"Selected validation timestep became invalid: {index}"
                )
            yield sample


__all__ = ["ParquetEpisodeHandler"]
