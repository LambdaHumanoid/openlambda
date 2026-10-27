"""Reader for the PSI SIMPLE G1 simulation benchmark."""

from __future__ import annotations

import random
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from lambda0.data.normalization import DatasetNormalizer

from .base import DomainHandler


ACTION_DIM = 36
PROPRIO_DIM = 32
ACTION_SPACE = "robot_g1_simple_sim"
COORDINATE_CONTRACT = "psi_simple_absolute_v1"
NORMALIZATION_CONTRACT = "quantile_q01_q99_v1"


def _list_column_to_matrix(column, width: int, name: str) -> np.ndarray:
    array = column.combine_chunks()
    if array.null_count:
        raise ValueError(f"{name} contains null values.")
    values = array.values.to_numpy(zero_copy_only=False)
    if values.size != len(array) * width:
        raise ValueError(
            f"{name} must contain {width} values per row, got "
            f"{values.size} values across {len(array)} rows."
        )
    matrix = np.asarray(values, dtype=np.float32).reshape(len(array), width)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    matrix.setflags(write=False)
    return matrix


def decode_video_segment(
    path: Path,
    *,
    start_frame: int,
    length: int,
    fps: int,
) -> list[np.ndarray]:
    """Seek to and decode an exact frame interval from a concatenated video."""
    if start_frame < 0 or length <= 0 or fps <= 0:
        raise ValueError("start_frame, length, and fps must be positive.")
    container = av.open(str(path), options={"threads": "2"})
    try:
        stream = container.streams.video[0]
        if stream.average_rate is None or stream.time_base is None:
            raise ValueError(f"Video lacks timing metadata: {path}")
        stream_fps = Fraction(stream.average_rate)
        expected_fps = Fraction(fps, 1)
        if stream_fps != expected_fps:
            raise ValueError(
                f"Video FPS {stream_fps} does not match metadata FPS {expected_fps}."
            )
        start_pts = int(stream.start_time or 0)
        target_pts = start_pts + int(
            Fraction(start_frame, 1) / stream_fps / Fraction(stream.time_base)
        )
        container.seek(
            target_pts,
            stream=stream,
            backward=True,
            any_frame=False,
        )
        stop_frame = start_frame + length
        expected_index = start_frame
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Video frame has no presentation timestamp: {path}")
            frame_index = round(
                Fraction(int(frame.pts) - start_pts)
                * Fraction(stream.time_base)
                * stream_fps
            )
            if frame_index < start_frame:
                continue
            if frame_index >= stop_frame:
                break
            if frame_index != expected_index:
                raise ValueError(
                    f"Video frame sequence is discontinuous at {frame_index}; "
                    f"expected {expected_index} in {path}."
                )
            frames.append(frame.to_ndarray(format="rgb24"))
            expected_index += 1
    finally:
        container.close()
    if len(frames) != length:
        raise ValueError(
            f"Decoded {len(frames)} frames from {path}, expected {length} "
            f"at global frame {start_frame}."
        )
    return frames


class PsiSimpleHandler(DomainHandler):
    """Emit 36D G1 simulator actions and their independent 32D proprio state."""

    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)
        if num_views <= 0:
            raise ValueError("num_views must be positive.")
        self.root = Path(meta["root_path"])
        self.storage_layout = str(meta.get("storage_layout", "merged_v3"))
        if self.storage_layout not in {"merged_v3", "episode_v2"}:
            raise ValueError(
                f"Unsupported SIMPLE storage layout: {self.storage_layout!r}."
            )
        self.data_path_template = str(meta["data_path"])
        self.video_path_template = str(meta["video_path"])
        self.fps = int(meta["fps"])
        normalization_path = Path(str(meta.get("normalization_path", "")))
        if not normalization_path.is_absolute():
            normalization_path = self.root / normalization_path
        self.normalizer = DatasetNormalizer(
            normalization_path,
            str(meta["dataset_name"]),
            expected_action_space=ACTION_SPACE,
            expected_coordinate_contract=COORDINATE_CONTRACT,
            expected_normalization_contract=NORMALIZATION_CONTRACT,
            action_dim=ACTION_DIM,
            proprio_dim=PROPRIO_DIM,
        )
        self._states: np.ndarray | None = None
        self._actions: np.ndarray | None = None

    def _read_data_file(
        self,
        path: Path,
        *,
        expected_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"SIMPLE Parquet file is missing: {path}")
        table = pq.read_table(path, columns=["states", "action"], memory_map=True)
        states = _list_column_to_matrix(table["states"], PROPRIO_DIM, "states")
        actions = _list_column_to_matrix(table["action"], ACTION_DIM, "action")
        if len(states) != len(actions):
            raise ValueError(f"states and action row counts differ in {path}.")
        if len(states) != expected_frames:
            raise ValueError(
                f"Parquet contains {len(states)} rows, expected "
                f"{expected_frames}: {path}"
            )
        return states, actions

    def _load_merged_data(self) -> None:
        if self._states is not None:
            return
        states, actions = self._read_data_file(
            self.root / self.data_path_template,
            expected_frames=int(self.meta["total_frames"]),
        )
        self._states = states
        self._actions = actions

    def _episode_paths(self, episode_index: int) -> tuple[Path, Path]:
        chunks_size = int(self.meta["chunks_size"])
        if chunks_size <= 0:
            raise ValueError("chunks_size must be positive.")
        values = {
            "episode_chunk": episode_index // chunks_size,
            "episode_index": episode_index,
            "video_key": "observation.images.egocentric",
        }
        return (
            self.root / self.data_path_template.format(**values),
            self.root / self.video_path_template.format(**values),
        )

    def _load_episode(
        self,
        item: dict,
    ) -> tuple[np.ndarray, np.ndarray, Path, int]:
        length = int(item["length"])
        episode_index = int(item["episode_index"])
        if self.storage_layout == "episode_v2":
            data_path, video_path = self._episode_paths(episode_index)
            states, actions = self._read_data_file(
                data_path,
                expected_frames=length,
            )
            return states, actions, video_path, 0

        self._load_merged_data()
        if self._states is None or self._actions is None:
            raise RuntimeError("SIMPLE merged data did not initialize.")
        start = int(item["dataset_from_index"])
        stop = int(item["dataset_to_index"])
        if stop - start != length or start < 0 or stop > len(self._states):
            raise ValueError(
                f"Invalid data interval [{start},{stop}) for episode "
                f"{episode_index}."
            )
        return (
            self._states[start:stop],
            self._actions[start:stop],
            self.root / self.video_path_template,
            int(item["video_start_frame"]),
        )

    def _build_sample(
        self,
        *,
        item: dict,
        states: np.ndarray,
        actions: np.ndarray,
        frame: np.ndarray,
        index: int,
        num_actions: int,
        image_aug,
        lang_aug_map: dict[str, list[str]] | None,
    ) -> dict:
        length = len(states)
        action_indices = np.arange(index + 1, index + num_actions + 1)
        action_rows_valid = action_indices < length
        action_indices = np.minimum(action_indices, length - 1)
        action = actions[action_indices].copy()
        action_valid_mask = np.ones((num_actions, ACTION_DIM), dtype=bool)
        proprio_valid_mask = np.ones(PROPRIO_DIM, dtype=bool)
        proprio = self.normalizer.normalize_proprio(
            states[index],
            proprio_valid_mask,
        )
        action = self.normalizer.normalize_action(action, action_valid_mask)

        image = image_aug(Image.fromarray(frame))
        views = [image] + [
            torch.zeros_like(image) for _ in range(self.num_views - 1)
        ]
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[0] = True
        base_instruction = str(item["task"])
        instruction = (
            random.choice(lang_aug_map[base_instruction])
            if lang_aug_map and base_instruction in lang_aug_map
            else base_instruction
        )
        return {
            "language_instruction": instruction,
            "image_input": torch.stack(views),
            "image_mask": image_mask,
            "proprio": torch.from_numpy(proprio),
            "action": torch.from_numpy(action),
            "proprio_valid_mask": torch.from_numpy(proprio_valid_mask),
            "action_valid_mask": torch.from_numpy(action_valid_mask),
            "sample_index": index,
        }

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

    def sample_episode(
        self,
        episode_index: int,
        *,
        num_actions: int,
        image_aug,
        lang_aug_map: dict[str, list[str]] | None,
        sample_index: int | None = None,
    ) -> dict:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        item = self.meta["datalist"][episode_index]
        length = int(item["length"])
        if length <= 0:
            raise ValueError(f"Episode {item['episode_index']} is empty.")
        states, actions, video_path, video_start_frame = self._load_episode(item)
        if sample_index is None:
            index = random.randrange(length)
        else:
            index = int(sample_index)
            if not 0 <= index < length:
                raise IndexError(
                    f"Sample index {index} is outside episode range [0,{length})."
                )
        frame = decode_video_segment(
            video_path,
            start_frame=video_start_frame + index,
            length=1,
            fps=self.fps,
        )[0]
        return self._build_sample(
            item=item,
            states=states,
            actions=actions,
            frame=frame,
            index=index,
            num_actions=num_actions,
            image_aug=image_aug,
            lang_aug_map=lang_aug_map,
        )

    def iter_episode(
        self,
        episode_index: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict[str, list[str]] | None,
    ):
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        item = self.meta["datalist"][episode_index]
        length = int(item["length"])
        if length <= 0:
            raise ValueError(f"Episode {item['episode_index']} is empty.")
        states, actions, video_path, video_start_frame = self._load_episode(item)
        frames = decode_video_segment(
            video_path,
            start_frame=video_start_frame,
            length=length,
            fps=self.fps,
        )

        indices = list(range(length))
        if training:
            random.shuffle(indices)
        for index in indices:
            yield self._build_sample(
                item=item,
                states=states,
                actions=actions,
                frame=frames[index],
                index=index,
                num_actions=num_actions,
                image_aug=image_aug,
                lang_aug_map=lang_aug_map,
            )


__all__ = [
    "ACTION_SPACE",
    "COORDINATE_CONTRACT",
    "NORMALIZATION_CONTRACT",
    "PsiSimpleHandler",
    "decode_video_segment",
]
