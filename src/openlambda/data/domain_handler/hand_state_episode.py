"""Canonical egocentric hand-state episodes for VLM-only Stage I.

The source Parquet contains observation.state and observation.valid_mask,
both [frames, 138]. Future targets are read from the same state stream;
no action labels or action expert are involved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .parquet_episode import ParquetEpisodeHandler


class HandStateEpisodeHandler(ParquetEpisodeHandler):
    """Emit current and future hand states through the common sample schema."""

    def __init__(self, meta: dict, num_views: int) -> None:
        if meta.get("action_space") != "human_hand_geometry":
            raise ValueError("Stage-I hand episodes require human_hand_geometry (138D).")
        super().__init__(meta, num_views)

    def _load_episode(self, traj_idx: int):
        item = self.meta["datalist"][traj_idx]
        episode_index = int(item["episode_index"])
        root = Path(self.meta["root_path"])
        chunk = episode_index // int(self.meta.get("chunks_size", 1000))
        data_path = root / self.meta["data_path"].format(
            episode_chunk=chunk, episode_index=episode_index
        )
        video_path = root / self.meta["video_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key="observation.images.front",
        )
        table = pq.read_table(
            data_path, columns=["observation.state", "observation.valid_mask"]
        ).to_pydict()
        state = np.asarray(table["observation.state"], dtype=np.float32)
        valid = np.asarray(table["observation.valid_mask"], dtype=bool)
        if state.ndim != 2 or state.shape != (int(item["length"]), 138):
            raise ValueError(f"Expected [frames,138] hand states: {data_path}")
        if valid.shape != state.shape or not np.isfinite(state[valid]).all():
            raise ValueError(f"Invalid hand-state values or masks: {data_path}")
        # The shared normalizer checks every input value, including masked ones.
        # A missing coordinate has no target token and must not carry NaN into it.
        state = np.where(valid, state, 0.0)
        return item, video_path, state, valid, state, valid


__all__ = ["HandStateEpisodeHandler"]
