#!/usr/bin/env python3
"""Convert raw Pico LeRobot episodes into SONIC/GR00T LeRobot episodes.

This is the offline equivalent of the replay path:

    raw Pico body pose -> processed SMPL pose -> SONIC encoder -> motion token
    -> SONIC decoder -> 29-DOF G1 policy action/state -> GR00T dataset schema

The live replay script sends pose messages to the SONIC C++ deploy process.  This
script runs the same encoder/decoder ONNX models directly so a raw dataset can be
processed without starting the realtime ZMQ stack.
"""

from __future__ import annotations
from egoalign.runtime import asset_path
from egoalign.alignment import parse_alignment
from egoalign.io_safety import validate_output_root

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import datasets
import numpy as np
import onnxruntime as ort
import pandas as pd
import av
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egoalign.datasets.exporter import Gr00tDataExporter  # noqa: E402
from egoalign.datasets.features import (  # noqa: E402
    FPS,
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
)
from egoalign.robots.g1.hand_mapping import (  # noqa: E402
    retarget_openxr_hand_to_brainco,
)
from lerobot.common.datasets.lerobot_dataset import (  # noqa: E402
    compute_episode_stats,
    get_hf_features_from_features,
)
from lerobot.common.datasets.utils import (  # noqa: E402
    write_episode,
    write_episode_stats,
    write_info,
)
from scripts.human_data.processing.body_transforms import (  # noqa: E402
    compute_projected_gravity,
    compute_smpl_from_body_pose,
    compute_vr_3pt,
    compute_wrist_joints,
    quat_to_rot6d,
)
from scripts.human_data.processing.encode_motion_tokens import (  # noqa: E402
    ENCODER_OUTPUT_DIM,
    build_encoder_observations,
)


RAW_BODY_COLUMN = "observation.pico.body_pose"
LEFT_HAND_COLUMN = "observation.pico.left_hand_pose"
RIGHT_HAND_COLUMN = "observation.pico.right_hand_pose"
LEFT_HAND_ACTIVE_COLUMN = "observation.pico.left_hand_active"
RIGHT_HAND_ACTIVE_COLUMN = "observation.pico.right_hand_active"
LEFT_CONTROLLER_COLUMN = "observation.pico.left_controller_pose"
RIGHT_CONTROLLER_COLUMN = "observation.pico.right_controller_pose"

HISTORY_LEN = 10
POLICY_OBS_DIM = 994
POLICY_ACTION_DIM = 29
HAND_DOF_DATASET = 6
HAND_DOF_ROBOT = 7
DEFAULT_CHUNKS_SIZE = 1000
DEFAULT_OUTPUT_VIDEO_WIDTH = 1280
DEFAULT_OUTPUT_VIDEO_HEIGHT = 720

INDEX_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}

_WORKER_ENCODER_SESSION: ort.InferenceSession | None = None
_WORKER_DECODER_SESSION: ort.InferenceSession | None = None
_WORKER_ROBOT_MODEL = None
_WORKER_FEATURES: dict | None = None
_WORKER_HF_FEATURES: datasets.Features | None = None
_WORKER_NON_VIDEO_FEATURES: dict | None = None
_WORKER_OUTPUT_ROOT: Path | None = None
_WORKER_TASK: str | None = None
_WORKER_DECODER_HISTORY_MODE: str | None = None
_WORKER_MAX_FRAMES_PER_EPISODE: int | None = None
_WORKER_CHUNKS_SIZE: int = DEFAULT_CHUNKS_SIZE
_WORKER_VIDEO_WIDTH: int = DEFAULT_OUTPUT_VIDEO_WIDTH
_WORKER_VIDEO_HEIGHT: int = DEFAULT_OUTPUT_VIDEO_HEIGHT

ISAACLAB_TO_MUJOCO = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)
DEFAULT_ANGLES_MUJOCO = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def _g1_action_scale() -> np.ndarray:
    natural_freq = 10.0 * 2.0 * math.pi
    stiffness_5020 = 0.003609725 * natural_freq * natural_freq
    stiffness_7520_14 = 0.010177520 * natural_freq * natural_freq
    stiffness_7520_22 = 0.025101925 * natural_freq * natural_freq
    stiffness_4010 = 0.00425 * natural_freq * natural_freq

    scale_5020 = 0.25 * 25.0 / stiffness_5020
    scale_7520_14 = 0.25 * 88.0 / stiffness_7520_14
    scale_7520_22 = 0.25 * 139.0 / stiffness_7520_22
    scale_4010 = 0.25 * 5.0 / stiffness_4010

    return np.array(
        [
            scale_7520_22,
            scale_7520_22,
            scale_7520_14,
            scale_7520_22,
            scale_5020,
            scale_5020,
            scale_7520_22,
            scale_7520_22,
            scale_7520_14,
            scale_7520_22,
            scale_5020,
            scale_5020,
            scale_7520_14,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_4010,
            scale_4010,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_4010,
            scale_4010,
        ],
        dtype=np.float32,
    )


G1_ACTION_SCALE_MUJOCO = _g1_action_scale()


@dataclass(frozen=True)
class EpisodePaths:
    episode_index: int
    parquet_path: Path
    video_paths: dict[str, Path]
    video_time_offsets_s: dict[str, float]
    pico_time_interval_s: tuple[float, float] | None
    alignment_paths: dict[str, Path]


@dataclass
class ProcessedEpisode:
    smpl_joints: np.ndarray
    smpl_pose: np.ndarray
    body_quat_w: np.ndarray
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    target_body_orientation: np.ndarray
    vr_3pt_position: np.ndarray
    vr_3pt_orientation: np.ndarray
    projected_gravity: np.ndarray
    motion_token: np.ndarray
    policy_action: np.ndarray
    body_q: np.ndarray
    body_dq: np.ndarray
    left_hand_7: np.ndarray
    right_hand_7: np.ndarray
    left_hand_6: np.ndarray
    right_hand_6: np.ndarray
    eef_state: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stage_a_source_mapping(
    *,
    source_root: Path,
    output_root: Path,
    episodes: list[EpisodePaths],
    alignment_mode: str,
    max_frames_per_episode: int | None,
) -> Path:
    """Bind Stage-A output episodes to immutable raw source episodes.

    The endpoint minimal-A/B contract must not reconstruct this identity from
    row counts or camera alignment after the fact.  Ordered output parquets are
    paired with the ordered conversion request and every pair is hash-bound.
    """

    output_parquets = sorted((output_root / "data").rglob("episode_*.parquet"))
    if len(output_parquets) != len(episodes):
        raise RuntimeError(
            "Stage-A source mapping cannot be closed: "
            f"episodes={len(episodes)}, output_parquets={len(output_parquets)}"
        )
    records = []
    for episode, output_parquet in zip(episodes, output_parquets, strict=True):
        try:
            output_episode = int(output_parquet.stem.removeprefix("episode_"))
        except ValueError as exc:
            raise RuntimeError(f"Invalid Stage-A parquet name: {output_parquet}") from exc
        raw_rows = len(pd.read_parquet(episode.parquet_path, columns=["timestamp"]))
        output_rows = len(pd.read_parquet(output_parquet, columns=["timestamp"]))
        records.append(
            {
                "baseline_episode": output_episode,
                "source_episode": int(episode.episode_index),
                "raw_parquet": str(episode.parquet_path.resolve()),
                "raw_parquet_sha256": _sha256_file(episode.parquet_path),
                "raw_rows": raw_rows,
                "baseline_parquet": str(output_parquet.resolve()),
                "baseline_parquet_sha256": _sha256_file(output_parquet),
                "baseline_rows": output_rows,
            }
        )
    path = output_root / "meta/stage_a_source_mapping.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "contract": "stage_a_source_episode_mapping",
                "source_dataset": str(source_root.resolve()),
                "alignment_mode": alignment_mode,
                "max_frames_per_episode": max_frames_per_episode,
                "episodes": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class SyncVideoWriter:
    """Synchronous PyAV writer for process-parallel episode conversion."""

    def __init__(self, output_path: Path, width: int, height: int, fps: int, codec: str = "h264"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.container = av.open(str(output_path), mode="w")
        self.stream = self.container.add_stream(
            codec,
            rate=fps,
            options={
                "preset": "ultrafast",
                "crf": "23",
                "tune": "zerolatency",
            },
        )
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self._closed = False

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[:2] != (self.stream.height, self.stream.width):
            raise ValueError(
                f"Incorrect frame shape {frame_bgr.shape}; expected "
                f"{self.stream.height}x{self.stream.width}"
            )
        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def stop(self) -> None:
        if self._closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self._closed = True

    def cancel(self) -> None:
        if not self._closed:
            try:
                self.container.close()
            except Exception:
                pass
            self._closed = True
        try:
            self.output_path.unlink()
        except FileNotFoundError:
            pass


def parse_episode_selector(selector: str | None) -> set[int] | None:
    if not selector:
        return None
    episodes: set[int] = set()
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            episodes.update(range(start, end + 1))
        else:
            episodes.add(int(part))
    return episodes


def load_first_task(source_root: Path) -> str:
    tasks_path = source_root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return "task_test"
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)["task"]
    return "task_test"


def load_info(source_root: Path) -> dict:
    with (source_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _camera_name_from_source_video_key(video_key: str) -> str:
    prefix = "observation.images."
    if video_key.startswith(prefix) and len(video_key) > len(prefix):
        return video_key[len(prefix) :]
    return video_key.replace("/", "-").replace(".", "-")


def _source_video_specs(video_template: str | dict[str, str]) -> dict[str, tuple[str, str]]:
    """Map output video features to source templates and alignment camera names."""

    if isinstance(video_template, dict):
        specs: dict[str, tuple[str, str]] = {}
        known = {
            "observation.images.gopro": ("observation.images.ego_view", "gopro"),
            "observation.images.level-cam": ("observation.images.level_cam", "level-cam"),
        }
        for source_key, (output_key, camera_name) in known.items():
            template = video_template.get(source_key)
            if isinstance(template, str):
                specs[output_key] = (template, camera_name)
        if not specs and video_template:
            source_key, template = next(iter(video_template.items()))
            if not isinstance(template, str):
                raise ValueError("meta/info.json video_path mapping values must be strings")
            specs["observation.images.ego_view"] = (
                template,
                _camera_name_from_source_video_key(source_key),
            )
        return specs
    if isinstance(video_template, str):
        return {"observation.images.ego_view": (video_template, "gopro")}
    raise ValueError("meta/info.json video_path must be a string or contain a video path mapping")


def _load_alignment(
    alignment_path: Path,
    *,
    required: bool,
) -> tuple[float, float | None] | None:
    """Read an alignment result whose offset maps camera time onto Pico time."""

    if not alignment_path.exists():
        if required:
            raise FileNotFoundError(f"Missing camera alignment: {alignment_path}")
        return None
    try:
        payload = json.loads(alignment_path.read_text(encoding="utf-8"))
        offset_s, duration_s = parse_alignment(payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid camera alignment: {alignment_path}") from exc
    return offset_s, duration_s


def discover_episodes(
    source_root: Path,
    selector: set[int] | None,
    *,
    alignment_dir: Path | None = None,
    require_alignment: bool = False,
) -> list[EpisodePaths]:
    info = load_info(source_root)
    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/observation.images.gopro/episode_{episode_index:06d}.mp4",
    )
    source_video_specs = _source_video_specs(video_template)
    total_episodes = int(info.get("total_episodes", 0))
    candidates = selector if selector is not None else set(range(total_episodes))

    episodes: list[EpisodePaths] = []
    for ep_idx in sorted(candidates):
        chunk = episode_chunk(ep_idx, chunks_size)
        parquet_path = source_root / data_template.format(
            episode_chunk=chunk,
            episode_index=ep_idx,
        )
        video_paths = {
            key: source_root
            / template.format(episode_chunk=chunk, episode_index=ep_idx)
            for key, (template, _camera_name) in source_video_specs.items()
        }
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet for episode {ep_idx}: {parquet_path}")
        for key, video_path in video_paths.items():
            if not video_path.exists():
                raise FileNotFoundError(
                    f"Missing {key} video for episode {ep_idx}: {video_path}"
                )
        video_time_offsets_s: dict[str, float] = {}
        alignment_paths: dict[str, Path] = {}
        pico_intervals: list[tuple[float, float]] = []
        for output_key, (_template, camera_name) in source_video_specs.items():
            if alignment_dir is None:
                video_time_offsets_s[output_key] = 0.0
                continue
            alignment_path = alignment_dir / (
                f"episode_{ep_idx:06d}.{camera_name}_alignment.json"
            )
            alignment = _load_alignment(alignment_path, required=require_alignment)
            if alignment is None:
                video_time_offsets_s[output_key] = 0.0
                continue
            offset_s, duration_s = alignment
            video_time_offsets_s[output_key] = offset_s
            alignment_paths[output_key] = alignment_path
            if duration_s is not None:
                pico_intervals.append((offset_s, offset_s + duration_s))
        pico_time_interval_s = None
        if pico_intervals:
            pico_start_s = max(interval[0] for interval in pico_intervals)
            pico_end_s = min(interval[1] for interval in pico_intervals)
            if pico_end_s <= pico_start_s:
                raise ValueError(
                    f"Camera videos have no common Pico-time interval for episode {ep_idx}"
                )
            pico_time_interval_s = (pico_start_s, pico_end_s)
        episodes.append(
            EpisodePaths(
                ep_idx,
                parquet_path,
                video_paths,
                video_time_offsets_s,
                pico_time_interval_s,
                alignment_paths,
            )
        )
    return episodes


def array_cell(value, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.stack([np.asarray(item, dtype=dtype) for item in arr])
    else:
        arr = arr.astype(dtype, copy=False)
    if arr.shape != shape:
        arr = arr.reshape(shape)
    return np.ascontiguousarray(arr, dtype=dtype)


def _episode_pico_timestamps(df: pd.DataFrame) -> np.ndarray:
    timestamps = (
        df["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in df.columns
        else np.arange(len(df), dtype=np.float64) / FPS
    )
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0.0):
        raise ValueError("Episode timestamps must be finite and monotonic")
    return timestamps


def load_aligned_episode_dataframe(
    ep: EpisodePaths,
    max_frames_per_episode: int | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load one episode and retain only the common camera/Pico time interval.

    ``source_pico_timestamps`` stays on the original Pico time axis for camera
    lookup.  ``output_timestamps`` is reset to zero for the new output episode.
    """

    df = pd.read_parquet(ep.parquet_path).reset_index(drop=True)
    source_pico_timestamps = _episode_pico_timestamps(df)
    if ep.pico_time_interval_s is not None:
        start_s, end_s = ep.pico_time_interval_s
        keep = (source_pico_timestamps >= start_s) & (source_pico_timestamps <= end_s)
        df = df.loc[keep].reset_index(drop=True)
        source_pico_timestamps = source_pico_timestamps[keep]
    if max_frames_per_episode is not None:
        df = df.iloc[:max_frames_per_episode].reset_index(drop=True)
        source_pico_timestamps = source_pico_timestamps[:max_frames_per_episode]
    if df.empty:
        interval = ep.pico_time_interval_s
        raise ValueError(
            f"Episode {ep.episode_index} has no frames in the aligned camera interval {interval}"
        )
    output_timestamps = source_pico_timestamps - source_pico_timestamps[0]
    return df, source_pico_timestamps, output_timestamps


def camera_timestamps_for_pico(
    ep: EpisodePaths,
    video_key: str,
    pico_timestamps: np.ndarray,
) -> np.ndarray:
    """Convert Pico timestamps to a camera's native MP4 time axis."""

    return pico_timestamps - float(ep.video_time_offsets_s.get(video_key, 0.0))


def controller_pose_to_eef_part(value) -> np.ndarray | None:
    pose = np.asarray(value, dtype=np.float32).reshape(-1)
    if pose.size < 7 or not np.isfinite(pose[:7]).all():
        return None
    if float(np.linalg.norm(pose[:7])) < 1e-6:
        return None
    pos = pose[:3]
    quat_wxyz = pose[[6, 3, 4, 5]]
    return np.concatenate([pos, quat_wxyz]).astype(np.float64)


def retarget_hand(row, hand_col: str, active_col: str) -> tuple[np.ndarray, np.ndarray]:
    hand_7 = np.zeros(HAND_DOF_ROBOT, dtype=np.float32)
    if active_col in row and bool(row[active_col]):
        try:
            hand_pose = array_cell(row[hand_col], (26, 7), dtype=np.float32)
            hand_7 = retarget_openxr_hand_to_brainco(hand_pose).astype(np.float32)
        except Exception:
            hand_7 = np.zeros(HAND_DOF_ROBOT, dtype=np.float32)
    return hand_7, hand_7[:HAND_DOF_DATASET].copy()


def compute_body_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n_frames = len(df)
    smpl_joints = np.zeros((n_frames, 72), dtype=np.float32)
    smpl_pose = np.zeros((n_frames, 63), dtype=np.float32)
    body_quat_w = np.zeros((n_frames, 4), dtype=np.float32)
    body_quat_w[:, 0] = 1.0
    left_wrist = np.zeros((n_frames, 3), dtype=np.float32)
    right_wrist = np.zeros((n_frames, 3), dtype=np.float32)
    vr_3pt_position = np.zeros((n_frames, 9), dtype=np.float32)
    vr_3pt_orientation = np.zeros((n_frames, 18), dtype=np.float32)
    target_body_orientation = np.zeros((n_frames, 6), dtype=np.float32)
    projected_gravity = np.zeros((n_frames, 3), dtype=np.float32)
    projected_gravity[:, 2] = -1.0
    left_hand_7 = np.zeros((n_frames, HAND_DOF_ROBOT), dtype=np.float32)
    right_hand_7 = np.zeros((n_frames, HAND_DOF_ROBOT), dtype=np.float32)
    left_hand_6 = np.zeros((n_frames, HAND_DOF_DATASET), dtype=np.float32)
    right_hand_6 = np.zeros((n_frames, HAND_DOF_DATASET), dtype=np.float32)

    initial_yaw: float | None = None
    for i, (_, row) in enumerate(df.iterrows()):
        body = array_cell(row[RAW_BODY_COLUMN], (24, 7), dtype=np.float32)
        smpl = compute_smpl_from_body_pose(body)
        smpl_joints[i] = smpl["smpl_joints"]
        smpl_pose[i] = smpl["smpl_pose"]
        body_quat_w[i] = smpl["body_quat_w"]
        left_wrist[i], right_wrist[i] = compute_wrist_joints(smpl_pose[i])
        vr_3pt_position[i], vr_3pt_orientation[i] = compute_vr_3pt(body)
        projected_gravity[i] = compute_projected_gravity(body_quat_w[i])

        body_rot = R.from_quat(body_quat_w[i], scalar_first=True)
        euler = body_rot.as_euler("ZYX", degrees=False)
        if initial_yaw is None:
            initial_yaw = float(euler[0])
        normalized_euler = np.array([euler[0] - initial_yaw, euler[1], euler[2]])
        target_quat = R.from_euler("ZYX", normalized_euler).as_quat(scalar_first=True)
        target_body_orientation[i] = quat_to_rot6d(target_quat.astype(np.float32))

        left_hand_7[i], left_hand_6[i] = retarget_hand(row, LEFT_HAND_COLUMN, LEFT_HAND_ACTIVE_COLUMN)
        right_hand_7[i], right_hand_6[i] = retarget_hand(
            row,
            RIGHT_HAND_COLUMN,
            RIGHT_HAND_ACTIVE_COLUMN,
        )

    return {
        "smpl_joints": smpl_joints,
        "smpl_pose": smpl_pose,
        "body_quat_w": body_quat_w,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "target_body_orientation": target_body_orientation,
        "vr_3pt_position": vr_3pt_position,
        "vr_3pt_orientation": vr_3pt_orientation,
        "projected_gravity": projected_gravity,
        "left_hand_7": left_hand_7,
        "right_hand_7": right_hand_7,
        "left_hand_6": left_hand_6,
        "right_hand_6": right_hand_6,
    }


def run_encoder(session: ort.InferenceSession, body_features: dict[str, np.ndarray]) -> np.ndarray:
    obs = build_encoder_observations(
        body_features["smpl_joints"],
        body_features["body_quat_w"],
        body_features["left_wrist"],
        body_features["right_wrist"],
    )
    n_frames = obs.shape[0]
    motion_tokens = np.zeros((n_frames, ENCODER_OUTPUT_DIM), dtype=np.float32)
    for i in range(n_frames):
        motion_tokens[i] = session.run(None, {"obs_dict": obs[i : i + 1]})[0][0]
    return motion_tokens


def padded_history(
    values: deque[np.ndarray],
    dim: int,
    pad_value: np.ndarray | None = None,
) -> np.ndarray:
    if pad_value is None:
        pad_value = np.zeros(dim, dtype=np.float32)
    if len(values) >= HISTORY_LEN:
        selected = list(values)[-HISTORY_LEN:]
    else:
        selected = [pad_value.astype(np.float32)] * (HISTORY_LEN - len(values)) + list(values)
    return np.concatenate(selected).astype(np.float32)


def build_policy_observation(
    motion_token: np.ndarray,
    base_ang_history: deque[np.ndarray],
    body_q_history: deque[np.ndarray],
    body_dq_history: deque[np.ndarray],
    last_action_history: deque[np.ndarray],
    gravity_history: deque[np.ndarray],
    current_gravity: np.ndarray,
) -> np.ndarray:
    obs = np.zeros((1, POLICY_OBS_DIM), dtype=np.float32)
    offset = 0
    obs[0, offset : offset + 64] = motion_token
    offset += 64
    obs[0, offset : offset + 30] = padded_history(base_ang_history, 3)
    offset += 30
    obs[0, offset : offset + 290] = padded_history(body_q_history, POLICY_ACTION_DIM)
    offset += 290
    obs[0, offset : offset + 290] = padded_history(body_dq_history, POLICY_ACTION_DIM)
    offset += 290
    obs[0, offset : offset + 290] = padded_history(last_action_history, POLICY_ACTION_DIM)
    offset += 290
    obs[0, offset : offset + 30] = padded_history(gravity_history, 3, current_gravity)
    return obs


def policy_action_to_body_state(
    action_isaaclab: np.ndarray,
    previous_body_q: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_target_mujoco = (
        DEFAULT_ANGLES_MUJOCO
        + action_isaaclab[ISAACLAB_TO_MUJOCO] * G1_ACTION_SCALE_MUJOCO
    )
    body_q_isaaclab = (
        q_target_mujoco[MUJOCO_TO_ISAACLAB]
        - DEFAULT_ANGLES_MUJOCO[MUJOCO_TO_ISAACLAB]
    )
    body_dq_isaaclab = (body_q_isaaclab - previous_body_q) / dt
    return body_q_isaaclab.astype(np.float32), body_dq_isaaclab.astype(np.float32)


def run_decoder(
    session: ort.InferenceSession,
    motion_tokens: np.ndarray,
    projected_gravity: np.ndarray,
    history_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames = motion_tokens.shape[0]
    actions = np.zeros((n_frames, POLICY_ACTION_DIM), dtype=np.float32)
    body_q = np.zeros((n_frames, POLICY_ACTION_DIM), dtype=np.float32)
    body_dq = np.zeros((n_frames, POLICY_ACTION_DIM), dtype=np.float32)

    use_closed_loop = history_mode == "closed_loop"
    base_ang_history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
    body_q_history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
    body_dq_history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
    last_action_history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
    gravity_history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
    previous_body_q = np.zeros(POLICY_ACTION_DIM, dtype=np.float32)
    dt = 1.0 / FPS

    for i in range(n_frames):
        gravity = projected_gravity[i].astype(np.float32)
        if not use_closed_loop:
            base_ang_history.clear()
            body_q_history.clear()
            body_dq_history.clear()
            last_action_history.clear()
            gravity_history.clear()
        obs = build_policy_observation(
            motion_tokens[i],
            base_ang_history,
            body_q_history,
            body_dq_history,
            last_action_history,
            gravity_history,
            gravity,
        )
        action = session.run(None, {"obs_dict": obs})[0][0].astype(np.float32)
        current_body_q, current_body_dq = policy_action_to_body_state(
            action,
            previous_body_q,
            dt,
        )

        actions[i] = action
        body_q[i] = current_body_q
        body_dq[i] = current_body_dq

        if use_closed_loop:
            base_ang_history.append(np.zeros(3, dtype=np.float32))
            body_q_history.append(current_body_q)
            body_dq_history.append(current_body_dq)
            last_action_history.append(action)
            gravity_history.append(gravity)
        previous_body_q = current_body_q

    return actions, body_q, body_dq


def compute_eef_state(
    df: pd.DataFrame,
    robot_model,
    body_q: np.ndarray,
    left_hand_7: np.ndarray,
    right_hand_7: np.ndarray,
) -> np.ndarray:
    n_frames = len(df)
    eef_state = np.zeros((n_frames, 14), dtype=np.float64)
    for i, (_, row) in enumerate(df.iterrows()):
        left_part = controller_pose_to_eef_part(row[LEFT_CONTROLLER_COLUMN])
        right_part = controller_pose_to_eef_part(row[RIGHT_CONTROLLER_COLUMN])
        if left_part is not None and right_part is not None:
            eef_state[i] = np.concatenate([left_part, right_part])
            continue

        whole_q = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_q[i],
            left_hand_actuated_joint_values=left_hand_7[i],
            right_hand_actuated_joint_values=right_hand_7[i],
        )
        robot_model.cache_forward_kinematics(whole_q)
        parts = []
        for side in ("left", "right"):
            placement = robot_model.frame_placement(
                robot_model.supplemental_info.hand_frame_names[side]
            )
            pos = placement.translation[:3]
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            parts.append(np.concatenate([pos, quat]))
        eef_state[i] = np.concatenate(parts).astype(np.float64)
    return eef_state


def process_episode_arrays(
    df: pd.DataFrame,
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    robot_model,
    decoder_history_mode: str,
) -> ProcessedEpisode:
    body_features = compute_body_features(df)
    motion_tokens = run_encoder(encoder_session, body_features)
    policy_action, body_q, body_dq = run_decoder(
        decoder_session,
        motion_tokens,
        body_features["projected_gravity"],
        decoder_history_mode,
    )
    eef_state = compute_eef_state(
        df,
        robot_model,
        body_q,
        body_features["left_hand_7"],
        body_features["right_hand_7"],
    )
    return ProcessedEpisode(
        smpl_joints=body_features["smpl_joints"],
        smpl_pose=body_features["smpl_pose"],
        body_quat_w=body_features["body_quat_w"],
        left_wrist=body_features["left_wrist"],
        right_wrist=body_features["right_wrist"],
        target_body_orientation=body_features["target_body_orientation"],
        vr_3pt_position=body_features["vr_3pt_position"],
        vr_3pt_orientation=body_features["vr_3pt_orientation"],
        projected_gravity=body_features["projected_gravity"],
        motion_token=motion_tokens,
        policy_action=policy_action,
        body_q=body_q,
        body_dq=body_dq,
        left_hand_7=body_features["left_hand_7"],
        right_hand_7=body_features["right_hand_7"],
        left_hand_6=body_features["left_hand_6"],
        right_hand_6=body_features["right_hand_6"],
        eef_state=eef_state,
    )


def make_video_info(video_width: int, video_height: int) -> dict:
    return {
        "video.height": video_height,
        "video.width": video_width,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": FPS,
        "video.channels": 3,
        "has_audio": False,
    }


def normalize_image_frame(
    frame: np.ndarray,
    video_width: int,
    video_height: int,
) -> np.ndarray:
    if frame.shape[:2] != (video_height, video_width):
        frame = cv2.resize(
            frame,
            (video_width, video_height),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(frame, dtype=np.uint8)


def iter_video_frames_for_timestamps(
    video_path: Path,
    timestamps: np.ndarray,
    video_width: int,
    video_height: int,
) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        current_index = -1
        current_frame: np.ndarray | None = None

        for timestamp in timestamps:
            target_index = int(round(float(timestamp) * video_fps))
            if frame_count > 0:
                target_index = min(target_index, frame_count - 1)
            if target_index < 0:
                target_index = 0

            while current_index < target_index:
                ok, frame = cap.read()
                if not ok:
                    break
                current_index += 1
                current_frame = normalize_image_frame(frame, video_width, video_height)

            if current_frame is None:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video has no readable frames: {video_path}")
                current_index = 0
                current_frame = normalize_image_frame(frame, video_width, video_height)

            yield current_frame
    finally:
        cap.release()


def make_frame(
    processed: ProcessedEpisode,
    images_bgr: dict[str, np.ndarray],
    frame_idx: int,
    robot_model,
    task: str,
) -> dict:
    whole_q = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=processed.body_q[frame_idx],
        left_hand_actuated_joint_values=processed.left_hand_7[frame_idx],
        right_hand_actuated_joint_values=processed.right_hand_7[frame_idx],
    )
    whole_action_wbc = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=processed.policy_action[frame_idx],
        left_hand_actuated_joint_values=processed.left_hand_7[frame_idx],
        right_hand_actuated_joint_values=processed.right_hand_7[frame_idx],
    )
    return {
        **images_bgr,
        "observation.state": whole_q.astype(np.float64),
        "observation.eef_state": processed.eef_state[frame_idx].astype(np.float64),
        "action.wbc": whole_action_wbc.astype(np.float64),
        "observation.root_orientation": processed.body_quat_w[frame_idx].astype(np.float64),
        "observation.projected_gravity": processed.projected_gravity[frame_idx].astype(
            np.float64
        ),
        "observation.cpp_rotation_offset": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "observation.init_base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "teleop.delta_heading": np.array([0.0], dtype=np.float64),
        "action.motion_token": processed.motion_token[frame_idx].astype(np.float64),
        "teleop.smpl_joints": processed.smpl_joints[frame_idx],
        "teleop.smpl_pose": processed.smpl_pose[frame_idx],
        "teleop.body_quat_w": processed.body_quat_w[frame_idx],
        "teleop.g1_joint_pos": processed.body_q[frame_idx],
        "teleop.g1_joint_vel": processed.body_dq[frame_idx],
        "teleop.target_body_orientation": processed.target_body_orientation[frame_idx],
        "teleop.left_hand_joints": processed.left_hand_6[frame_idx],
        "teleop.right_hand_joints": processed.right_hand_6[frame_idx],
        "teleop.smpl_frame_index": np.array([frame_idx], dtype=np.int64),
        "teleop.left_wrist_joints": processed.left_wrist[frame_idx],
        "teleop.right_wrist_joints": processed.right_wrist[frame_idx],
        "teleop.stream_mode": np.array([1], dtype=np.int32),
        "teleop.planner_mode": np.array([0], dtype=np.int32),
        "teleop.planner_movement": np.zeros(3, dtype=np.float32),
        "teleop.planner_facing": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "teleop.planner_speed": np.array([-1.0], dtype=np.float32),
        "teleop.planner_height": np.array([-1.0], dtype=np.float32),
        "teleop.vr_3pt_position": processed.vr_3pt_position[frame_idx],
        "teleop.vr_3pt_orientation": processed.vr_3pt_orientation[frame_idx],
        "task": task,
    }


def build_features_and_modality(
    robot_model,
    video_width: int,
    video_height: int,
) -> tuple[dict, dict]:
    features = get_features_sonic_vla(robot_model, hand_dof=HAND_DOF_DATASET)
    features = {**features, **INDEX_FEATURES}
    features["observation.images.ego_view"] = {
        **features["observation.images.ego_view"],
        "shape": [video_height, video_width, 3],
        "info": make_video_info(video_width, video_height),
    }
    features["observation.images.level_cam"] = {
        **features["observation.images.ego_view"],
        "shape": [video_height, video_width, 3],
        "info": make_video_info(video_width, video_height),
    }
    modality_config = get_modality_config_sonic_vla(robot_model, hand_dof=HAND_DOF_DATASET)
    modality_config["video"]["level_cam"] = {
        "original_key": "observation.images.level_cam"
    }
    return features, modality_config


def data_file_path(root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_chunk(episode_index, chunks_size)
    return root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"


def video_file_path(
    root: Path, video_key: str, episode_index: int, chunks_size: int
) -> Path:
    chunk = episode_chunk(episode_index, chunks_size)
    return root / (
        f"videos/chunk-{chunk:03d}/{video_key}/"
        f"episode_{episode_index:06d}.mp4"
    )


def coerce_for_feature(value, feature: dict):
    shape = tuple(feature["shape"])
    dtype = feature["dtype"]
    arr = np.asarray(value)
    if shape == (1,):
        return arr.reshape(-1)[0].item()
    return np.ascontiguousarray(arr, dtype=dtype).reshape(shape)


def finalize_episode_table(
    buffers: dict[str, list],
    features: dict,
) -> dict[str, np.ndarray]:
    episode_table = {}
    for key, values in buffers.items():
        shape = tuple(features[key]["shape"])
        if shape == (1,):
            episode_table[key] = np.asarray(values, dtype=features[key]["dtype"])
        else:
            episode_table[key] = np.stack(values)
    return episode_table


def write_direct_episode(
    ep: EpisodePaths,
    start_index: int,
    output_root: Path,
    task: str,
    chunks_size: int,
    max_frames_per_episode: int | None,
    decoder_history_mode: str,
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    robot_model,
    features: dict,
    hf_features: datasets.Features,
    non_video_features: dict,
    video_width: int,
    video_height: int,
) -> tuple[int, int, dict]:
    df, source_pico_timestamps, output_timestamps = load_aligned_episode_dataframe(
        ep,
        max_frames_per_episode,
    )
    processed = process_episode_arrays(
        df,
        encoder_session,
        decoder_session,
        robot_model,
        decoder_history_mode,
    )

    parquet_path = data_file_path(output_root, ep.episode_index, chunks_size)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    buffers = {key: [] for key in non_video_features}
    video_writers = {}
    video_iters = {}
    for video_key, source_video_path in ep.video_paths.items():
        output_video_path = video_file_path(
            output_root, video_key, ep.episode_index, chunks_size
        )
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writers[video_key] = SyncVideoWriter(
            output_video_path, video_width, video_height, FPS, "h264"
        )
        video_iters[video_key] = iter(
            iter_video_frames_for_timestamps(
                source_video_path,
                camera_timestamps_for_pico(ep, video_key, source_pico_timestamps),
                video_width,
                video_height,
            )
        )
    frame_count = 0
    try:
        for frame_idx in range(len(df)):
            images_bgr = {key: next(iterator) for key, iterator in video_iters.items()}
            frame = make_frame(processed, images_bgr, frame_idx, robot_model, task)
            frame["timestamp"] = np.array([float(output_timestamps[frame_idx])], dtype=np.float32)
            frame["frame_index"] = np.array([frame_idx], dtype=np.int64)
            frame["episode_index"] = np.array([ep.episode_index], dtype=np.int64)
            frame["index"] = np.array([start_index + frame_idx], dtype=np.int64)
            frame["task_index"] = np.array([0], dtype=np.int64)

            for key, image_bgr in images_bgr.items():
                video_writers[key].add_frame(image_bgr)
            for key, feature in non_video_features.items():
                buffers[key].append(coerce_for_feature(frame[key], feature))
            frame_count += 1
        if frame_count != len(df):
            raise RuntimeError(
                f"Episode {ep.episode_index} video/parquet length mismatch after "
                f"timestamp sampling: video={frame_count}, parquet={len(df)}"
            )
        for writer in video_writers.values():
            writer.stop()
    except Exception:
        for writer in video_writers.values():
            writer.cancel()
        raise

    episode_table = finalize_episode_table(buffers, non_video_features)
    episode_dataset = datasets.Dataset.from_dict(
        {key: episode_table[key] for key in hf_features},
        features=hf_features,
        split="train",
    )
    episode_dataset.to_parquet(parquet_path)
    episode_stats = compute_episode_stats(episode_table, non_video_features)
    return ep.episode_index, frame_count, episode_stats


def worker_init(
    output_root: str,
    task: str,
    decoder_history_mode: str,
    max_frames_per_episode: int | None,
    chunks_size: int,
    video_width: int,
    video_height: int,
) -> None:
    global _WORKER_ENCODER_SESSION
    global _WORKER_DECODER_SESSION
    global _WORKER_ROBOT_MODEL
    global _WORKER_FEATURES
    global _WORKER_HF_FEATURES
    global _WORKER_NON_VIDEO_FEATURES
    global _WORKER_OUTPUT_ROOT
    global _WORKER_TASK
    global _WORKER_DECODER_HISTORY_MODE
    global _WORKER_MAX_FRAMES_PER_EPISODE
    global _WORKER_CHUNKS_SIZE
    global _WORKER_VIDEO_WIDTH
    global _WORKER_VIDEO_HEIGHT

    os.environ["HAND_TYPE"] = "brainco"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    _WORKER_ENCODER_SESSION = ort.InferenceSession(
        str(asset_path("gear_sonic_deploy/policy/release/model_encoder.onnx")),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    _WORKER_DECODER_SESSION = ort.InferenceSession(
        str(asset_path("gear_sonic_deploy/policy/release/model_decoder.onnx")),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    _WORKER_ROBOT_MODEL = get_g1_robot_model()
    _WORKER_FEATURES, _ = build_features_and_modality(
        _WORKER_ROBOT_MODEL,
        video_width,
        video_height,
    )
    _WORKER_HF_FEATURES = get_hf_features_from_features(_WORKER_FEATURES)
    _WORKER_NON_VIDEO_FEATURES = {
        key: feature
        for key, feature in _WORKER_FEATURES.items()
        if feature["dtype"] != "video"
    }
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_TASK = task
    _WORKER_DECODER_HISTORY_MODE = decoder_history_mode
    _WORKER_MAX_FRAMES_PER_EPISODE = max_frames_per_episode
    _WORKER_CHUNKS_SIZE = chunks_size
    _WORKER_VIDEO_WIDTH = video_width
    _WORKER_VIDEO_HEIGHT = video_height


def worker_convert_episode(payload: tuple[EpisodePaths, int]) -> tuple[int, int, dict]:
    ep, start_index = payload
    assert _WORKER_OUTPUT_ROOT is not None
    assert _WORKER_TASK is not None
    assert _WORKER_DECODER_HISTORY_MODE is not None
    assert _WORKER_ENCODER_SESSION is not None
    assert _WORKER_DECODER_SESSION is not None
    assert _WORKER_ROBOT_MODEL is not None
    assert _WORKER_FEATURES is not None
    assert _WORKER_HF_FEATURES is not None
    assert _WORKER_NON_VIDEO_FEATURES is not None
    return write_direct_episode(
        ep=ep,
        start_index=start_index,
        output_root=_WORKER_OUTPUT_ROOT,
        task=_WORKER_TASK,
        chunks_size=_WORKER_CHUNKS_SIZE,
        max_frames_per_episode=_WORKER_MAX_FRAMES_PER_EPISODE,
        decoder_history_mode=_WORKER_DECODER_HISTORY_MODE,
        encoder_session=_WORKER_ENCODER_SESSION,
        decoder_session=_WORKER_DECODER_SESSION,
        robot_model=_WORKER_ROBOT_MODEL,
        features=_WORKER_FEATURES,
        hf_features=_WORKER_HF_FEATURES,
        non_video_features=_WORKER_NON_VIDEO_FEATURES,
        video_width=_WORKER_VIDEO_WIDTH,
        video_height=_WORKER_VIDEO_HEIGHT,
    )


def build_exporter(
    output_root: Path,
    task: str,
    overwrite: bool,
    decoder_history_mode: str,
    video_width: int,
    video_height: int,
) -> tuple[Gr00tDataExporter, object]:
    os.environ["HAND_TYPE"] = "brainco"
    robot_model = get_g1_robot_model()
    features, modality_config = build_features_and_modality(
        robot_model,
        video_width,
        video_height,
    )
    exporter = Gr00tDataExporter.create(
        save_root=output_root,
        fps=FPS,
        features=features,
        modality_config=modality_config,
        task=task,
        script_config={
            "source": "raw_pico_lerobot",
            "pipeline": "raw_pico_to_smpl_to_sonic_encoder_decoder_to_gr00t",
            "encoder_model": str(asset_path("gear_sonic_deploy/policy/release/model_encoder.onnx")),
            "decoder_model": str(asset_path("gear_sonic_deploy/policy/release/model_decoder.onnx")),
            "hand_type": "brainco",
            "decoder_history_mode": decoder_history_mode,
            "decoder_history_note": (
                "zero mode keeps robot history at standing/zero state for stable "
                "offline batching; live replay uses realtime robot or simulation feedback"
            ),
        },
        robot_type="g1",
        overwrite_existing=overwrite,
    )
    return exporter, robot_model


def convert_dataset(
    source_root: Path,
    output_root: Path,
    episodes: list[EpisodePaths],
    task: str,
    overwrite: bool,
    max_frames_per_episode: int | None,
    decoder_history_mode: str,
    video_width: int,
    video_height: int,
) -> None:
    validate_output_root(output_root, [source_root, *_episode_inputs(episodes)])
    encoder_path = asset_path("gear_sonic_deploy/policy/release/model_encoder.onnx")
    decoder_path = asset_path("gear_sonic_deploy/policy/release/model_decoder.onnx")
    encoder_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    decoder_session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    exporter, robot_model = build_exporter(
        output_root,
        task,
        overwrite,
        decoder_history_mode,
        video_width,
        video_height,
    )

    total_frames = 0
    start_time = time.monotonic()
    try:
        for ep in tqdm(episodes, desc="Converting episodes"):
            df, source_pico_timestamps, _output_timestamps = load_aligned_episode_dataframe(
                ep,
                max_frames_per_episode,
            )

            processed = process_episode_arrays(
                df,
                encoder_session,
                decoder_session,
                robot_model,
                decoder_history_mode,
            )

            frame_count = 0
            video_iters = {
                key: iter(
                    iter_video_frames_for_timestamps(
                        path,
                        camera_timestamps_for_pico(ep, key, source_pico_timestamps),
                        video_width,
                        video_height,
                    )
                )
                for key, path in ep.video_paths.items()
            }
            for frame_idx in range(len(df)):
                images_bgr = {key: next(iterator) for key, iterator in video_iters.items()}
                exporter.add_frame(
                    make_frame(processed, images_bgr, frame_idx, robot_model, task)
                )
                frame_count += 1
            if frame_count != len(df):
                raise RuntimeError(
                    f"Episode {ep.episode_index} video/parquet length mismatch: "
                    f"video={frame_count}, parquet={len(df)}"
                )
            exporter.save_episode()
            total_frames += frame_count
    finally:
        exporter.cancel_video_writers()

    elapsed = time.monotonic() - start_time
    print(
        f"Done. Wrote {len(episodes)} episodes / {total_frames} frames to {output_root} "
        f"in {elapsed / 60:.1f} min."
    )


def episode_lengths(
    episodes: list[EpisodePaths],
    max_frames_per_episode: int | None,
) -> dict[int, int]:
    return {
        ep.episode_index: len(load_aligned_episode_dataframe(ep, max_frames_per_episode)[0])
        for ep in episodes
    }


def write_direct_metadata(
    output_root: Path,
    task: str,
    features: dict,
    modality_config: dict,
    episode_lengths_by_index: dict[int, int],
    episode_stats_by_index: dict[int, dict],
    decoder_history_mode: str,
    chunks_size: int,
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "g1",
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": chunks_size,
        "fps": FPS,
        "splits": {"train": "0:0"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "script_config": {
            "source": "raw_pico_lerobot",
            "pipeline": "raw_pico_to_smpl_to_sonic_encoder_decoder_to_gr00t",
            "encoder_model": str(asset_path("gear_sonic_deploy/policy/release/model_encoder.onnx")),
            "decoder_model": str(asset_path("gear_sonic_deploy/policy/release/model_decoder.onnx")),
            "hand_type": "brainco",
            "decoder_history_mode": decoder_history_mode,
            "decoder_history_note": (
                "zero mode keeps robot history at standing/zero state for stable "
                "offline batching; live replay uses realtime robot or simulation feedback"
            ),
        },
        "discarded_episode_indices": [],
    }
    write_info(info, output_root)
    with (meta_dir / "modality.json").open("w", encoding="utf-8") as f:
        json.dump(modality_config, f, indent=4)
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n")

    for ep_idx in sorted(episode_lengths_by_index):
        length = episode_lengths_by_index[ep_idx]
        stats = episode_stats_by_index[ep_idx]
        info["total_episodes"] += 1
        info["total_frames"] += length
        info["total_videos"] += sum(
            feature.get("dtype") == "video" for feature in features.values()
        )
        chunk = episode_chunk(ep_idx, chunks_size)
        if chunk >= info["total_chunks"]:
            info["total_chunks"] += 1
        info["splits"] = {"train": f"0:{info['total_episodes']}"}
        write_info(info, output_root)
        write_episode(
            {"episode_index": ep_idx, "tasks": [task], "length": length},
            output_root,
        )
        write_episode_stats(ep_idx, stats, output_root)


def _episode_inputs(episodes: list[EpisodePaths]) -> list[Path]:
    return [path for ep in episodes
            for path in (ep.parquet_path, *ep.video_paths.values(), *ep.alignment_paths.values())]


def convert_dataset_parallel(
    output_root: Path,
    episodes: list[EpisodePaths],
    task: str,
    overwrite: bool,
    max_frames_per_episode: int | None,
    decoder_history_mode: str,
    workers: int,
    video_width: int,
    video_height: int,
) -> None:
    validate_output_root(output_root, _episode_inputs(episodes))
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    elif output_root.exists():
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["HAND_TYPE"] = "brainco"
    robot_model = get_g1_robot_model()
    features, modality_config = build_features_and_modality(
        robot_model,
        video_width,
        video_height,
    )
    chunks_size = DEFAULT_CHUNKS_SIZE
    lengths = episode_lengths(episodes, max_frames_per_episode)
    start_indices = {}
    running = 0
    for ep in episodes:
        start_indices[ep.episode_index] = running
        running += lengths[ep.episode_index]

    episode_stats_by_index: dict[int, dict] = {}
    start_time = time.monotonic()
    payloads = [(ep, start_indices[ep.episode_index]) for ep in episodes]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=worker_init,
        initargs=(
            str(output_root),
            task,
            decoder_history_mode,
            max_frames_per_episode,
            chunks_size,
            video_width,
            video_height,
        ),
    ) as executor:
        futures = {
            executor.submit(worker_convert_episode, payload): payload[0].episode_index
            for payload in payloads
        }
        with tqdm(total=len(futures), desc=f"Converting episodes x{workers}") as progress:
            for future in concurrent.futures.as_completed(futures):
                ep_idx = futures[future]
                try:
                    out_ep_idx, frame_count, stats = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"Episode {ep_idx} failed") from exc
                if frame_count != lengths[out_ep_idx]:
                    raise RuntimeError(
                        f"Episode {out_ep_idx} length mismatch: "
                        f"expected {lengths[out_ep_idx]}, got {frame_count}"
                    )
                episode_stats_by_index[out_ep_idx] = stats
                progress.update(1)

    write_direct_metadata(
        output_root=output_root,
        task=task,
        features=features,
        modality_config=modality_config,
        episode_lengths_by_index=lengths,
        episode_stats_by_index=episode_stats_by_index,
        decoder_history_mode=decoder_history_mode,
        chunks_size=chunks_size,
    )
    elapsed = time.monotonic() - start_time
    print(
        f"Done. Wrote {len(episodes)} episodes / {sum(lengths.values())} frames "
        f"to {output_root} in {elapsed / 60:.1f} min with {workers} workers."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw Pico LeRobot data to SONIC/GR00T LeRobot format.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Raw Pico LeRobot dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to <source-root>_sonic_gr00t.",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="Episode selector, e.g. '0', '0,3,5', or '0-9'. Defaults to all episodes.",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Debug limit for smoke tests.",
    )
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=None,
        help="Directory containing episode_*.{camera}_alignment.json; defaults to SOURCE_ROOT/sync.",
    )
    parser.add_argument(
        "--ignore-alignment",
        action="store_true",
        help="Sample each video at raw camera time instead of applying sync alignment.",
    )
    parser.add_argument(
        "--require-alignment",
        action="store_true",
        help="Fail if a source camera has no usable alignment JSON.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task string. Defaults to the first task in source meta/tasks.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-root first if it already exists.",
    )
    parser.add_argument(
        "--decoder-history-mode",
        choices=("zero", "closed_loop"),
        default="zero",
        help=(
            "How to fill decoder robot-state history. zero is stable for offline "
            "batching; closed_loop feeds predicted actions back and can diverge "
            "without simulation feedback."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of episode worker processes. Use >1 for parallel conversion.",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=DEFAULT_OUTPUT_VIDEO_WIDTH,
        help="Output ego-view video width.",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=DEFAULT_OUTPUT_VIDEO_HEIGHT,
        help="Output ego-view video height.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ignore_alignment and args.require_alignment:
        raise ValueError("--ignore-alignment cannot be combined with --require-alignment")
    source_root = args.source_root.expanduser().resolve()
    if args.output_root is None:
        output_root = source_root.with_name(f"{source_root.name}_sonic_gr00t")
    else:
        output_root = args.output_root.expanduser().resolve()

    validate_output_root(output_root, [source_root])
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_root}")

    selected = parse_episode_selector(args.episodes)
    alignment_dir = (
        None
        if args.ignore_alignment
        else (args.alignment_dir.expanduser().resolve() if args.alignment_dir is not None else source_root / "sync")
    )
    if alignment_dir is not None:
        validate_output_root(output_root, [alignment_dir])
    episodes = discover_episodes(
        source_root,
        selected,
        alignment_dir=alignment_dir,
        require_alignment=args.require_alignment,
    )
    if not episodes:
        raise ValueError("No source episodes selected; existing output was not changed")
    validate_output_root(output_root, _episode_inputs(episodes))
    task = args.task or load_first_task(source_root)
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Episodes: {len(episodes)}")
    print(f"Task: {task}")
    print(f"Video size: {args.video_width}x{args.video_height}")
    if alignment_dir is None:
        print("Video alignment: disabled")
    else:
        aligned_videos = sum(len(ep.alignment_paths) for ep in episodes)
        print(f"Video alignment: {aligned_videos} streams from {alignment_dir}")

    if args.workers > 1:
        convert_dataset_parallel(
            output_root=output_root,
            episodes=episodes,
            task=task,
            overwrite=args.overwrite,
            max_frames_per_episode=args.max_frames_per_episode,
            decoder_history_mode=args.decoder_history_mode,
            workers=args.workers,
            video_width=args.video_width,
            video_height=args.video_height,
        )
    else:
        convert_dataset(
            source_root=source_root,
            output_root=output_root,
            episodes=episodes,
            task=task,
            overwrite=args.overwrite,
            max_frames_per_episode=args.max_frames_per_episode,
            decoder_history_mode=args.decoder_history_mode,
            video_width=args.video_width,
            video_height=args.video_height,
        )
    mapping_path = write_stage_a_source_mapping(
        source_root=source_root,
        output_root=output_root,
        episodes=episodes,
        alignment_mode=(
            "ignore"
            if alignment_dir is None
            else ("require" if args.require_alignment else "apply_if_present")
        ),
        max_frames_per_episode=args.max_frames_per_episode,
    )
    print(f"Stage-A source mapping: {mapping_path}")


if __name__ == "__main__":
    main()
