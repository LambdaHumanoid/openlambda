"""Shared contracts for SONIC C++ + MuJoCo dataset rollout.

These constants belong to Stage B/Stage C dataset conversion.  The downstream
velocity-quality gate consumes validated rollout telemetry but does not own or
define the rollout lifecycle.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


MIN_VALID_ROOT_HEIGHT_M = 0.5
EPISODE_ROLLOUT_CONTRACT_VERSION = 1
SIM_EPISODE_LIFECYCLE_CONTRACT = (
    "deploy-exit->sim-reset+bridge-reset+elastic-reset->fresh-deploy"
    "->planner-history-settle->elastic-release+released-physical-settle"
    "->pose-ramp+first-pose-physical-settle"
    "+support-integrity-gate->formal-replay:v4"
)
INITIAL_BODY_Q_REFERENCE_CONTRACT = "baseline_measured_g1_first_frame_v1"
STRICT_AB_INITIALIZATION_CONTRACT = (
    "feet_strict_baseline_pose_and_measured_body_q_v1"
)
STRICT_AB_INITIAL_BODY_Q_MAX_ERROR_DEG = 6.0
STRICT_AB_INITIAL_BODY_Q_RMS_ERROR_DEG = 2.5


def sha256_file(path: str | Path) -> str:
    """Hash the exact Stage-A parquet consumed by one rollout."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_episode_rollout_contract(
    *,
    episode_index: int,
    source_sha256: str,
    source_rows: int,
    replay_rows: int,
    protocol: str,
    replay_rate: float,
    video_name: str,
    pose_wrist_contract: Mapping[str, Any],
    initialization_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-episode provenance stamp written after Stage-B validation."""

    result = {
        "format_version": EPISODE_ROLLOUT_CONTRACT_VERSION,
        "episode_index": int(episode_index),
        "source_sha256": str(source_sha256),
        "source_rows": int(source_rows),
        "replay_rows": int(replay_rows),
        "protocol": str(protocol),
        "replay_rate": float(replay_rate),
        "video_name": str(video_name),
        "pose_wrist_contract": dict(pose_wrist_contract),
        "sim_episode_lifecycle_contract": SIM_EPISODE_LIFECYCLE_CONTRACT,
    }
    if initialization_contract is not None:
        result["initialization_contract"] = dict(initialization_contract)
    return result


def validate_episode_rollout_contract(
    value: Mapping[str, Any],
    *,
    episode_index: int,
    source_sha256: str,
    source_rows: int,
    replay_rows: int,
    protocol: str,
    replay_rate: float,
    video_name: str,
    pose_wrist_contract: Mapping[str, Any],
    initialization_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject an episode stamp that belongs to different Stage-B inputs."""

    expected = build_episode_rollout_contract(
        episode_index=episode_index,
        source_sha256=source_sha256,
        source_rows=source_rows,
        replay_rows=replay_rows,
        protocol=protocol,
        replay_rate=replay_rate,
        video_name=video_name,
        pose_wrist_contract=pose_wrist_contract,
        initialization_contract=initialization_contract,
    )
    actual = dict(value)
    mismatches = [
        f"{key}={actual.get(key)!r} (expected {expected_value!r})"
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "episode rollout contract mismatch: " + ", ".join(mismatches)
        )
    return actual
