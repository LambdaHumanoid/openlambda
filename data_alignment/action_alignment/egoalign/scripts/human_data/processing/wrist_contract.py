"""Canonical SONIC wrist ordering shared by replay and offline diagnostics."""

from __future__ import annotations

import numpy as np


WRIST_AXES_PER_HAND = 3
G1_BODY_JOINT_COUNT = 29
LEFT_WRIST_ISAACLAB_INDICES = np.asarray([23, 25, 27], dtype=np.int64)
RIGHT_WRIST_ISAACLAB_INDICES = np.asarray([24, 26, 28], dtype=np.int64)
WRIST_ISAACLAB_ORDER = (
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "left_wrist_yaw",
    "right_wrist_yaw",
)
POSE_WRIST_CONTRACT = {
    "version": 1,
    "source": "teleop.left_wrist_joints+teleop.right_wrist_joints",
    "per_hand_order": ["roll", "pitch", "yaw"],
    "cpp_order": list(WRIST_ISAACLAB_ORDER),
    "joint_pos_indices": [23, 24, 25, 26, 27, 28],
    "zero_fallback": False,
}


def wrist_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    """Return finite float32 wrist rows with shape ``[N, 3]``."""

    result = np.asarray(values, dtype=np.float32)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    if result.ndim != 2 or result.shape[1] != WRIST_AXES_PER_HAND:
        raise ValueError(
            f"{name} must have shape [N, {WRIST_AXES_PER_HAND}], got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite wrist positions")
    return np.ascontiguousarray(result)


def interleave_wrist_positions(
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
) -> np.ndarray:
    """Convert per-hand ``[roll,pitch,yaw]`` rows to C++/IsaacLab order."""

    left = wrist_matrix(left_wrist, name="left_wrist")
    right = wrist_matrix(right_wrist, name="right_wrist")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "left/right wrist row count mismatch: "
            f"{left.shape[0]} != {right.shape[0]}"
        )
    return np.stack(
        (
            left[:, 0],
            right[:, 0],
            left[:, 1],
            right[:, 1],
            left[:, 2],
            right[:, 2],
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def wrist_positions_to_g1_joint_pos(
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
) -> np.ndarray:
    """Place real wrist positions into the canonical 29D SONIC joint vector."""

    left = wrist_matrix(left_wrist, name="left_wrist")
    right = wrist_matrix(right_wrist, name="right_wrist")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "left/right wrist row count mismatch: "
            f"{left.shape[0]} != {right.shape[0]}"
        )
    joint_pos = np.zeros((left.shape[0], G1_BODY_JOINT_COUNT), dtype=np.float32)
    joint_pos[:, LEFT_WRIST_ISAACLAB_INDICES] = left
    joint_pos[:, RIGHT_WRIST_ISAACLAB_INDICES] = right
    return joint_pos
