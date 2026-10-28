# SPDX-License-Identifier: Apache-2.0
"""Physical field validation for G1/SONIC/Revo2 demonstration episodes."""

from __future__ import annotations

import cv2
import numpy as np

JOINT_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")

FPS = 50
ACTION_HORIZON = 50
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

G1_MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

MAX_ABS_BODY_POSITION_RAD = 16.0
MAX_ABS_BODY_VELOCITY_RAD_S = 200.0
MAX_ABS_MOTION_TOKEN = 32.0
MIN_REALTIME_NS = 1_500_000_000_000_000_000
MAX_REALTIME_NS = 4_102_444_800_000_000_000


def validate_robot_sample(
    body_q: object,
    body_dq: object,
    token: object,
    *,
    robot_index: object,
    previous_robot_index: int | None,
    realtime_ns: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate one decoded ``g1_debug`` sample before it reaches LeRobot."""
    arrays = (
        np.asarray(body_q, dtype=np.float64),
        np.asarray(body_dq, dtype=np.float64),
        np.asarray(token, dtype=np.float64),
    )
    expected_shapes = ((29,), (29,), (64,))
    labels = ("body_q", "body_dq", "token_state")
    limits = (
        MAX_ABS_BODY_POSITION_RAD,
        MAX_ABS_BODY_VELOCITY_RAD_S,
        MAX_ABS_MOTION_TOKEN,
    )
    for label, array, shape, limit in zip(labels, arrays, expected_shapes, limits):
        if array.shape != shape:
            raise ValueError(f"Sonic feedback {label} must have shape {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Sonic feedback {label} contains non-finite values")
        maximum = float(np.max(np.abs(array)))
        if maximum > limit:
            raise ValueError(
                f"Sonic feedback {label} exceeds physical validity bound "
                f"({maximum:.6g} > {limit:g})"
            )

    index = int(robot_index)
    if index < 0:
        raise ValueError("Sonic feedback index must be non-negative")
    if previous_robot_index is not None and index <= previous_robot_index:
        raise ValueError(
            "Sonic feedback index must advance strictly "
            f"({index} <= {previous_robot_index})"
        )

    source_realtime_ns = int(realtime_ns)
    if not MIN_REALTIME_NS <= source_realtime_ns <= MAX_REALTIME_NS:
        raise ValueError(
            "Sonic feedback timestamp_realtime_ns is outside the supported "
            "2017-2100 epoch range"
        )
    return (*arrays, index, source_realtime_ns)


def normalize_hand_positions(values: object) -> np.ndarray:
    positions = np.asarray(values, dtype=np.float32)
    if positions.shape != (6,) or not np.isfinite(positions).all():
        raise ValueError("BrainCo hand feedback must contain six finite positions")
    if np.any(positions < 0.0) or np.any(positions > 1000.0):
        raise ValueError("BrainCo hand feedback positions must be within [0, 1000]")
    return positions / np.float32(1000.0)


def validate_hand_actions(values: object) -> np.ndarray:
    actions = np.asarray(values, dtype=np.float32)
    if actions.shape != (6,) or not np.isfinite(actions).all():
        raise ValueError("Revo2 hand action must contain six finite targets")
    if np.any(actions < 0.0) or np.any(actions > 1.0):
        raise ValueError("Revo2 hand action targets must be within [0, 1]")
    return actions


def prepare_ego_image(image: object) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("camera frame must be an uint8 RGB image")
    source_height, source_width = frame.shape[:2]
    target_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    source_ratio = source_width / source_height
    if source_ratio > target_ratio:
        crop_width = max(1, int(round(source_height * target_ratio)))
        left = (source_width - crop_width) // 2
        frame = frame[:, left : left + crop_width]
    elif source_ratio < target_ratio:
        crop_height = max(1, int(round(source_width / target_ratio)))
        top = (source_height - crop_height) // 2
        frame = frame[top : top + crop_height, :]
    resized = cv2.resize(
        frame,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized, dtype=np.uint8)
