"""Shared, immutable MuJoCo body-point definitions for endpoint retargeting."""

from __future__ import annotations

import numpy as np


G1_LEFT_PALM_ENDPOINT_BODY = "left_wrist_yaw_link"
G1_RIGHT_PALM_ENDPOINT_BODY = "right_wrist_yaw_link"
G1_LEFT_FOOT_ENDPOINT_BODY = "left_ankle_roll_link"
G1_RIGHT_FOOT_ENDPOINT_BODY = "right_ankle_roll_link"

G1_LEFT_PALM_ENDPOINT_OFFSET_M = np.array(
    [0.0415, 0.0030, 0.0], dtype=np.float64
)
G1_RIGHT_PALM_ENDPOINT_OFFSET_M = np.array(
    [0.0415, -0.0030, 0.0], dtype=np.float64
)
G1_LEFT_FOOT_ENDPOINT_OFFSET_M = np.array(
    [0.10, 0.0, -0.025], dtype=np.float64
)
G1_RIGHT_FOOT_ENDPOINT_OFFSET_M = np.array(
    [0.10, 0.0, -0.025], dtype=np.float64
)
