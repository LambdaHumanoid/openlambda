#!/usr/bin/env python3
"""Measure the G1 whole-body endpoint morphology from a MuJoCo model."""

from __future__ import annotations
from egoalign.runtime import asset_path

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.human_data.endpoint_geometry import (  # noqa: E402
    G1_LEFT_FOOT_ENDPOINT_OFFSET_M,
    G1_LEFT_PALM_ENDPOINT_OFFSET_M,
    G1_RIGHT_FOOT_ENDPOINT_OFFSET_M,
    G1_RIGHT_PALM_ENDPOINT_OFFSET_M,
)


DEFAULT_ROBOT_XML = (
    asset_path("gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml")
)
LEFT_PALM_OFFSET = G1_LEFT_PALM_ENDPOINT_OFFSET_M
RIGHT_PALM_OFFSET = G1_RIGHT_PALM_ENDPOINT_OFFSET_M
LEFT_FOOT_OFFSET = G1_LEFT_FOOT_ENDPOINT_OFFSET_M
RIGHT_FOOT_OFFSET = G1_RIGHT_FOOT_ENDPOINT_OFFSET_M


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _joint_anchor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
) -> np.ndarray:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint not found: {name}")
    return data.xanchor[joint_id].copy()


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {name}")
    return int(body_id)


def _body_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    local_offset: np.ndarray,
) -> np.ndarray:
    body_id = _body_id(model, body_name)
    rotation = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + rotation @ local_offset


def _chain_length(points: list[np.ndarray]) -> float:
    return float(
        sum(np.linalg.norm(right - left) for left, right in zip(points[:-1], points[1:]))
    )


def measure_morphology(robot_xml: Path) -> dict[str, object]:
    robot_xml = robot_xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    result: dict[str, object] = {
        "format_version": 2,
        "coordinate_frame": "G1 floating-base frame: x forward, y left, z up",
        "measurement_pose": "floating base identity; all actuated joint positions zero",
        "robot_xml": str(robot_xml),
        "robot_xml_sha256": sha256_file(robot_xml),
        "endpoint_definition": {
            "left": {
                "body": "left_wrist_yaw_link",
                "local_offset_m": LEFT_PALM_OFFSET.tolist(),
            },
            "right": {
                "body": "right_wrist_yaw_link",
                "local_offset_m": RIGHT_PALM_OFFSET.tolist(),
            },
        },
        "foot_endpoint_definition": {
            "left": {
                "body": "left_ankle_roll_link",
                "local_offset_m": LEFT_FOOT_OFFSET.tolist(),
            },
            "right": {
                "body": "right_ankle_roll_link",
                "local_offset_m": RIGHT_FOOT_OFFSET.tolist(),
            },
        },
        "sides": {},
    }

    root = data.xpos[_body_id(model, "pelvis")].copy()
    for side, palm_offset, foot_offset in (
        ("left", LEFT_PALM_OFFSET, LEFT_FOOT_OFFSET),
        ("right", RIGHT_PALM_OFFSET, RIGHT_FOOT_OFFSET),
    ):
        hip_points = [
            _joint_anchor(model, data, f"{side}_hip_pitch_joint"),
            _joint_anchor(model, data, f"{side}_hip_roll_joint"),
            _joint_anchor(model, data, f"{side}_hip_yaw_joint"),
            _joint_anchor(model, data, f"{side}_knee_joint"),
        ]
        lower_leg_points = [
            hip_points[-1],
            _joint_anchor(model, data, f"{side}_ankle_pitch_joint"),
            _joint_anchor(model, data, f"{side}_ankle_roll_joint"),
        ]
        foot = _body_point(model, data, f"{side}_ankle_roll_link", foot_offset)
        shoulder = _joint_anchor(model, data, f"{side}_shoulder_pitch_joint")
        upper_points = [
            shoulder,
            _joint_anchor(model, data, f"{side}_shoulder_roll_joint"),
            _joint_anchor(model, data, f"{side}_shoulder_yaw_joint"),
            _joint_anchor(model, data, f"{side}_elbow_joint"),
        ]
        forearm_points = [
            upper_points[-1],
            _joint_anchor(model, data, f"{side}_wrist_roll_joint"),
            _joint_anchor(model, data, f"{side}_wrist_pitch_joint"),
            _joint_anchor(model, data, f"{side}_wrist_yaw_joint"),
        ]
        palm = _body_point(model, data, f"{side}_wrist_yaw_link", palm_offset)
        result["sides"][side] = {
            "hip_pitch_anchor_from_root_m": (hip_points[0] - root).tolist(),
            "hip_pitch_anchor_distance_m": float(np.linalg.norm(hip_points[0] - root)),
            "upper_leg_chain_length_m": _chain_length(hip_points),
            "lower_leg_chain_length_m": _chain_length(lower_leg_points),
            "foot_offset_length_m": float(np.linalg.norm(foot - lower_leg_points[-1])),
            "foot_position_from_root_m": (foot - root).tolist(),
            "shoulder_pitch_anchor_from_root_m": (shoulder - root).tolist(),
            "upper_arm_chain_length_m": _chain_length(upper_points),
            "forearm_chain_length_m": _chain_length(forearm_points),
            "palm_offset_length_m": float(np.linalg.norm(palm - forearm_points[-1])),
            "palm_position_from_root_m": (palm - root).tolist(),
            "joint_anchors_m": {
                "hip_pitch": hip_points[0].tolist(),
                "hip_roll": hip_points[1].tolist(),
                "hip_yaw": hip_points[2].tolist(),
                "knee": hip_points[3].tolist(),
                "ankle_pitch": lower_leg_points[1].tolist(),
                "ankle_roll": lower_leg_points[2].tolist(),
                "shoulder_pitch": upper_points[0].tolist(),
                "shoulder_roll": upper_points[1].tolist(),
                "shoulder_yaw": upper_points[2].tolist(),
                "elbow": upper_points[3].tolist(),
                "wrist_roll": forearm_points[1].tolist(),
                "wrist_pitch": forearm_points[2].tolist(),
                "wrist_yaw": forearm_points[3].tolist(),
            },
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = measure_morphology(args.robot_xml)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
