#!/usr/bin/env python3
"""Produce a fail-closed production audit for one endpoint-retarget replay.

This audit is intentionally separate from the three-round position controller.
It reports the remaining production concerns without changing a candidate or
turning an uncalibrated smoke result into a train-ready acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egoalign.adapters.sonic.telemetry_schema import (  # noqa: E402
    MUJOCO_BODY_JOINT_NAMES,
)
from scripts.human_data.smpl_endpoint_retargeting import (
    FOOT_CHAIN_JOINT_INDICES,
    ARM_OPTIMIZED_JOINT_INDICES,
    ENDPOINT_PARTS_HANDS,
    G1_LEFT_FOOT_ENDPOINT_OFFSET_M,
    G1_LEFT_PALM_ENDPOINT_OFFSET_M,
    G1_RIGHT_FOOT_ENDPOINT_OFFSET_M,
    G1_RIGHT_PALM_ENDPOINT_OFFSET_M,
    VR3PT_ORIENTATION_POLICY,
    validate_endpoint_candidate_archive,
)


LEFT_PALM_OFFSET = G1_LEFT_PALM_ENDPOINT_OFFSET_M
RIGHT_PALM_OFFSET = G1_RIGHT_PALM_ENDPOINT_OFFSET_M
LEFT_FOOT_OFFSET = G1_LEFT_FOOT_ENDPOINT_OFFSET_M
RIGHT_FOOT_OFFSET = G1_RIGHT_FOOT_ENDPOINT_OFFSET_M
VELOCITY_LIMIT_RAD_S = 6.0
ACCELERATION_LIMIT_RAD_S2 = 120.0
JERK_LIMIT_RAD_S3 = 2400.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def sonic_input_ood_report(
    candidate_smpl_pose: np.ndarray,
    baseline_smpl_pose: np.ndarray,
) -> dict[str, Any]:
    """Report whole-pose deltas and separate arm/lower-body diagnostics.

    The lower-body group verifies that hand alignment preserves the input legs;
    it does not enable foot optimization.
    """

    candidate = np.asarray(candidate_smpl_pose, dtype=np.float64)
    baseline = np.asarray(baseline_smpl_pose, dtype=np.float64)
    if candidate.shape != baseline.shape:
        raise ValueError(
            "candidate/baseline SMPL pose shape mismatch for SONIC-input OOD: "
            f"{candidate.shape} != {baseline.shape}"
        )
    if candidate.ndim != 2 or candidate.shape[1] % 3:
        raise ValueError(
            "SMPL pose must have shape [frames, joints*3] for SONIC-input OOD, "
            f"got {candidate.shape}"
        )
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("SMPL pose contains non-finite values for SONIC-input OOD")

    pose_delta = candidate - baseline
    joint_delta = pose_delta.reshape(len(pose_delta), -1, 3)
    joint_count = joint_delta.shape[1]

    def group_report(indices: tuple[int, ...]) -> dict[str, Any]:
        if not indices or max(indices) >= joint_count:
            raise ValueError(
                f"SMPL pose has {joint_count} joints but OOD group requests {indices}"
            )
        selected = joint_delta[:, indices, :]
        return {
            "joint_indices": list(indices),
            "per_frame_l2_rad": summary(
                np.linalg.norm(selected.reshape(len(selected), -1), axis=1)
            ),
            "per_value_abs_rad": summary(np.abs(selected)),
        }

    whole_pose = {
        "joint_indices": list(range(joint_count)),
        "per_frame_l2_rad": summary(np.linalg.norm(pose_delta, axis=1)),
        "per_value_abs_rad": summary(np.abs(pose_delta)),
    }
    return {
        "metric": "candidate minus same-episode baseline baseline SMPL pose L2 rad",
        # Backward-compatible aliases: these intentionally remain whole pose.
        "per_frame_l2_rad": whole_pose["per_frame_l2_rad"],
        "per_value_abs_rad": whole_pose["per_value_abs_rad"],
        "whole_pose": whole_pose,
        "groups": {
            "hand": group_report(ARM_OPTIMIZED_JOINT_INDICES),
            "spine": group_report((3, 6, 9)),
            "foot": group_report(FOOT_CHAIN_JOINT_INDICES),
        },
        "threshold_status": "pending_current_dataset_smoke_profile",
    }


def kinematic_report(q: np.ndarray, timestamp_s: np.ndarray) -> dict[str, Any]:
    velocity = np.diff(q, axis=0) / np.diff(timestamp_s)[:, None]
    velocity_t = 0.5 * (timestamp_s[1:] + timestamp_s[:-1])
    acceleration = np.diff(velocity, axis=0) / np.diff(velocity_t)[:, None]
    acceleration_t = 0.5 * (velocity_t[1:] + velocity_t[:-1])
    jerk = np.diff(acceleration, axis=0) / np.diff(acceleration_t)[:, None]
    values = {
        "velocity": (np.max(np.abs(velocity), axis=1), VELOCITY_LIMIT_RAD_S),
        "acceleration": (
            np.max(np.abs(acceleration), axis=1),
            ACCELERATION_LIMIT_RAD_S2,
        ),
        "jerk": (np.max(np.abs(jerk), axis=1), JERK_LIMIT_RAD_S3),
    }
    return {
        name: {
            "limit": limit,
            "summary": summary(array),
            "over_limit_fraction": float(np.mean(array > limit)) if len(array) else 0.0,
        }
        for name, (array, limit) in values.items()
    }


def model_contract(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    for name in MUJOCO_BODY_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name}")
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return (
        np.asarray(joint_ids, dtype=np.int64),
        np.asarray(qpos_addresses, dtype=np.int64),
        np.asarray(dof_addresses, dtype=np.int64),
        np.asarray(model.jnt_range[joint_ids], dtype=np.float64),
    )


def body_point(data: mujoco.MjData, body_id: int, offset: np.ndarray) -> np.ndarray:
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset


def forward_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    q: np.ndarray,
    root_position: np.ndarray | None = None,
    root_quaternion: np.ndarray | None = None,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    if root_position is not None:
        data.qpos[:3] = root_position
    if root_quaternion is not None:
        data.qpos[3:7] = root_quaternion / max(np.linalg.norm(root_quaternion), 1e-12)
    data.qpos[qpos_addresses] = q
    mujoco.mj_forward(model, data)


def fk_and_collision_report(
    model: mujoco.MjModel,
    qpos_addresses: np.ndarray,
    q: np.ndarray,
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = mujoco.MjData(model)
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    if min(left_foot_id, right_foot_id) < 0:
        raise ValueError("MuJoCo model is missing ankle bodies")
    feet = np.zeros((len(q), 2, 3), dtype=np.float64)
    pair_counts: dict[str, int] = {}
    frames_with_contact = 0
    for index, row in enumerate(q):
        forward_pose(
            model,
            data,
            qpos_addresses,
            row,
            root_position[index],
            root_quaternion[index],
        )
        feet[index, 0] = body_point(data, left_foot_id, LEFT_FOOT_OFFSET)
        feet[index, 1] = body_point(data, right_foot_id, RIGHT_FOOT_OFFSET)
        if data.ncon:
            frames_with_contact += 1
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or str(contact.geom1)
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or str(contact.geom2)
            key = " <-> ".join(sorted((geom1, geom2)))
            pair_counts[key] = pair_counts.get(key, 0) + 1

    ground_z = float(np.percentile(np.min(feet[:, :, 2], axis=1), 5))
    contact_mask = feet[:, :, 2] <= ground_z + 0.02
    horizontal_step = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2)
    slip_samples = horizontal_step[contact_mask[:-1] & contact_mask[1:]]
    foot_report = {
        "coordinate_contract": "world FK; contact proxy is foot point <= episode ground P05 + 0.02m",
        "ground_z_m": ground_z,
        "contact_frames": {
            "left": int(np.sum(contact_mask[:, 0])),
            "right": int(np.sum(contact_mask[:, 1])),
        },
        "contact_horizontal_step_m": summary(slip_samples),
        "threshold_status": "pending_current_smoke_calibration",
    }
    collision_report = {
        "frames_with_model_contacts": frames_with_contact,
        "frame_fraction": float(frames_with_contact / max(len(q), 1)),
        "geom_pair_counts": dict(sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))),
        "threshold_status": "pending_allowed-contact-pair_profile",
    }
    return foot_report, collision_report


def grounded_anchor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    forward_pose(model, data, qpos_addresses, q)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    foot_z = min(
        body_point(data, left, LEFT_FOOT_OFFSET)[2],
        body_point(data, right, RIGHT_FOOT_OFFSET)[2],
    )
    return np.array([data.xpos[pelvis, 0], data.xpos[pelvis, 1], foot_z])


def reachability_report(
    model: mujoco.MjModel,
    qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray,
    limits: np.ndarray,
    q_seed: np.ndarray,
    target_left_grounded: np.ndarray,
    target_right_grounded: np.ndarray,
    stride: int,
    hand_offsets: tuple[np.ndarray, np.ndarray] = (
        LEFT_PALM_OFFSET,
        RIGHT_PALM_OFFSET,
    ),
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    solve_indices = np.arange(12, 29, dtype=np.int64)
    solve_dofs = dof_addresses[solve_indices]
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    sample_indices = np.unique(np.r_[np.arange(0, len(q_seed), stride), len(q_seed) - 1]).astype(int)
    residuals: list[float] = []
    side_residuals: list[list[float]] = []
    for frame_index in sample_indices:
        q = np.asarray(q_seed[frame_index], dtype=np.float64).copy()
        anchor = grounded_anchor(model, data, qpos_addresses, q)
        targets = (
            (left_body, hand_offsets[0], target_left_grounded[frame_index] + anchor),
            (right_body, hand_offsets[1], target_right_grounded[frame_index] + anchor),
        )
        for _ in range(25):
            forward_pose(model, data, qpos_addresses, q)
            errors: list[np.ndarray] = []
            jacobians: list[np.ndarray] = []
            for body_id, offset, target in targets:
                point = body_point(data, body_id, offset)
                errors.append(target - point)
                mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
                jacobians.append(jacp[:, solve_dofs].copy())
            error = np.concatenate(errors)
            if np.max(np.linalg.norm(np.stack(errors), axis=1)) <= 0.005:
                break
            jacobian = np.vstack(jacobians)
            delta = np.linalg.solve(
                jacobian.T @ jacobian + 1e-3 * np.eye(len(solve_indices)),
                jacobian.T @ error,
            )
            norm = float(np.linalg.norm(delta))
            if norm > 0.08:
                delta *= 0.08 / norm
            q[solve_indices] = np.clip(
                q[solve_indices] + delta,
                limits[solve_indices, 0],
                limits[solve_indices, 1],
            )
        forward_pose(model, data, qpos_addresses, q)
        side = [
            float(np.linalg.norm(target - body_point(data, body_id, offset)))
            for body_id, offset, target in targets
        ]
        side_residuals.append(side)
        residuals.extend(side)
    values = np.asarray(side_residuals, dtype=np.float64)
    tolerance = 0.02
    return {
        "oracle": "independent MuJoCo damped-least-squares FK/IK with model joint limits",
        "sample_stride": stride,
        "sampled_frames": int(len(sample_indices)),
        "side_residual_m": {"left": summary(values[:, 0]), "right": summary(values[:, 1])},
        "tolerance_m": tolerance,
        "reachable_side_fraction": float(np.mean(np.asarray(residuals) <= tolerance)),
        "all_sampled_sides_reachable": bool(np.all(np.asarray(residuals) <= tolerance)),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    candidate = np.load(args.candidate_npz, allow_pickle=False)
    candidate_contract = validate_endpoint_candidate_archive(candidate)
    control = np.load(args.control_npz, allow_pickle=False)
    sim = np.load(args.sim_npz, allow_pickle=False)
    q = np.asarray(control["body_q_measured"], dtype=np.float64)
    declared = [str(item) for item in control["joint_names"]]
    # Capture v2 wrote IsaacLab names but body_q_measured was physically serialized
    # in the canonical MuJoCo order. Preserve that producer recovery contract.
    recovered_names = list(MUJOCO_BODY_JOINT_NAMES)
    if q.shape != (len(control["replay_frame_index"]), len(recovered_names)):
        raise ValueError(f"unexpected measured body_q shape: {q.shape}")
    capture_ns = np.asarray(control["capture_monotonic_ns"], dtype=np.int64)
    timestamp_s = (capture_ns - capture_ns[0]).astype(np.float64) / 1e9
    if len(timestamp_s) < 4 or np.any(np.diff(timestamp_s) <= 0):
        raise ValueError("control capture timestamps must be strictly increasing")

    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    joint_ids, qpos_addresses, dof_addresses, limits = model_contract(model)
    lower_margin = q - limits[:, 0]
    upper_margin = limits[:, 1] - q
    limit_margin = np.minimum(lower_margin, upper_margin)
    finite_inputs = {
        "candidate": bool(
            all(
                np.isfinite(candidate[key]).all()
                for key in candidate.files
                if candidate[key].dtype.kind not in "USO"
            )
        ),
        "control": bool(all(np.isfinite(control[key]).all() for key in control.files if control[key].dtype.kind not in "USO")),
        "sim": bool(all(np.isfinite(sim[key]).all() for key in sim.files if sim[key].dtype.kind not in "USO")),
    }
    foot, collisions = fk_and_collision_report(
        model,
        qpos_addresses,
        np.asarray(sim["body_joint_position"], dtype=np.float64),
        np.asarray(sim["root_position"], dtype=np.float64),
        np.asarray(sim["root_quaternion_wxyz"], dtype=np.float64),
    )
    endpoint_parts = candidate_contract.get("endpoint_parts", ENDPOINT_PARTS_HANDS)
    reachability: dict[str, Any] = reachability_report(
        model,
        qpos_addresses,
        dof_addresses,
        limits,
        q,
        np.asarray(candidate["target_left_grounded_m"], dtype=np.float64),
        np.asarray(candidate["target_right_grounded_m"], dtype=np.float64),
        args.reachability_stride,
        (LEFT_PALM_OFFSET, RIGHT_PALM_OFFSET),
    )
    payload = {
        "format_version": 2,
        "status": "reported_pending_current_smoke_profile",
        "production_accepted": False,
        "inputs": {
            "candidate_npz": {"path": str(args.candidate_npz.resolve()), "sha256": sha256_file(args.candidate_npz)},
            "control_npz": {"path": str(args.control_npz.resolve()), "sha256": sha256_file(args.control_npz)},
            "sim_npz": {"path": str(args.sim_npz.resolve()), "sha256": sha256_file(args.sim_npz)},
            "robot_xml": {"path": str(args.robot_xml.resolve()), "sha256": sha256_file(args.robot_xml)},
        },
        "finite_telemetry": {"checks": finite_inputs, "passed": bool(all(finite_inputs.values()))},
        "candidate_contract": candidate_contract,
        "endpoint_parts": endpoint_parts,
        "joint_order": {"declared_metadata": declared, "used_measured_contract": recovered_names},
        "joint_limits": {
            "minimum_margin_rad": float(np.min(limit_margin)),
            "violating_value_fraction": float(np.mean(limit_margin < 0.0)),
            "passed": bool(np.all(limit_margin >= 0.0)),
        },
        "kinematics": kinematic_report(q, timestamp_s),
        "root": {
            "minimum_height_m": float(np.min(sim["root_position"][:, 2])),
            "all_finite": bool(np.isfinite(sim["root_position"]).all() and np.isfinite(sim["root_quaternion_wxyz"]).all()),
        },
        "foot_slip": foot,
        "self_collision": collisions,
        "reachability": reachability,
        "orientation": {
            "target_policy": (
                (VR3PT_ORIENTATION_POLICY)
            ),
            "metric_status": "pending_coordinate-contract_calibration",
        },
        "sonic_input_ood": sonic_input_ood_report(
            candidate["candidate_smpl_pose"],
            candidate["baseline_smpl_pose"],
        ),
        "pending_before_production_acceptance": [
            "orientation coordinate/profile threshold",
            "foot-slip threshold",
            "allowed self-collision pair profile",
            "dataset SONIC-input OOD threshold",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--control-npz", type=Path, required=True)
    parser.add_argument("--sim-npz", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reachability-stride", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.reachability_stride < 1:
        raise SystemExit("--reachability-stride must be >= 1")
    print(json.dumps(audit(parsed), ensure_ascii=False, indent=2))
