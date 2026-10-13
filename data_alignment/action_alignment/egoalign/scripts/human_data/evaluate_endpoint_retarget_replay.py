#!/usr/bin/env python3
"""Evaluate measured G1 palm tracking and build the next closed-loop correction."""

from __future__ import annotations
from egoalign.runtime import asset_path

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from egoalign.adapters.sonic.telemetry_schema import (
    MUJOCO_BODY_JOINT_NAMES,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.human_data.smpl_endpoint_retargeting import (
    ENDPOINT_PARTS_HANDS,
    G1_LEFT_FOOT_ENDPOINT_OFFSET_M,
    G1_LEFT_PALM_ENDPOINT_OFFSET_M,
    G1_RIGHT_FOOT_ENDPOINT_OFFSET_M,
    G1_RIGHT_PALM_ENDPOINT_OFFSET_M,
    PICO_FALSE_GRASP_WRIST_REPAIR_POLICY,
    PICO_WRIST_MOTION_FILTER_POLICY,
    WRIST_LOCAL_ROTATION_POLICY,
    HAND_ENDPOINT_SIDE_BOTH,
    endpoint_retarget_contract,
    validate_endpoint_candidate_archive,
)

DEFAULT_ROBOT_XML = (
    asset_path("gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml")
)
LEFT_PALM_OFFSET = G1_LEFT_PALM_ENDPOINT_OFFSET_M
RIGHT_PALM_OFFSET = G1_RIGHT_PALM_ENDPOINT_OFFSET_M
LEFT_FOOT_OFFSET = G1_LEFT_FOOT_ENDPOINT_OFFSET_M
RIGHT_FOOT_OFFSET = G1_RIGHT_FOOT_ENDPOINT_OFFSET_M
MAX_CLOSED_LOOP_REPLAYS = 3


@dataclass(frozen=True)
class EarlyStopThresholds:
    """Auditable acceptance thresholds for ending the closed-loop search."""

    combined_mean_m: float = 0.030
    combined_p95_m: float = 0.045
    combined_max_m: float = 0.060
    per_hand_mean_m: float = 0.035
    min_mean_reduction_percent: float = 75.0
    side_regression_tolerance_m: float = 0.0
    min_active_frames: int = 20
    min_root_height_m: float = 0.50
    max_root_tilt_deg: float = 30.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measured_endpoints_from_control(
    control_path: Path,
    robot_xml: Path,
    *,
    grounded: bool = False,
    hand_position_mode: str = "palm_center",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild palm centers and foot-ground references from measured G1 joints."""

    import mujoco

    control = np.load(control_path, allow_pickle=False)
    declared_joint_names = [str(value) for value in control["joint_names"]]
    if "measured_joint_names" in control.files:
        joint_names = [str(value) for value in control["measured_joint_names"]]
        joint_order_contract = "explicit measured_joint_names"
    else:
        # g1_debug has always serialized physical measured joints in MuJoCo
        # order.  Legacy capture format v2 incorrectly wrote IsaacLab names in
        # the metadata; recover the producer contract explicitly instead of
        # attaching those values to the mislabeled joints.
        joint_names = list(MUJOCO_BODY_JOINT_NAMES)
        joint_order_contract = "legacy g1_debug MuJoCo order (metadata recovered)"
    if tuple(joint_names) != tuple(MUJOCO_BODY_JOINT_NAMES):
        raise ValueError(
            "measured G1 joint order does not match the MuJoCo producer contract: "
            f"{joint_names}"
        )
    body_q = np.asarray(control["body_q_measured"], dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    qpos_addresses: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"control joint not found in MuJoCo model: {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    if left_id < 0 or right_id < 0:
        raise ValueError("wrist_yaw_link body missing from MuJoCo model")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_ankle_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
    )
    right_ankle_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
    )
    if min(pelvis_id, left_ankle_id, right_ankle_id) < 0:
        raise ValueError("pelvis/ankle bodies required for endpoint coordinates")

    if hand_position_mode not in {"palm_center", "wrist_anchor"}:
        raise ValueError(f"unsupported hand_position_mode={hand_position_mode!r}")
    left_hand_offset = (
        LEFT_PALM_OFFSET
        if hand_position_mode == "palm_center"
        else np.zeros(3, dtype=np.float64)
    )
    right_hand_offset = (
        RIGHT_PALM_OFFSET
        if hand_position_mode == "palm_center"
        else np.zeros(3, dtype=np.float64)
    )

    endpoints = {
        name: np.zeros((len(body_q), 3), dtype=np.float64)
        for name in ("hand_left", "hand_right", "foot_left", "foot_right")
    }
    for frame_index, row in enumerate(body_q):
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        data.qpos[np.asarray(qpos_addresses, dtype=np.int64)] = row
        mujoco.mj_forward(model, data)
        endpoints["hand_left"][frame_index] = (
            data.xpos[left_id]
            + data.xmat[left_id].reshape(3, 3) @ left_hand_offset
        )
        endpoints["hand_right"][frame_index] = (
            data.xpos[right_id]
            + data.xmat[right_id].reshape(3, 3) @ right_hand_offset
        )
        endpoints["foot_left"][frame_index] = (
            data.xpos[left_ankle_id]
            + data.xmat[left_ankle_id].reshape(3, 3) @ LEFT_FOOT_OFFSET
        )
        endpoints["foot_right"][frame_index] = (
            data.xpos[right_ankle_id]
            + data.xmat[right_ankle_id].reshape(3, 3) @ RIGHT_FOOT_OFFSET
        )
        if grounded:
            anchor = np.array(
                [
                    data.xpos[pelvis_id, 0],
                    data.xpos[pelvis_id, 1],
                    min(
                        endpoints["foot_left"][frame_index, 2],
                        endpoints["foot_right"][frame_index, 2],
                    ),
                ],
                dtype=np.float64,
            )
            for values in endpoints.values():
                values[frame_index] -= anchor
    info = {
        "rows": int(len(body_q)),
        "joint_names": joint_names,
        "declared_joint_names": declared_joint_names,
        "joint_order_contract": joint_order_contract,
        "legacy_joint_metadata_mislabeled": declared_joint_names != joint_names,
        "control_format_version": int(np.asarray(control["format_version"]).item()),
        "comparison_frame": (
            "horizontal pelvis-centered and feet-grounded" if grounded else "raw MuJoCo FK"
        ),
        "endpoint_offsets_m": {
            "hand_left": left_hand_offset.tolist(),
            "hand_right": right_hand_offset.tolist(),
            "foot_left": LEFT_FOOT_OFFSET.tolist(),
            "foot_right": RIGHT_FOOT_OFFSET.tolist(),
        },
        "hand_position_mode": hand_position_mode,
    }
    return endpoints, info


def measured_palms_from_control(
    control_path: Path,
    robot_xml: Path,
    *,
    grounded: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Backward-compatible palm-only view of measured endpoint FK."""

    endpoints, info = measured_endpoints_from_control(
        control_path, robot_xml, grounded=grounded
    )
    return endpoints["hand_left"], endpoints["hand_right"], info


def estimate_lag_frames(
    target_left: np.ndarray,
    target_right: np.ndarray,
    actual_left: np.ndarray,
    actual_right: np.ndarray,
    *,
    max_lag_frames: int,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> tuple[int, list[dict[str, float]]]:
    endpoint_retarget_contract(ENDPOINT_PARTS_HANDS, hand_endpoint_side)
    target = np.concatenate([target_left, target_right], axis=1)
    actual = np.concatenate([actual_left, actual_right], axis=1)
    target_velocity = np.diff(target, axis=0)
    actual_velocity = np.diff(actual, axis=0)
    rows: list[dict[str, float]] = []
    best_lag = 0
    best_score = float("inf")
    for lag in range(-int(max_lag_frames), int(max_lag_frames) + 1):
        if lag >= 0:
            target_slice = target_velocity[: len(target_velocity) - lag or None]
            actual_slice = actual_velocity[lag:]
        else:
            target_slice = target_velocity[-lag:]
            actual_slice = actual_velocity[: len(actual_velocity) + lag]
        if len(target_slice) < 20:
            continue
        target_centered = target_slice - np.mean(target_slice, axis=0, keepdims=True)
        actual_centered = actual_slice - np.mean(actual_slice, axis=0, keepdims=True)
        numerator = float(np.sum(target_centered * actual_centered))
        denominator = float(
            np.linalg.norm(target_centered) * np.linalg.norm(actual_centered)
        )
        correlation = numerator / denominator if denominator > 1e-12 else 0.0
        scale = float(np.linalg.norm(target_centered) / max(np.linalg.norm(actual_centered), 1e-12))
        rmse = float(
            np.sqrt(np.mean((target_centered - actual_centered * scale) ** 2))
        )
        score = rmse - 0.002 * correlation
        rows.append(
            {"lag_frames": float(lag), "correlation": correlation, "velocity_rmse": rmse}
        )
        if score < best_score:
            best_score = score
            best_lag = lag
    return int(best_lag), rows


def align_actual_to_target(actual: np.ndarray, lag_frames: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(actual, dtype=np.float64)
    aligned = np.full_like(values, np.nan)
    valid = np.zeros(len(values), dtype=bool)
    lag = int(lag_frames)
    if lag >= 0:
        if lag < len(values):
            aligned[: len(values) - lag] = values[lag:]
            valid[: len(values) - lag] = True
    else:
        aligned[-lag:] = values[: len(values) + lag]
        valid[-lag:] = True
    return aligned, valid


def _metric_block(actual: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    residual = target[mask] - actual[mask]
    distance = np.linalg.norm(residual, axis=1)
    return {
        "frames": int(np.sum(mask)),
        "mean_m": float(np.mean(distance)),
        "p95_m": float(np.percentile(distance, 95)),
        "max_m": float(np.max(distance)),
        "axis_mean_abs_m": np.mean(np.abs(residual), axis=0).tolist(),
        "axis_bias_m": np.mean(residual, axis=0).tolist(),
    }


def error_metrics(
    target_left: np.ndarray,
    target_right: np.ndarray,
    actual_left: np.ndarray,
    actual_right: np.ndarray,
    valid: np.ndarray,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> tuple[dict[str, Any], np.ndarray]:
    endpoint_retarget_contract(ENDPOINT_PARTS_HANDS, hand_endpoint_side)
    left_speed = np.r_[0.0, np.linalg.norm(np.diff(target_left, axis=0), axis=1)]
    right_speed = np.r_[0.0, np.linalg.norm(np.diff(target_right, axis=0), axis=1)]
    left_displacement = np.linalg.norm(target_left - target_left[0], axis=1)
    right_displacement = np.linalg.norm(target_right - target_right[0], axis=1)
    speed = np.maximum(left_speed, right_speed)
    displacement = np.maximum(left_displacement, right_displacement)
    active = valid & (
        (speed >= max(0.0015, float(np.percentile(speed[valid], 60))))
        | (displacement >= 0.03)
    )
    if int(np.sum(active)) < 20:
        active = valid.copy()
    metrics: dict[str, Any] = {"active_frames": int(np.sum(active))}
    for side, target, actual in (
        ("left", target_left, actual_left),
        ("right", target_right, actual_right),
    ):
        metrics[side] = {
            "all": _metric_block(actual, target, valid),
            "active": _metric_block(actual, target, active),
        }
    combined_active = np.concatenate(
        [
            np.linalg.norm(target_left[active] - actual_left[active], axis=1),
            np.linalg.norm(target_right[active] - actual_right[active], axis=1),
        ]
    )
    metrics["combined_active"] = {
        "mean_m": float(np.mean(combined_active)),
        "p95_m": float(np.percentile(combined_active, 95)),
        "max_m": float(np.max(combined_active)),
    }
    return metrics, active


def metric_improvement(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Report positive values only when the candidate reduces error."""

    result: dict[str, Any] = {}
    for name in ("mean_m", "p95_m", "max_m"):
        before = float(baseline["combined_active"][name])
        after = float(candidate["combined_active"][name])
        result[name] = {
            "baseline": before,
            "candidate": after,
            "reduction_m": before - after,
            "reduction_percent": 100.0 * (before - after) / max(before, 1e-12),
        }
    return result


def summarize_dynamics(sim_path: Path | None) -> dict[str, Any]:
    """Extract the fail-closed dynamics evidence used by the early-stop gate."""

    if sim_path is None:
        return {
            "available": False,
            "finite": False,
            "reason": "--sim-npz is required for an accepted early stop",
        }
    with np.load(sim_path, allow_pickle=False) as sim:
        required = (
            "root_position",
            "root_quaternion_wxyz",
            "root_linear_velocity",
            "root_angular_velocity",
            "body_joint_position",
            "body_joint_velocity",
        )
        missing = [name for name in required if name not in sim.files]
        if missing:
            return {
                "available": False,
                "finite": False,
                "missing_fields": missing,
            }
        arrays = {name: np.asarray(sim[name], dtype=np.float64) for name in required}

    root = arrays["root_position"]
    quaternion = arrays["root_quaternion_wxyz"]
    finite = bool(all(np.isfinite(values).all() for values in arrays.values()))
    quaternion_norm = np.linalg.norm(quaternion, axis=1)
    valid_quaternion = bool(
        np.isfinite(quaternion_norm).all() and np.all(quaternion_norm > 1e-8)
    )
    if valid_quaternion:
        normalized = quaternion / quaternion_norm[:, None]
        # For wxyz quaternions, R_zz = 1 - 2 * (x^2 + y^2).  The angle
        # between the body vertical and world vertical is the root tilt.
        upright_z = 1.0 - 2.0 * (
            normalized[:, 1] ** 2 + normalized[:, 2] ** 2
        )
        root_tilt_deg = np.degrees(
            np.arccos(np.clip(upright_z, -1.0, 1.0))
        )
        root_tilt_max_deg = float(np.max(root_tilt_deg))
        root_tilt_p95_deg = float(np.percentile(root_tilt_deg, 95))
    else:
        root_tilt_max_deg = float("inf")
        root_tilt_p95_deg = float("inf")
        finite = False

    root_linear_speed = np.linalg.norm(arrays["root_linear_velocity"], axis=1)
    root_vertical_speed = np.abs(arrays["root_linear_velocity"][:, 2])
    root_angular_speed = np.linalg.norm(arrays["root_angular_velocity"], axis=1)
    body_joint_speed = np.abs(arrays["body_joint_velocity"])
    return {
        "available": True,
        "sim_rows": int(len(root)),
        "root_height_min_m": float(np.min(root[:, 2])),
        "root_height_p05_m": float(np.percentile(root[:, 2], 5)),
        "root_tilt_max_deg": root_tilt_max_deg,
        "root_tilt_p95_deg": root_tilt_p95_deg,
        "root_linear_speed_max_mps": float(np.max(root_linear_speed)),
        "root_vertical_speed_max_mps": float(np.max(root_vertical_speed)),
        "root_vertical_speed_p95_mps": float(
            np.percentile(root_vertical_speed, 95)
        ),
        "root_angular_speed_max_radps": float(np.max(root_angular_speed)),
        "body_joint_speed_max_radps": float(np.max(body_joint_speed)),
        "body_joint_speed_p99_radps": float(np.percentile(body_joint_speed, 99)),
        "finite": finite,
        "fell_below_0p5m": bool(np.min(root[:, 2]) < 0.5),
    }


def evaluate_early_stop_gate(
    *,
    baseline_metrics: dict[str, Any],
    metrics: dict[str, Any],
    dynamics: dict[str, Any],
    lag_frames: int,
    max_lag_frames: int,
    round_index: int,
    thresholds: EarlyStopThresholds,
    budget_limit_improvement_only: bool = False,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> dict[str, Any]:
    """Decide whether to accept, continue, or stop after the replay budget."""

    endpoint_retarget_contract(ENDPOINT_PARTS_HANDS, hand_endpoint_side)
    if not 0 <= int(round_index) < MAX_CLOSED_LOOP_REPLAYS:
        raise ValueError(
            f"round_index must be in [0, {MAX_CLOSED_LOOP_REPLAYS - 1}], "
            f"got {round_index}"
        )

    combined = metrics["combined_active"]
    baseline_combined = baseline_metrics["combined_active"]
    left_mean = float(metrics["left"]["active"]["mean_m"])
    right_mean = float(metrics["right"]["active"]["mean_m"])
    baseline_left_mean = float(baseline_metrics["left"]["active"]["mean_m"])
    baseline_right_mean = float(baseline_metrics["right"]["active"]["mean_m"])
    mean_reduction_percent = 100.0 * (
        float(baseline_combined["mean_m"]) - float(combined["mean_m"])
    ) / max(float(baseline_combined["mean_m"]), 1e-12)
    endpoint_values = (
        float(combined["mean_m"]),
        float(combined["p95_m"]),
        float(combined["max_m"]),
        left_mean,
        right_mean,
        mean_reduction_percent,
    )
    lag_inside_search_window = (
        int(max_lag_frames) == 0
        or abs(int(lag_frames)) < abs(int(max_lag_frames))
    )
    checks = {
        "endpoint_metrics_finite": bool(np.isfinite(endpoint_values).all()),
        "combined_mean_within_limit": bool(
            float(combined["mean_m"]) <= thresholds.combined_mean_m
        ),
        "combined_p95_within_limit": bool(
            float(combined["p95_m"]) <= thresholds.combined_p95_m
        ),
        "combined_max_within_limit": bool(
            float(combined["max_m"]) <= thresholds.combined_max_m
        ),
        "left_mean_within_limit": bool(
            left_mean <= thresholds.per_hand_mean_m
        ),
        "right_mean_within_limit": bool(right_mean <= thresholds.per_hand_mean_m),
        "combined_mean_reduction_sufficient": bool(
            mean_reduction_percent >= thresholds.min_mean_reduction_percent
        ),
        "left_not_worse_than_baseline": bool(
            left_mean <= baseline_left_mean + thresholds.side_regression_tolerance_m
        ),
        "right_not_worse_than_baseline": bool(
            right_mean
            <= baseline_right_mean + thresholds.side_regression_tolerance_m
        ),
        "enough_active_frames": bool(
            int(metrics["active_frames"]) >= thresholds.min_active_frames
        ),
        "lag_inside_search_window": lag_inside_search_window,
        "dynamics_available": bool(dynamics.get("available", False)),
        "dynamics_finite": bool(dynamics.get("finite", False)),
        "root_height_within_limit": bool(
            float(dynamics.get("root_height_min_m", float("-inf")))
            >= thresholds.min_root_height_m
        ),
        "root_tilt_within_limit": bool(
            float(dynamics.get("root_tilt_max_deg", float("inf")))
            <= thresholds.max_root_tilt_deg
        ),
    }
    strict_accepted = bool(all(checks.values()))
    replays_used = int(round_index) + 1
    remaining = MAX_CLOSED_LOOP_REPLAYS - replays_used
    budget_limit_checks = {
        "endpoint_metrics_finite": checks["endpoint_metrics_finite"],
        "combined_mean_improved_from_baseline": bool(mean_reduction_percent > 0.0),
        "enough_active_frames": checks["enough_active_frames"],
        "lag_inside_search_window": checks["lag_inside_search_window"],
        "dynamics_available": checks["dynamics_available"],
        "dynamics_finite": checks["dynamics_finite"],
        "root_height_within_limit": checks["root_height_within_limit"],
        "root_tilt_within_limit": checks["root_tilt_within_limit"],
    }
    improvement_only_accepted = bool(
        budget_limit_improvement_only
        and remaining == 0
        and all(budget_limit_checks.values())
    )
    accepted = strict_accepted or improvement_only_accepted
    if accepted:
        decision = (
            "accepted_early_stop" if remaining > 0 else "accepted_at_budget_limit"
        )
        status = "optimized_pass"
    elif remaining == 0:
        decision = "replay_budget_exhausted"
        status = "optimization_fail"
    else:
        decision = "continue"
        status = "in_progress"
    return {
        "format_version": 1,
        "max_closed_loop_replays": MAX_CLOSED_LOOP_REPLAYS,
        "round_index": int(round_index),
        "replays_used": replays_used,
        "replays_remaining": remaining,
        "thresholds": asdict(thresholds),
        "observed": {
            "combined_mean_m": float(combined["mean_m"]),
            "combined_p95_m": float(combined["p95_m"]),
            "combined_max_m": float(combined["max_m"]),
            "left_mean_m": left_mean,
            "right_mean_m": right_mean,
            "combined_mean_reduction_percent": mean_reduction_percent,
            "active_frames": int(metrics["active_frames"]),
            "lag_frames": int(lag_frames),
            "root_height_min_m": dynamics.get("root_height_min_m"),
            "root_tilt_max_deg": dynamics.get("root_tilt_max_deg"),
        },
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "acceptance_profile": (
            "budget_limit_improvement_only"
            if improvement_only_accepted and not strict_accepted
            else "strict_endpoint_thresholds"
        ),
        "budget_limit_improvement_only": {
            "enabled": bool(budget_limit_improvement_only),
            "eligible_this_round": remaining == 0,
            "checks": budget_limit_checks,
            "accepted": improvement_only_accepted,
            "reference": "same-episode baseline baseline measured G1 endpoint metrics",
        },
        "accepted": accepted,
        "status": status,
        "decision": decision,
        "should_stop": decision != "continue",
        "should_generate_next_correction": decision == "continue",
    }


def _fill_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    indices = np.arange(len(result))
    for axis in range(result.shape[1]):
        result[:, axis] = np.interp(indices, indices[valid], result[valid, axis])
    return result


def _smooth(values: np.ndarray, window: int = 21, polyorder: int = 3) -> np.ndarray:
    size = min(int(window), len(values) if len(values) % 2 else len(values) - 1)
    if size < 3:
        return values.copy()
    if size % 2 == 0:
        size -= 1
    return savgol_filter(values, size, min(int(polyorder), size - 1), axis=0, mode="interp")


def _clip_vector_norm(values: np.ndarray, maximum: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    scale = np.minimum(1.0, float(maximum) / np.maximum(norms, 1e-12))
    return result * scale


def smoothstep_ramp(rows: int, ramp_frames: int = 35) -> np.ndarray:
    ramp = np.ones(int(rows), dtype=np.float64)
    count = max(1, min(int(ramp_frames), int(rows)))
    t = np.linspace(0.0, 1.0, count)
    ramp[:count] = t * t * (3.0 - 2.0 * t)
    return ramp


def next_correction(
    *,
    target: np.ndarray,
    actual_aligned: np.ndarray,
    valid: np.ndarray,
    current_correction: np.ndarray,
    previous_actual_aligned: np.ndarray | None,
    previous_correction: np.ndarray | None,
    alpha: float,
    max_increment_m: float,
    max_total_m: float,
    ramp_frames: int = 0,
    gain_mode: str = "diagonal_secant",
) -> tuple[np.ndarray, dict[str, Any]]:
    if gain_mode not in {"unit", "diagonal_secant"}:
        raise ValueError(
            "gain_mode must be 'unit' or 'diagonal_secant', "
            f"got {gain_mode!r}"
        )
    residual = _fill_invalid(target - actual_aligned, valid)
    residual = _smooth(residual)
    gain = np.ones(3, dtype=np.float64)
    observable = np.zeros(3, dtype=bool)
    if (
        gain_mode == "diagonal_secant"
        and previous_actual_aligned is not None
        and previous_correction is not None
    ):
        delta_u = current_correction - previous_correction
        delta_p = actual_aligned - previous_actual_aligned
        joint_valid = valid & np.isfinite(delta_p).all(axis=1)
        for axis in range(3):
            excitation = delta_u[joint_valid, axis]
            response = delta_p[joint_valid, axis]
            denominator = float(np.sum(excitation * excitation))
            if denominator >= 1e-4 and float(np.std(excitation)) >= 0.003:
                estimate = float(np.sum(excitation * response) / denominator)
                if np.isfinite(estimate) and 0.15 <= abs(estimate) <= 3.0:
                    gain[axis] = estimate
                    observable[axis] = True
    increment = float(alpha) * residual / gain[None, :]
    increment = _clip_vector_norm(increment, max_increment_m)
    updated = _clip_vector_norm(current_correction + increment, max_total_m)
    if int(ramp_frames) > 0:
        updated *= smoothstep_ramp(len(updated), int(ramp_frames))[:, None]
    info = {
        "method": "diagonal_secant" if np.any(observable) else "bounded_unit_gain",
        "gain": gain.tolist(),
        "observable_axes": observable.tolist(),
        "alpha": float(alpha),
        "max_increment_m": float(max_increment_m),
        "max_total_m": float(max_total_m),
        "ramp_frames": int(ramp_frames),
        "increment_norm_m": {
            "mean": float(np.mean(np.linalg.norm(increment, axis=1))),
            "max": float(np.max(np.linalg.norm(increment, axis=1))),
        },
    }
    return updated, info


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    candidate = np.load(args.candidate_npz, allow_pickle=False)
    candidate_contract = validate_endpoint_candidate_archive(candidate)
    endpoint_parts = candidate_contract.get("endpoint_parts", ENDPOINT_PARTS_HANDS)
    hand_endpoint_side = candidate_contract.get(
        "hand_endpoint_side", HAND_ENDPOINT_SIDE_BOTH
    )
    contract = endpoint_retarget_contract(endpoint_parts, hand_endpoint_side)
    wrist_policy = WRIST_LOCAL_ROTATION_POLICY
    parent_candidate_sha256 = str(
        np.asarray(candidate["parent_candidate_sha256"]).item()
    )
    previous_arrays: dict[str, np.ndarray] | None = None
    if args.round_index == 0:
        if parent_candidate_sha256 or args.previous_evaluation_npz is not None:
            raise ValueError(
                "round 0 must not carry a parent candidate or previous evaluation"
            )
    else:
        if not parent_candidate_sha256 or args.previous_evaluation_npz is None:
            raise ValueError(
                "rounds 1/2 require a hash-linked parent candidate and previous evaluation"
            )
        with np.load(args.previous_evaluation_npz, allow_pickle=False) as previous:
            previous_contract = {
                "algorithm": contract.algorithm,
                "candidate_contract_version": contract.candidate_contract_version,
                "endpoint_parts": ENDPOINT_PARTS_HANDS,
                "endpoint_definition": contract.endpoint_definition,
                "wrist_false_grasp_repair_policy": (
                    PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
                ),
                "wrist_motion_filter_policy": PICO_WRIST_MOTION_FILTER_POLICY,
                "source_candidate_sha256": parent_candidate_sha256,
            }
            for key, expected in previous_contract.items():
                if key not in previous.files or np.asarray(previous[key]).shape != ():
                    raise ValueError(
                        f"previous evaluation is missing scalar contract field {key}"
                    )
                if np.asarray(previous[key]).item() != expected:
                    raise ValueError(f"previous evaluation has incompatible {key}")
            previous_arrays = {
                key: np.asarray(previous[key], dtype=np.float64).copy()
                for key in (
                    "actual_left_aligned_m",
                    "actual_right_aligned_m",
                    "left_target_correction_m",
                    "right_target_correction_m",
                )
            }
    current_left_correction = np.asarray(
        candidate["left_target_correction_m"], dtype=np.float64
    )
    current_right_correction = np.asarray(
        candidate["right_target_correction_m"], dtype=np.float64
    )

    baseline_left, baseline_right, baseline_info = measured_palms_from_control(
        args.baseline_control_npz,
        args.robot_xml,
        grounded=True,
    )
    actual_left, actual_right, control_info = measured_palms_from_control(
        args.control_npz,
        args.robot_xml,
        grounded=True,
    )
    target_left = np.asarray(candidate["target_left_grounded_m"], dtype=np.float64)
    target_right = np.asarray(candidate["target_right_grounded_m"], dtype=np.float64)
    target_contract = contract.endpoint_target_contract
    algorithm = contract.algorithm
    rows = len(target_left)
    if not all(
        len(values) == rows
        for values in (target_right, baseline_left, baseline_right, actual_left, actual_right)
    ):
        raise ValueError("candidate, baseline control, and candidate control row counts differ")
    baseline_lag_frames, baseline_lag_search = estimate_lag_frames(
        target_left,
        target_right,
        baseline_left,
        baseline_right,
        max_lag_frames=args.max_lag_frames,
        hand_endpoint_side=hand_endpoint_side,
    )
    baseline_left_aligned, baseline_valid_left = align_actual_to_target(
        baseline_left, baseline_lag_frames
    )
    baseline_right_aligned, baseline_valid_right = align_actual_to_target(
        baseline_right, baseline_lag_frames
    )
    baseline_valid = baseline_valid_left & baseline_valid_right
    baseline_metrics, baseline_active = error_metrics(
        target_left,
        target_right,
        baseline_left_aligned,
        baseline_right_aligned,
        baseline_valid,
        hand_endpoint_side=hand_endpoint_side,
    )
    lag_frames, lag_search = estimate_lag_frames(
        target_left,
        target_right,
        actual_left,
        actual_right,
        max_lag_frames=args.max_lag_frames,
        hand_endpoint_side=hand_endpoint_side,
    )
    aligned_left, valid_left = align_actual_to_target(actual_left, lag_frames)
    aligned_right, valid_right = align_actual_to_target(actual_right, lag_frames)
    valid = valid_left & valid_right
    metrics, active = error_metrics(
        target_left,
        target_right,
        aligned_left,
        aligned_right,
        valid,
        hand_endpoint_side=hand_endpoint_side,
    )

    thresholds = EarlyStopThresholds(
        combined_mean_m=args.gate_combined_mean_m,
        combined_p95_m=args.gate_combined_p95_m,
        combined_max_m=args.gate_combined_max_m,
        per_hand_mean_m=args.gate_per_hand_mean_m,
        min_mean_reduction_percent=args.gate_min_mean_reduction_percent,
        side_regression_tolerance_m=args.gate_side_regression_tolerance_m,
        min_active_frames=args.gate_min_active_frames,
        min_root_height_m=args.gate_min_root_height_m,
        max_root_tilt_deg=args.gate_max_root_tilt_deg,
    )
    dynamics = summarize_dynamics(args.sim_npz)
    closed_loop_gate = evaluate_early_stop_gate(
        baseline_metrics=baseline_metrics,
        metrics=metrics,
        dynamics=dynamics,
        lag_frames=lag_frames,
        max_lag_frames=args.max_lag_frames,
        round_index=args.round_index,
        thresholds=thresholds,
        budget_limit_improvement_only=args.gate_budget_limit_improvement_only,
        hand_endpoint_side=hand_endpoint_side,
    )
    closed_loop_gate["force_all_rounds"] = False
    closed_loop_gate["force_all_rounds_applied"] = False

    correction_info: dict[str, Any] | None = None
    next_left = current_left_correction.copy()
    next_right = current_right_correction.copy()
    if args.next_correction_npz is not None:
        if closed_loop_gate["should_generate_next_correction"]:
            previous_left_actual = (
                None
                if previous_arrays is None
                else previous_arrays["actual_left_aligned_m"]
            )
            previous_right_actual = (
                None
                if previous_arrays is None
                else previous_arrays["actual_right_aligned_m"]
            )
            previous_left_correction = (
                None
                if previous_arrays is None
                else previous_arrays["left_target_correction_m"]
            )
            previous_right_correction = (
                None
                if previous_arrays is None
                else previous_arrays["right_target_correction_m"]
            )
            next_left, left_info = next_correction(
                target=target_left,
                actual_aligned=aligned_left,
                valid=valid,
                current_correction=current_left_correction,
                previous_actual_aligned=previous_left_actual,
                previous_correction=previous_left_correction,
                alpha=args.correction_alpha,
                max_increment_m=args.max_correction_increment_m,
                max_total_m=args.max_total_correction_m,
                ramp_frames=args.correction_ramp_frames,
            )
            next_right, right_info = next_correction(
                target=target_right,
                actual_aligned=aligned_right,
                valid=valid,
                current_correction=current_right_correction,
                previous_actual_aligned=previous_right_actual,
                previous_correction=previous_right_correction,
                alpha=args.correction_alpha,
                max_increment_m=args.max_correction_increment_m,
                max_total_m=args.max_total_correction_m,
                ramp_frames=args.correction_ramp_frames,
            )
            args.next_correction_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.next_correction_npz,
                algorithm=np.asarray(contract.algorithm),
                candidate_contract_version=np.asarray(
                    contract.candidate_contract_version, dtype=np.int32
                ),
                endpoint_parts=np.asarray(ENDPOINT_PARTS_HANDS),
                hand_endpoint_side=np.asarray(hand_endpoint_side),
                endpoint_definition=np.asarray(contract.endpoint_definition),
                wrist_local_rotation_policy=np.asarray(wrist_policy),
                wrist_false_grasp_repair_policy=np.asarray(
                    PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
                ),
                wrist_motion_filter_policy=np.asarray(
                    PICO_WRIST_MOTION_FILTER_POLICY
                ),
                source_candidate_sha256=np.asarray(sha256_file(args.candidate_npz)),
                left_target_correction_m=next_left.astype(np.float32),
                right_target_correction_m=next_right.astype(np.float32),
            )
            correction_info = {
                "generated": True,
                "path": str(args.next_correction_npz.resolve()),
                "left": left_info,
                "right": right_info,
            }
        else:
            # The supplied path is a generated handoff artifact.  Remove a
            # stale copy on rerun so a downstream step cannot accidentally
            # launch a fourth or already-unnecessary candidate.
            args.next_correction_npz.unlink(missing_ok=True)
            correction_info = {
                "generated": False,
                "path": str(args.next_correction_npz.resolve()),
                "reason": closed_loop_gate["decision"],
            }

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        algorithm=np.asarray(contract.algorithm),
        candidate_contract_version=np.asarray(
            contract.candidate_contract_version, dtype=np.int32
        ),
        endpoint_parts=np.asarray(ENDPOINT_PARTS_HANDS),
        hand_endpoint_side=np.asarray(hand_endpoint_side),
        endpoint_definition=np.asarray(contract.endpoint_definition),
        wrist_false_grasp_repair_policy=np.asarray(
            PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
        ),
        wrist_motion_filter_policy=np.asarray(PICO_WRIST_MOTION_FILTER_POLICY),
        source_candidate_sha256=np.asarray(sha256_file(args.candidate_npz)),
        target_left_m=target_left.astype(np.float32),
        target_right_m=target_right.astype(np.float32),
        baseline_left_endpoint_m=baseline_left.astype(np.float32),
        baseline_right_endpoint_m=baseline_right.astype(np.float32),
        baseline_left_palm_m=baseline_left.astype(np.float32),
        baseline_right_palm_m=baseline_right.astype(np.float32),
        baseline_left_aligned_m=baseline_left_aligned.astype(np.float32),
        baseline_right_aligned_m=baseline_right_aligned.astype(np.float32),
        baseline_valid_mask=baseline_valid,
        baseline_active_mask=baseline_active,
        baseline_lag_frames=np.asarray(baseline_lag_frames, dtype=np.int32),
        actual_left_palm_m=actual_left.astype(np.float32),
        actual_right_palm_m=actual_right.astype(np.float32),
        actual_left_endpoint_m=actual_left.astype(np.float32),
        actual_right_endpoint_m=actual_right.astype(np.float32),
        actual_left_aligned_m=aligned_left.astype(np.float32),
        actual_right_aligned_m=aligned_right.astype(np.float32),
        valid_mask=valid,
        active_mask=active,
        lag_frames=np.asarray(lag_frames, dtype=np.int32),
        left_target_correction_m=current_left_correction.astype(np.float32),
        right_target_correction_m=current_right_correction.astype(np.float32),
        next_left_target_correction_m=next_left.astype(np.float32),
        next_right_target_correction_m=next_right.astype(np.float32),
    )
    payload = {
        "format_version": 4,
        "algorithm": algorithm,
        "endpoint_parts": ENDPOINT_PARTS_HANDS,
        "hand_endpoint_side": hand_endpoint_side,
        "round": int(args.round_index),
        "candidate_npz": str(args.candidate_npz.resolve()),
        "candidate_npz_sha256": sha256_file(args.candidate_npz),
        "control_npz": str(args.control_npz.resolve()),
        "control_npz_sha256": sha256_file(args.control_npz),
        "baseline_control_npz": str(args.baseline_control_npz.resolve()),
        "baseline_control_npz_sha256": sha256_file(args.baseline_control_npz),
        "endpoint": {
            "left": {
                "body": "left_wrist_yaw_link",
                "local_offset_m": LEFT_PALM_OFFSET.tolist(),
            },
            "right": {
                "body": "right_wrist_yaw_link",
                "local_offset_m": RIGHT_PALM_OFFSET.tolist(),
            },
        },
        "target_contract": target_contract,
        "candidate_contract": candidate_contract,
        "control": control_info,
        "baseline_control": baseline_info,
        "lag": {
            "frames": int(lag_frames),
            "seconds_at_50hz": float(lag_frames / 50.0),
            "search": lag_search,
        },
        "baseline_lag": {
            "frames": int(baseline_lag_frames),
            "seconds_at_50hz": float(baseline_lag_frames / 50.0),
            "search": baseline_lag_search,
        },
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "improvement": metric_improvement(baseline_metrics, metrics),
        "dynamics": dynamics,
        "closed_loop_gate": closed_loop_gate,
        "next_correction": correction_info,
        "output_npz": str(args.output_npz.resolve()),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round-index",
        type=int,
        choices=range(MAX_CLOSED_LOOP_REPLAYS),
        required=True,
    )
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--control-npz", type=Path, required=True)
    parser.add_argument("--baseline-control-npz", type=Path, required=True)
    parser.add_argument("--sim-npz", type=Path, default=None)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--previous-evaluation-npz", type=Path, default=None)
    parser.add_argument("--next-correction-npz", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--max-lag-frames", type=int, default=20)
    parser.add_argument("--correction-alpha", type=float, default=0.65)
    parser.add_argument("--max-correction-increment-m", type=float, default=0.08)
    parser.add_argument("--max-total-correction-m", type=float, default=0.12)
    parser.add_argument(
        "--correction-ramp-frames",
        type=int,
        default=0,
        help=(
            "Optional in-episode ramp for feedback precompensation. Zero applies it "
            "from formal frame 0; replay initialization performs the safe pose ramp."
        ),
    )
    gate = parser.add_argument_group(
        "closed-loop early-stop gate",
        "Accepted rounds do not emit --next-correction-npz. Round 2 always stops.",
    )
    defaults = EarlyStopThresholds()
    gate.add_argument(
        "--gate-combined-mean-m", type=float, default=defaults.combined_mean_m
    )
    gate.add_argument(
        "--gate-combined-p95-m", type=float, default=defaults.combined_p95_m
    )
    gate.add_argument(
        "--gate-combined-max-m", type=float, default=defaults.combined_max_m
    )
    gate.add_argument(
        "--gate-per-hand-mean-m", type=float, default=defaults.per_hand_mean_m
    )
    gate.add_argument(
        "--gate-min-mean-reduction-percent",
        type=float,
        default=defaults.min_mean_reduction_percent,
    )
    gate.add_argument(
        "--gate-side-regression-tolerance-m",
        type=float,
        default=defaults.side_regression_tolerance_m,
    )
    gate.add_argument(
        "--gate-min-active-frames", type=int, default=defaults.min_active_frames
    )
    gate.add_argument(
        "--gate-min-root-height-m", type=float, default=defaults.min_root_height_m
    )
    gate.add_argument(
        "--gate-max-root-tilt-deg", type=float, default=defaults.max_root_tilt_deg
    )
    gate.add_argument(
        "--gate-budget-limit-improvement-only",
        action="store_true",
        help=(
            "Keep strict gates for rounds 00/01; at round 02 accept when combined "
            "mean improves from baseline and finite/dynamics/root gates pass."
        ),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
