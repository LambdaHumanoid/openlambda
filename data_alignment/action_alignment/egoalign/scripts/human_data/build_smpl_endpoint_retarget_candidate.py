#!/usr/bin/env python3
"""Build one isolated SMPL endpoint-retarget candidate parquet and audit bundle."""

from __future__ import annotations
from egoalign.runtime import (
    human_skeleton_path,
)
from egoalign.alignment import parse_alignment

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.human_data.processing.body_transforms import (
    PICO_FALSE_GRASP_MAX_ISLAND_DURATION_S,
    PICO_FALSE_GRASP_MAX_UPSTREAM_STEP_DEG,
    PICO_FALSE_GRASP_MIN_ENDPOINT_JUMP_DEG,
    PICO_FALSE_GRASP_WRIST_REPAIR_POLICY,
    PICO_WRIST_FILTER_CUTOFF_HZ,
    PICO_WRIST_FILTER_ORDER,
    PICO_WRIST_GRASP_ENTER_THRESHOLD,
    PICO_WRIST_GRASP_EXIT_THRESHOLD,
    PICO_WRIST_GRASP_MAX_DURATION_S,
    PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG,
    PICO_WRIST_GRASP_MIN_RESIDUAL_DEG,
    PICO_WRIST_GRASP_PADDING_S,
    PICO_WRIST_MOTION_FILTER_POLICY,
    compute_smpl_wrist_rotvecs,
    compute_vr3_wrist_orientation_blocks,
    compute_wrist_joints,
    filter_pico_wrist_motion_artifacts,
    repair_pico_false_grasp_wrist_artifacts,
)
from scripts.human_data.smpl_endpoint_retargeting import (
    DEFAULT_CANONICAL_HEAD_HEIGHT_M,
    ENDPOINT_PART_CHOICES,
    ENDPOINT_PARTS_HANDS,
    HAND_ENDPOINT_SIDE_BOTH,
    HAND_ENDPOINT_SIDE_CHOICES,
    LEFT_FOOT_INDEX,
    LEFT_PALM_MCP_INDICES,
    OUTPUT_FROM_SMPL_INTERNAL,
    RIGHT_FOOT_INDEX,
    RIGHT_PALM_MCP_INDICES,
    VR3PT_ORIENTATION_POLICY,
    WRIST_LOCAL_ROTATION_POLICY,
    OptimizationWeights,
    build_virtual_g1_sized_rest_joints,
    canonical_output_joints_with_full,
    direct_big_smpl_endpoint_targets,
    endpoint_retarget_contract,
    endpoint_error_summary,
    forward_joints_torch,
    optimize_virtual_smpl_trajectory,
    output_frame_joints_torch,
    palm_centers_torch,
    propagate_endpoint_delta_to_vr3pt,
    root_centered_grounded_joints,
    scale_canonical_rest_to_head_height,
    validate_endpoint_candidate_archive,
    validate_endpoint_correction_archive,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alignment_interval(sync_dir: Path, episode_index: int) -> tuple[float, float]:
    intervals: list[tuple[float, float]] = []
    for camera in ("gopro", "level-cam"):
        path = sync_dir / f"episode_{episode_index:06d}.{camera}_alignment.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        offset, duration = parse_alignment(payload, require_duration=True)
        intervals.append((offset, offset + duration))
    start = max(value[0] for value in intervals)
    end = min(value[1] for value in intervals)
    if end <= start:
        raise ValueError(f"camera intervals do not overlap: {intervals}")
    return start, end


def aligned_raw_dataframe(
    raw: pd.DataFrame,
    *,
    sync_dir: Path,
    episode_index: int,
    expected_rows: int,
    raw_selection_mode: str = "alignment",
) -> tuple[pd.DataFrame, np.ndarray]:
    if raw_selection_mode == "full":
        indices = np.arange(len(raw), dtype=np.int64)
        selected = raw.reset_index(drop=True)
        if len(selected) != expected_rows:
            raise ValueError(
                f"full raw row count {len(selected)} does not match baseline "
                f"{expected_rows}"
            )
        return selected, indices
    if raw_selection_mode != "alignment":
        raise ValueError(f"unsupported raw_selection_mode={raw_selection_mode!r}")
    timestamps = raw["timestamp"].to_numpy(dtype=np.float64)
    start, end = _alignment_interval(sync_dir, episode_index)
    indices = np.flatnonzero((timestamps >= start) & (timestamps <= end))
    aligned = raw.iloc[indices].reset_index(drop=True)
    if len(aligned) != expected_rows:
        raise ValueError(
            f"aligned raw row count {len(aligned)} does not match baseline {expected_rows}; "
            f"interval=[{start:.6f}, {end:.6f}]"
        )
    return aligned, indices


def _trajectory_quality(values: np.ndarray) -> dict[str, float]:
    xyz = np.asarray(values, dtype=np.float64)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return {
        "finite_fraction": float(np.mean(np.isfinite(xyz).all(axis=1))),
        "nonzero_fraction": float(np.mean(np.linalg.norm(xyz, axis=1) > 1e-6)),
        "path_length_m": float(np.sum(steps)),
        "max_step_m": float(np.max(steps)) if len(steps) else 0.0,
        "p95_step_m": float(np.percentile(steps, 95)) if len(steps) else 0.0,
        "max_displacement_m": float(np.max(np.linalg.norm(xyz - xyz[0], axis=1))),
    }


def _load_human_joint_model() -> tuple[np.ndarray, list[int]]:
    payload = torch.load(
        human_skeleton_path(),
        map_location="cpu",
        weights_only=False,
    )
    return payload["J"].detach().cpu().numpy().astype(np.float64), list(payload["parents_list"])


def _virtual_baseline_endpoints(
    pose: np.ndarray,
    virtual_rest: np.ndarray,
    parents: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_t = torch.as_tensor(pose, dtype=torch.float32)
    rest_t = torch.as_tensor(virtual_rest, dtype=torch.float32)
    with torch.no_grad():
        joints = output_frame_joints_torch(forward_joints_torch(pose_t, rest_t, parents))
        left, right = palm_centers_torch(joints)
    values = joints.cpu().numpy().astype(np.float32)
    return values, left.cpu().numpy().astype(np.float32), right.cpu().numpy().astype(np.float32)


def _invalidate_stale_robot_fields(candidate: pd.DataFrame) -> None:
    zero_columns = {
        "observation.state": 43,
        "observation.eef_state": 14,
        "action.wbc": 43,
        "action.motion_token": 64,
        "teleop.g1_joint_pos": 29,
        "teleop.g1_joint_vel": 29,
    }
    for column, width in zero_columns.items():
        if column in candidate.columns:
            candidate[column] = [np.zeros(width, dtype=np.float64) for _ in range(len(candidate))]


def _stack_raw_array_column(
    frame: pd.DataFrame,
    column: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"raw Pico episode is missing required column {column}")
    try:
        rows: list[np.ndarray] = []
        for value in frame[column]:
            nested = np.asarray(value)
            if nested.dtype == object:
                nested = np.stack(
                    [
                        np.asarray(item, dtype=np.float64)
                        for item in nested.reshape(-1)
                    ]
                )
            rows.append(np.asarray(nested, dtype=np.float64).reshape(shape))
        values = np.stack(rows)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"raw Pico column {column} cannot be reshaped to {shape}"
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError(f"raw Pico column {column} must be finite")
    return values


def _stack_raw_bool_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"raw Pico episode is missing required column {column}")
    values: list[bool] = []
    for value in frame[column]:
        flattened = np.asarray(value).reshape(-1)
        if flattened.shape != (1,):
            raise ValueError(f"raw Pico column {column} must contain scalar booleans")
        values.append(bool(flattened[0]))
    return np.asarray(values, dtype=bool)


def _repair_palm_wrist_baseline(
    aligned_raw: pd.DataFrame,
    source_pose: np.ndarray,
    source_vr3pt_orientation: np.ndarray,
    left_hand_command: np.ndarray,
    right_hand_command: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    """Rebuild palm wrist features after event repair and temporal denoising."""

    source_pico = _stack_raw_array_column(
        aligned_raw, "observation.pico.body_pose", (24, 7)
    )
    left_active = _stack_raw_bool_column(
        aligned_raw, "observation.pico.left_hand_active"
    )
    right_active = _stack_raw_bool_column(
        aligned_raw, "observation.pico.right_hand_active"
    )
    pico_timestamps = aligned_raw["timestamp"].to_numpy(dtype=np.float64)
    false_grasp_repaired_pico, false_grasp_report = (
        repair_pico_false_grasp_wrist_artifacts(
            source_pico,
            left_active,
            right_active,
            pico_timestamps,
        )
    )
    repaired_pico, filter_report = filter_pico_wrist_motion_artifacts(
        false_grasp_repaired_pico,
        left_hand_command,
        right_hand_command,
        pico_timestamps,
    )

    repaired_pose = np.asarray(source_pose, dtype=np.float32).copy()
    repaired_orientation = np.asarray(
        source_vr3pt_orientation, dtype=np.float32
    ).copy()
    combined_mask = np.asarray(false_grasp_report["repair_mask"], dtype=bool) | np.asarray(
        filter_report["filter_mask"], dtype=bool
    )
    converted_wrist_pose = compute_smpl_wrist_rotvecs(repaired_pico)
    converted_wrist_orientation = compute_vr3_wrist_orientation_blocks(
        repaired_pico
    )
    for side_index, (pose_index, orientation_block) in enumerate(
        ((19, slice(0, 6)), (20, slice(6, 12)))
    ):
        side_mask = combined_mask[:, side_index]
        repaired_pose.reshape(-1, 21, 3)[side_mask, pose_index] = (
            converted_wrist_pose[side_mask, side_index]
        )
        repaired_orientation[side_mask, orientation_block] = (
            converted_wrist_orientation[side_mask, side_index]
        )

    return (
        source_pico,
        false_grasp_repaired_pico,
        repaired_pico,
        repaired_pose,
        repaired_orientation,
        false_grasp_report,
        filter_report,
    )


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    contract = endpoint_retarget_contract(
        args.endpoint_parts,
        args.hand_endpoint_side,
    )
    selected_device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    baseline = pd.read_parquet(args.baseline_parquet)
    raw_full = pd.read_parquet(args.raw_parquet)
    aligned_raw, raw_indices = aligned_raw_dataframe(
        raw_full,
        sync_dir=args.sync_dir,
        episode_index=args.source_episode,
        expected_rows=len(baseline),
        raw_selection_mode=args.raw_selection_mode,
    )
    source_baseline_pose = np.stack(
        [np.asarray(value, dtype=np.float32).reshape(63) for value in baseline["teleop.smpl_pose"]]
    )
    baseline_joints = np.stack(
        [np.asarray(value, dtype=np.float32).reshape(24, 3) for value in baseline["teleop.smpl_joints"]]
    )
    pre_swing_baseline_pose = source_baseline_pose.copy()
    source_canonical_rest, parents = _load_human_joint_model()
    canonical_rest, canonical_scale = scale_canonical_rest_to_head_height(
        source_canonical_rest, args.canonical_head_height_m
    )
    if "teleop.vr_3pt_orientation" not in baseline.columns:
        raise ValueError(
            "POSE candidate is missing teleop.vr_3pt_orientation; the current "
            "endpoint contract requires exact optical-orientation preservation"
        )
    source_baseline_vr3pt_orientation = np.stack(
        [
            np.asarray(value, dtype=np.float32).reshape(18)
            for value in baseline["teleop.vr_3pt_orientation"]
        ]
    )
    source_pico_body_pose: np.ndarray | None = None
    false_grasp_repaired_pico_body_pose: np.ndarray | None = None
    repaired_pico_body_pose: np.ndarray | None = None
    false_grasp_repair: dict[str, Any] | None = None
    wrist_motion_filter: dict[str, Any] | None = None
    baseline_left_hand_command: np.ndarray | None = None
    baseline_right_hand_command: np.ndarray | None = None
    baseline_left_hand_command = np.stack(
        [
            np.asarray(value, dtype=np.float32).reshape(-1)[:6]
            for value in baseline["teleop.left_hand_joints"]
        ]
    )
    baseline_right_hand_command = np.stack(
        [
            np.asarray(value, dtype=np.float32).reshape(-1)[:6]
            for value in baseline["teleop.right_hand_joints"]
        ]
    )
    (
        source_pico_body_pose,
        false_grasp_repaired_pico_body_pose,
        repaired_pico_body_pose,
        baseline_pose,
        baseline_vr3pt_orientation,
        false_grasp_repair,
        wrist_motion_filter,
    ) = _repair_palm_wrist_baseline(
        aligned_raw,
        source_baseline_pose,
        source_baseline_vr3pt_orientation,
        baseline_left_hand_command,
        baseline_right_hand_command,
    )
    (
        source_canonical_baseline_joints,
        source_canonical_baseline_full_joints,
    ) = canonical_output_joints_with_full(
        baseline_pose,
        source_canonical_rest,
        parents,
        baseline_joints,
        device=selected_device,
    )
    (
        scaled_baseline_joints,
        scaled_baseline_full_joints,
    ) = canonical_output_joints_with_full(
        baseline_pose,
        canonical_rest,
        parents,
        baseline_joints,
        device=selected_device,
    )
    morphology = json.loads(args.g1_morphology.read_text(encoding="utf-8"))
    virtual_rest, virtual_fit = build_virtual_g1_sized_rest_joints(
        canonical_rest,
        parents,
        morphology,
    )
    virtual_baseline, virtual_left_0, virtual_right_0 = _virtual_baseline_endpoints(
        baseline_pose,
        virtual_rest,
        parents,
    )
    scaled_grounded, _ = root_centered_grounded_joints(scaled_baseline_joints)
    virtual_baseline_grounded, _ = root_centered_grounded_joints(virtual_baseline)
    direct_grounded = {
        "hand_left": scaled_grounded[:, 22].copy(),
        "hand_right": scaled_grounded[:, 23].copy(),
        "foot_left": scaled_grounded[:, LEFT_FOOT_INDEX].copy(),
        "foot_right": scaled_grounded[:, RIGHT_FOOT_INDEX].copy(),
    }
    endpoint_quality = {
        "left": _trajectory_quality(direct_grounded["hand_left"]),
        "right": _trajectory_quality(direct_grounded["hand_right"]),
        "selected_source": (
            "height-scaled canonical big-SMPL four-MCP palm centers in "
            "slots 22/23 after Pico wrist denoising"
        ),
        "comparison_frame": "horizontal pelvis-centered and feet-grounded",
    }

    corrections = {
        name: np.zeros_like(values) for name, values in direct_grounded.items()
    }
    parent_candidate_sha256 = ""
    correction_npz_sha256 = ""
    if (args.correction_npz is None) != (args.initial_candidate_npz is None):
        raise ValueError(
            "correction_npz and initial_candidate_npz must be supplied together; "
            "a closed-loop correction cannot be detached from its parent candidate"
        )

    initial_pose = None
    if args.initial_candidate_npz is not None:
        parent_candidate_sha256 = sha256_file(args.initial_candidate_npz)
        with np.load(args.initial_candidate_npz, allow_pickle=False) as initial_archive:
            initial_contract = validate_endpoint_candidate_archive(initial_archive)
            if initial_contract.get("endpoint_parts", ENDPOINT_PARTS_HANDS) != args.endpoint_parts:
                raise ValueError(
                    "initial candidate endpoint selection differs from this run: "
                    f"{initial_contract.get('endpoint_parts', ENDPOINT_PARTS_HANDS)} "
                    f"!= {args.endpoint_parts}"
                )
            initial_pose = np.asarray(
                initial_archive["candidate_smpl_pose"], dtype=np.float32
            ).copy()

        correction_npz_sha256 = sha256_file(args.correction_npz)
        with np.load(args.correction_npz, allow_pickle=False) as correction:
            validate_endpoint_correction_archive(
                correction,
                expected_parent_candidate_sha256=parent_candidate_sha256,
                expected_shape=direct_grounded["hand_left"].shape,
                expected_endpoint_parts=args.endpoint_parts,
                expected_hand_endpoint_side=args.hand_endpoint_side,
            )
            corrections["hand_left"] = np.asarray(
                correction["left_target_correction_m"], dtype=np.float64
            ).copy()
            corrections["hand_right"] = np.asarray(
                correction["right_target_correction_m"], dtype=np.float64
            ).copy()
    optimization_grounded = {
        name: direct_grounded[name] + corrections[name]
        for name in direct_grounded
    }

    weights = OptimizationWeights(
        position=args.position_weight,
        pose_prior=args.pose_prior_weight,
        velocity=args.velocity_weight,
        acceleration=args.acceleration_weight,
        first_frame=args.first_frame_weight,
    )
    target_left, target_right, big_left_grounded, big_right_grounded = (
        direct_big_smpl_endpoint_targets(
            scaled_baseline_joints, virtual_baseline
        )
    )
    direct_target_left = target_left.copy()
    direct_target_right = target_right.copy()
    target_left = target_left + corrections["hand_left"]
    target_right = target_right + corrections["hand_right"]
    result = optimize_virtual_smpl_trajectory(
        baseline_pose=baseline_pose,
        virtual_rest_joints=virtual_rest,
        parents=parents,
        target_left=target_left,
        target_right=target_right,
        hand_endpoint_side=args.hand_endpoint_side,
        optimized_joint_indices=contract.optimized_joint_indices,
        initial_pose=initial_pose,
        weights=weights,
        ramp_frames=args.ramp_frames,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        device=selected_device,
    )
    candidate_pose = result.pose
    candidate_canonical, candidate_canonical_full_joints = (
        canonical_output_joints_with_full(
            candidate_pose,
            canonical_rest,
            parents,
            baseline_joints,
            device=selected_device,
        )
    )
    with torch.no_grad():
        virtual_joints_t = torch.as_tensor(result.virtual_joints, dtype=torch.float32)
        virtual_left_t, virtual_right_t = palm_centers_torch(virtual_joints_t)
    virtual_left = virtual_left_t.cpu().numpy().astype(np.float32)
    virtual_right = virtual_right_t.cpu().numpy().astype(np.float32)
    virtual_baseline_grounded, _ = root_centered_grounded_joints(virtual_baseline)
    virtual_candidate_grounded, _ = root_centered_grounded_joints(
        result.virtual_joints
    )
    virtual_baseline_endpoints_grounded = {
        "hand_left": np.mean(
            virtual_baseline_grounded[:, list(LEFT_PALM_MCP_INDICES)], axis=1
        ),
        "hand_right": np.mean(
            virtual_baseline_grounded[:, list(RIGHT_PALM_MCP_INDICES)], axis=1
        ),
        "foot_left": virtual_baseline_grounded[:, LEFT_FOOT_INDEX],
        "foot_right": virtual_baseline_grounded[:, RIGHT_FOOT_INDEX],
    }
    virtual_candidate_endpoints_grounded = {
        "hand_left": np.mean(
            virtual_candidate_grounded[:, list(LEFT_PALM_MCP_INDICES)], axis=1
        ),
        "hand_right": np.mean(
            virtual_candidate_grounded[:, list(RIGHT_PALM_MCP_INDICES)], axis=1
        ),
        "foot_left": virtual_candidate_grounded[:, LEFT_FOOT_INDEX],
        "foot_right": virtual_candidate_grounded[:, RIGHT_FOOT_INDEX],
    }

    candidate = baseline.copy()
    candidate["teleop.smpl_pose"] = [row.astype(np.float32) for row in candidate_pose]
    candidate_wire_joints = candidate_canonical.astype(np.float32)
    candidate["teleop.smpl_joints"] = [
        row.reshape(72).astype(np.float32) for row in candidate_wire_joints
    ]
    if "teleop.vr_3pt_position" not in candidate.columns:
        raise ValueError(
            "POSE candidate is missing teleop.vr_3pt_position; SONIC would receive "
            "an endpoint target inconsistent with the retargeted SMPL"
        )
    baseline_vr3pt = np.stack(
        [
            np.asarray(value, dtype=np.float32).reshape(9)
            for value in baseline["teleop.vr_3pt_position"]
        ]
    )
    candidate_vr3pt = propagate_endpoint_delta_to_vr3pt(
        baseline_vr3pt,
        source_canonical_baseline_joints,
        candidate_canonical,
        propagate_head=True,
    )
    candidate["teleop.vr_3pt_position"] = [row for row in candidate_vr3pt]
    candidate["teleop.vr_3pt_orientation"] = [
        row for row in baseline_vr3pt_orientation
    ]
    candidate_vr3pt_orientation = np.stack(
        [
            np.asarray(value, dtype=np.float32).reshape(18)
            for value in candidate["teleop.vr_3pt_orientation"]
        ]
    )
    if not np.array_equal(baseline_vr3pt_orientation, candidate_vr3pt_orientation):
        raise ValueError(
            "endpoint candidate did not preserve the selected baseline "
            "teleop.vr_3pt_orientation"
        )
    left_wrist: list[np.ndarray] = []
    right_wrist: list[np.ndarray] = []
    for pose in candidate_pose:
        left, right = compute_wrist_joints(pose)
        left_wrist.append(left)
        right_wrist.append(right)
    candidate["teleop.left_wrist_joints"] = left_wrist
    candidate["teleop.right_wrist_joints"] = right_wrist
    _invalidate_stale_robot_fields(candidate)

    pose_delta = candidate_pose.reshape(len(candidate_pose), 21, 3) - baseline_pose.reshape(
        len(baseline_pose), 21, 3
    )
    wrist_pose_delta = pose_delta[:, [19, 20]]
    wrist_pose_delta_max = float(np.max(np.abs(wrist_pose_delta)))
    if wrist_pose_delta_max > 1e-7:
        raise ValueError(
            "endpoint optimization modified Pico wrist-local rotation: "
            f"max_abs_delta_rad={wrist_pose_delta_max}"
        )
    active_zero_based = {index - 1 for index in contract.optimized_joint_indices}
    locked_indices = [index for index in range(21) if index not in active_zero_based]
    locked_joint_delta_max = float(
        np.max(np.abs(pose_delta[:, locked_indices]), initial=0.0)
    )
    if locked_joint_delta_max > 1e-7:
        raise ValueError(
            "endpoint optimization modified a locked SMPL joint: "
            f"max_abs_delta_rad={locked_joint_delta_max}"
        )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_parquet(args.output_parquet, index=False)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    wrist_policy = (
        (WRIST_LOCAL_ROTATION_POLICY)
    )
    orientation_policy = (
        (VR3PT_ORIENTATION_POLICY)
    )
    archive_payload: dict[str, np.ndarray] = {
        "algorithm": np.asarray(contract.algorithm),
        "candidate_contract_version": np.asarray(
            contract.candidate_contract_version, dtype=np.int32
        ),
        "endpoint_definition": np.asarray(contract.endpoint_definition),
        "endpoint_target_contract": np.asarray(contract.endpoint_target_contract),
        "hand_endpoint_side": np.asarray(args.hand_endpoint_side),
        "left_palm_mcp_indices": np.asarray(
            LEFT_PALM_MCP_INDICES, dtype=np.int32
        ),
        "right_palm_mcp_indices": np.asarray(
            RIGHT_PALM_MCP_INDICES, dtype=np.int32
        ),
        "optimized_joint_indices": np.asarray(
            contract.optimized_joint_indices, dtype=np.int32
        ),
        "wrist_local_rotation_policy": np.asarray(wrist_policy),
        "wrist_local_rotation_max_abs_delta_rad": np.asarray(
            wrist_pose_delta_max, dtype=np.float64
        ),
        "vr3pt_orientation_policy": np.asarray(orientation_policy),
        "baseline_vr3pt_orientation": baseline_vr3pt_orientation.astype(np.float32),
        "candidate_vr3pt_orientation": candidate_vr3pt_orientation.astype(np.float32),
        "canonical_source_head_height_m": np.asarray(
            canonical_scale["source_head_height_m"], dtype=np.float64
        ),
        "canonical_target_head_height_m": np.asarray(
            canonical_scale["target_head_height_m"], dtype=np.float64
        ),
        "canonical_uniform_scale": np.asarray(
            canonical_scale["uniform_scale"], dtype=np.float64
        ),
        "raw_row_indices": raw_indices.astype(np.int64),
        "timestamp_s": baseline["timestamp"].to_numpy(dtype=np.float64),
        "virtual_baseline_joints": virtual_baseline.astype(np.float32),
        "virtual_candidate_joints": result.virtual_joints.astype(np.float32),
        "source_canonical_baseline_joints": source_canonical_baseline_joints.astype(
            np.float32
        ),
        "source_canonical_baseline_full_joints": (
            source_canonical_baseline_full_joints.astype(np.float32)
        ),
        "canonical_baseline_joints": scaled_baseline_joints.astype(np.float32),
        "canonical_candidate_joints": candidate_canonical.astype(np.float32),
        "baseline_wire_smpl_joints": baseline_joints.astype(np.float32),
        "candidate_wire_smpl_joints": candidate_wire_joints.astype(np.float32),
        "baseline_vr3pt_position": baseline_vr3pt.astype(np.float32),
        "candidate_vr3pt_position": candidate_vr3pt.astype(np.float32),
        "baseline_smpl_pose": baseline_pose.astype(np.float32),
        "candidate_smpl_pose": candidate_pose.astype(np.float32),
        "pre_swing_baseline_smpl_pose": pre_swing_baseline_pose.astype(np.float32),
        "parent_candidate_sha256": np.asarray(parent_candidate_sha256),
        "correction_npz_sha256": np.asarray(correction_npz_sha256),
        "canonical_baseline_full_joints": scaled_baseline_full_joints.astype(
            np.float32
        ),
        "canonical_candidate_full_joints": candidate_canonical_full_joints.astype(
            np.float32
        ),
        "output_from_smpl_internal": OUTPUT_FROM_SMPL_INTERNAL.astype(np.float32),
    }
    if (
        source_pico_body_pose is None
        or false_grasp_repaired_pico_body_pose is None
        or repaired_pico_body_pose is None
        or false_grasp_repair is None
        or wrist_motion_filter is None
        or baseline_left_hand_command is None
        or baseline_right_hand_command is None
    ):
        raise AssertionError("palm wrist filtering proof was not constructed")
    repair_event_bounds = np.asarray(
        [
            [
                int(event["side_index"]),
                int(event["start_frame"]),
                int(event["end_frame_exclusive"]),
            ]
            for event in false_grasp_repair["events"]
        ],
        dtype=np.int32,
    ).reshape(-1, 3)
    grasp_event_bounds = np.asarray(
        [
            [
                int(event["side_index"]),
                int(event["padded_start_frame"]),
                int(event["padded_end_frame_exclusive"]),
            ]
            for event in wrist_motion_filter["events"]
            if bool(event["accepted"])
        ],
        dtype=np.int32,
    ).reshape(-1, 3)
    archive_payload.update(
        {
            "endpoint_parts": np.asarray(ENDPOINT_PARTS_HANDS),
            "wrist_false_grasp_repair_policy": np.asarray(
                PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
            ),
            "wrist_false_grasp_max_island_duration_s": np.asarray(
                PICO_FALSE_GRASP_MAX_ISLAND_DURATION_S, dtype=np.float64
            ),
            "wrist_false_grasp_min_endpoint_jump_deg": np.asarray(
                PICO_FALSE_GRASP_MIN_ENDPOINT_JUMP_DEG, dtype=np.float64
            ),
            "wrist_false_grasp_max_upstream_step_deg": np.asarray(
                PICO_FALSE_GRASP_MAX_UPSTREAM_STEP_DEG, dtype=np.float64
            ),
            "wrist_false_grasp_repair_mask": np.asarray(
                false_grasp_repair["repair_mask"], dtype=bool
            ),
            "wrist_false_grasp_repair_event_bounds": repair_event_bounds,
            "wrist_motion_filter_policy": np.asarray(
                PICO_WRIST_MOTION_FILTER_POLICY
            ),
            "wrist_filter_cutoff_hz": np.asarray(
                PICO_WRIST_FILTER_CUTOFF_HZ, dtype=np.float64
            ),
            "wrist_filter_order": np.asarray(
                PICO_WRIST_FILTER_ORDER, dtype=np.int32
            ),
            "wrist_grasp_enter_threshold": np.asarray(
                PICO_WRIST_GRASP_ENTER_THRESHOLD, dtype=np.float64
            ),
            "wrist_grasp_exit_threshold": np.asarray(
                PICO_WRIST_GRASP_EXIT_THRESHOLD, dtype=np.float64
            ),
            "wrist_grasp_padding_s": np.asarray(
                PICO_WRIST_GRASP_PADDING_S, dtype=np.float64
            ),
            "wrist_grasp_max_duration_s": np.asarray(
                PICO_WRIST_GRASP_MAX_DURATION_S, dtype=np.float64
            ),
            "wrist_grasp_min_residual_deg": np.asarray(
                PICO_WRIST_GRASP_MIN_RESIDUAL_DEG, dtype=np.float64
            ),
            "wrist_grasp_max_upstream_residual_deg": np.asarray(
                PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG,
                dtype=np.float64,
            ),
            "wrist_motion_filter_mask": np.asarray(
                wrist_motion_filter["filter_mask"], dtype=bool
            ),
            "wrist_grasp_repair_mask": np.asarray(
                wrist_motion_filter["grasp_repair_mask"], dtype=bool
            ),
            "wrist_grasp_repair_event_bounds": grasp_event_bounds,
            "pico_timestamp_s": aligned_raw["timestamp"].to_numpy(
                dtype=np.float64
            ),
            "pico_left_hand_active": _stack_raw_bool_column(
                aligned_raw, "observation.pico.left_hand_active"
            ),
            "pico_right_hand_active": _stack_raw_bool_column(
                aligned_raw, "observation.pico.right_hand_active"
            ),
            "baseline_left_hand_command": baseline_left_hand_command.astype(
                np.float32
            ),
            "baseline_right_hand_command": baseline_right_hand_command.astype(
                np.float32
            ),
            "source_pico_body_pose": source_pico_body_pose.astype(np.float32),
            "false_grasp_repaired_pico_body_pose": (
                false_grasp_repaired_pico_body_pose.astype(np.float32)
            ),
            "repaired_pico_body_pose": repaired_pico_body_pose.astype(
                np.float32
            ),
            "source_baseline_smpl_pose": source_baseline_pose.astype(
                np.float32
            ),
            "source_baseline_vr3pt_orientation": (
                source_baseline_vr3pt_orientation.astype(np.float32)
            ),
            "big_smpl_left_endpoint_grounded_m": big_left_grounded.astype(
                np.float32
            ),
            "big_smpl_right_endpoint_grounded_m": big_right_grounded.astype(
                np.float32
            ),
            "left_target_correction_m": corrections["hand_left"].astype(
                np.float32
            ),
            "right_target_correction_m": corrections["hand_right"].astype(
                np.float32
            ),
            "target_left_m": target_left.astype(np.float32),
            "target_right_m": target_right.astype(np.float32),
            "target_left_grounded_m": direct_grounded["hand_left"].astype(
                np.float32
            ),
            "target_right_grounded_m": direct_grounded["hand_right"].astype(
                np.float32
            ),
            "optimization_target_left_grounded_m": optimization_grounded[
                "hand_left"
            ].astype(np.float32),
            "optimization_target_right_grounded_m": optimization_grounded[
                "hand_right"
            ].astype(np.float32),
        }
    )
    if result.staged_proof is None:
        raise ValueError("EgoAlign candidate is missing staged IK proof")
    archive_payload.update(result.staged_proof)
    np.savez_compressed(args.output_npz, **archive_payload)
    with np.load(args.output_npz, allow_pickle=False) as candidate_archive:
        candidate_contract = validate_endpoint_candidate_archive(candidate_archive)

    optimization_baseline_error: dict[str, Any] = endpoint_error_summary(
        virtual_left_0,
        virtual_right_0,
        target_left,
        target_right,
        hand_endpoint_side=args.hand_endpoint_side,
    )
    optimization_candidate_error: dict[str, Any] = endpoint_error_summary(
        virtual_left,
        virtual_right,
        target_left,
        target_right,
        hand_endpoint_side=args.hand_endpoint_side,
    )
    direct_baseline_error: dict[str, Any] = endpoint_error_summary(
        virtual_left_0,
        virtual_right_0,
        direct_target_left,
        direct_target_right,
        hand_endpoint_side=args.hand_endpoint_side,
    )
    direct_candidate_error: dict[str, Any] = endpoint_error_summary(
        virtual_left,
        virtual_right,
        direct_target_left,
        direct_target_right,
        hand_endpoint_side=args.hand_endpoint_side,
    )
    optimization_metrics = {
        "device": selected_device,
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "ramp_frames": (
            (int(args.ramp_frames))
        ),
        "weights": vars(weights).copy(),
        "optimized_joint_indices": list(result.optimized_joint_indices),
        "joint_delta_limits_rad": result.joint_delta_limits_rad.tolist(),
        "loss_trace": result.losses,
        "pose_delta_abs_rad": {
            "mean": float(np.mean(np.abs(pose_delta))),
            "p95": float(np.percentile(np.abs(pose_delta), 95)),
            "max": float(np.max(np.abs(pose_delta))),
        },
        "pico_wrist_local_rotation": {
            "policy": wrist_policy,
            "body_joint_indices": [20, 21],
            "max_abs_delta_rad": wrist_pose_delta_max,
        },
    }
    endpoint_delta_source = (
        "scaled canonical palm-center candidate minus unscaled source "
        "wrist-denoised canonical palm-center baseline slots 22/23 "
        "plus head 15"
    )

    metrics = {
        "format_version": 4,
        "algorithm": contract.algorithm,
        "hand_endpoint_side": args.hand_endpoint_side,
        "source_episode": int(args.source_episode),
        "rows": int(len(candidate)),
        "source": {
            "raw_parquet": str(args.raw_parquet.resolve()),
            "raw_sha256": sha256_file(args.raw_parquet),
            "baseline_parquet": str(args.baseline_parquet.resolve()),
            "baseline_sha256": sha256_file(args.baseline_parquet),
        },
        "output": {
            "candidate_parquet": str(args.output_parquet.resolve()),
            "candidate_sha256": sha256_file(args.output_parquet),
            "candidate_npz": str(args.output_npz.resolve()),
        },
        "endpoint_quality": endpoint_quality,
        "raw_row_indices": {"first": int(raw_indices[0]), "last": int(raw_indices[-1])},
        "raw_selection_mode": args.raw_selection_mode,
        "target_contract": contract.endpoint_target_contract,
        "candidate_contract": candidate_contract,
        "lineage": {
            "parent_candidate_sha256": parent_candidate_sha256 or None,
            "correction_npz_sha256": correction_npz_sha256 or None,
        },
        "canonical_scale": canonical_scale,
        "morphology": {
            "path": str(args.g1_morphology.resolve()),
            "sha256": sha256_file(args.g1_morphology),
            "fit": virtual_fit,
        },
        "optimization": optimization_metrics,
        "optimization_target_error_before": optimization_baseline_error,
        "optimization_target_error_after": optimization_candidate_error,
        "direct_big_smpl_endpoint_error_before": direct_baseline_error,
        "direct_big_smpl_endpoint_error_after": direct_candidate_error,
        "sonic_pose_contract": {
            "vr3pt_position_updated": True,
            "left_position_slice": [0, 3],
            "right_position_slice": [3, 6],
            "head_position_slice": [6, 9],
            "endpoint_delta_source": endpoint_delta_source,
            "head_position_updated": True,
            "orientation_policy": orientation_policy,
        },
        "staging_warning": "robot feedback/action/token columns are zero placeholders; this parquet is not train-ready",
    }
    if false_grasp_repair is None or wrist_motion_filter is None:
        raise AssertionError("palm wrist filtering metrics are unavailable")
    metrics["pico_false_grasp_wrist_repair"] = {
        "policy": false_grasp_repair["policy"],
        "max_island_duration_s": false_grasp_repair[
            "max_island_duration_s"
        ],
        "min_endpoint_jump_deg": false_grasp_repair[
            "min_endpoint_jump_deg"
        ],
        "max_upstream_step_deg": false_grasp_repair[
            "max_upstream_step_deg"
        ],
        "event_count": false_grasp_repair["event_count"],
        "repaired_frame_count": false_grasp_repair[
            "repaired_frame_count"
        ],
        "repaired_side_frame_count": false_grasp_repair[
            "repaired_side_frame_count"
        ],
        "events": false_grasp_repair["events"],
    }
    metrics["pico_wrist_motion_filter"] = {
        key: value
        for key, value in wrist_motion_filter.items()
        if key not in ("filter_mask", "grasp_repair_mask")
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-parts",
        choices=ENDPOINT_PART_CHOICES,
        default=ENDPOINT_PARTS_HANDS,
        help=(
            "Bilateral palm alignment with bounded Pico frame-switch repair "
            "and elbow-local wrist denoising; EgoAlign uses arm-first gated spine assistance."
        ),
    )
    parser.add_argument(
        "--hand-endpoint-side",
        choices=HAND_ENDPOINT_SIDE_CHOICES,
        default=HAND_ENDPOINT_SIDE_BOTH,
        help=(
            "Both palm centers participate in geometric and replay-residual alignment."
        ),
    )
    parser.add_argument("--raw-parquet", type=Path, required=True)
    parser.add_argument("--baseline-parquet", type=Path, required=True)
    parser.add_argument("--sync-dir", type=Path, required=True)
    parser.add_argument("--source-episode", type=int, required=True)
    parser.add_argument(
        "--raw-selection-mode",
        choices=("alignment", "full"),
        default="alignment",
        help=(
            "Select raw rows from camera sync (alignment) or preserve the full "
            "episode (full). The orchestrator derives this from the validated "
            "Stage-A source mapping; do not guess it from row counts."
        ),
    )
    parser.add_argument("--g1-morphology", type=Path, required=True)
    parser.add_argument(
        "--canonical-head-height-m",
        type=float,
        default=DEFAULT_CANONICAL_HEAD_HEIGHT_M,
        help=(
            "Target canonical foot-joint to head-joint height in meters; "
            f"default {DEFAULT_CANONICAL_HEAD_HEIGHT_M}."
        ),
    )
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--correction-npz", type=Path, default=None)
    parser.add_argument("--initial-candidate-npz", type=Path, default=None)
    parser.add_argument("--ramp-frames", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--position-weight", type=float, default=500.0)
    parser.add_argument("--pose-prior-weight", type=float, default=0.6)
    parser.add_argument("--velocity-weight", type=float, default=18.0)
    parser.add_argument("--acceleration-weight", type=float, default=8.0)
    parser.add_argument("--first-frame-weight", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    metrics = build_candidate(parse_args())
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
