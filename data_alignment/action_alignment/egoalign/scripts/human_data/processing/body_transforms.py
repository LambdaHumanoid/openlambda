"""
Body pose transforms for human data processing.

Converts raw PICO body_pose (24, 7) to SMPL format and derived features,
matching the conventions used by SONIC data exporter (run_data_exporter.py).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as sRot, Slerp

# Add project root so we can import egoalign
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from egoalign.runtime import human_skeleton_path
from egoalign.geometry.torch_transforms import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)

try:
    from egoalign.geometry.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None

# SMPL parent joint hierarchy (24 joints)
PARENT_INDICES = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
    18, 19, 20, 22, 23,
][:24]

PICO_FALSE_GRASP_WRIST_REPAIR_POLICY = (
    "hand_active_false_grasp_true_island_bounded_wrist_slerp_v1"
)
PICO_FALSE_GRASP_MAX_ISLAND_DURATION_S = 0.20
PICO_FALSE_GRASP_MIN_ENDPOINT_JUMP_DEG = 20.0
PICO_FALSE_GRASP_MAX_UPSTREAM_STEP_DEG = 8.0

# Offline palm wrist cleanup. Pico's optical body solver can inject both
# frame-to-frame jitter and a slower wrist excursion while the fingers close.
# Work in the elbow-relative wrist frame so genuine shoulder/elbow motion is
# preserved, remove grasp-correlated excursions only when the upstream arm is
# quiet, then apply a zero-phase low-pass so the dataset does not gain latency.
PICO_WRIST_MOTION_FILTER_POLICY = (
    "elbow_local_grasp_guarded_zero_phase_butterworth_v1"
)
PICO_WRIST_FILTER_CUTOFF_HZ = 1.5
PICO_WRIST_FILTER_ORDER = 2
PICO_WRIST_GRASP_ENTER_THRESHOLD = 0.50
PICO_WRIST_GRASP_EXIT_THRESHOLD = 0.35
PICO_WRIST_GRASP_PADDING_S = 0.20
PICO_WRIST_GRASP_MAX_DURATION_S = 2.50
PICO_WRIST_GRASP_MIN_RESIDUAL_DEG = 10.0
PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG = 8.0

# PICO body joint chains: collar, shoulder, elbow, wrist, hand.  The optical
# hand root and body wrist can switch coordinate frames together when hand
# tracking flickers.  Upstream joints remain a useful independent motion gate.
_PICO_ARM_CHAINS = {
    "left": (13, 16, 18, 20, 22),
    "right": (14, 17, 19, 21, 23),
}

# BrainCo binary hand commands (6-DOF)
BRAINCO_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
BRAINCO_CLOSE = np.array([0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

# VR 3pt rotation offsets (from pico_manager_thread_server.py)
_OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),
    sRot.from_euler("xyz", [-90, 0, 180], degrees=True),
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),
]

# Unity-to-robot position transform.  Keep the public name so endpoint
# processing can transform measured PICO joint positions with exactly the same
# convention as the VR3 path instead of duplicating a look-alike matrix.
PICO_TO_ROBOT_POSITION_MATRIX = np.array(
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]], dtype=np.float64
)
_Q = PICO_TO_ROBOT_POSITION_MATRIX


def _quaternion_step_deg(left_xyzw: np.ndarray, right_xyzw: np.ndarray) -> float:
    left = np.asarray(left_xyzw, dtype=np.float64).reshape(4)
    right = np.asarray(right_xyzw, dtype=np.float64).reshape(4)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        raise ValueError("PICO body quaternion has near-zero norm")
    cosine = abs(float(np.dot(left / left_norm, right / right_norm)))
    return float(np.degrees(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))))


def repair_pico_false_grasp_wrist_artifacts(
    body_pose: np.ndarray,
    left_hand_active: np.ndarray,
    right_hand_active: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    max_island_duration_s: float = PICO_FALSE_GRASP_MAX_ISLAND_DURATION_S,
    min_endpoint_jump_deg: float = PICO_FALSE_GRASP_MIN_ENDPOINT_JUMP_DEG,
    max_upstream_step_deg: float = PICO_FALSE_GRASP_MAX_UPSTREAM_STEP_DEG,
) -> tuple[np.ndarray, dict]:
    """Repair wrist-frame jumps caused by a bounded false grasp recognition.

    A repair candidate must be a short, interior ``False -> True -> False``
    ``hand_active`` island: PICO briefly recognizes a grasp when the surrounding
    samples are ungrasped.  Both island boundaries must contain a large
    wrist/hand orientation jump while the collar, shoulder, and elbow stay
    stable.  Only the wrist and hand global quaternions inside that false-grasp
    island are replaced, using timestamped SLERP between the reliable boundary
    frames.  Positions and every other joint are byte-preserved.

    This deliberately does not low-pass ordinary motion, does not alter an
    open-ended or sustained grasp transition, does not repair ``True -> False
    -> True`` tracking loss, and does not guess when only one discontinuity is
    available.
    """

    source = np.asarray(body_pose, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (24, 7) or len(source) == 0:
        raise ValueError(
            f"body_pose must have non-empty shape (N, 24, 7), got {source.shape}"
        )
    if not np.isfinite(source).all():
        raise ValueError("body_pose must be finite")
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if len(timestamps) != len(source) or not np.isfinite(timestamps).all():
        raise ValueError("timestamps_s must be finite and match body_pose rows")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps_s must be strictly increasing")
    if max_island_duration_s <= 0.0:
        raise ValueError("max_island_duration_s must be positive")
    if min_endpoint_jump_deg <= 0.0 or max_upstream_step_deg < 0.0:
        raise ValueError("wrist artifact angle thresholds are invalid")

    active_tracks = {
        "left": np.asarray(left_hand_active, dtype=bool).reshape(-1),
        "right": np.asarray(right_hand_active, dtype=bool).reshape(-1),
    }
    for side, active in active_tracks.items():
        if len(active) != len(source):
            raise ValueError(f"{side}_hand_active must match body_pose rows")

    repaired = source.copy()
    repair_mask = np.zeros((len(source), 2), dtype=bool)
    events: list[dict[str, object]] = []

    for side_index, side in enumerate(("left", "right")):
        active = active_tracks[side]
        collar_index, shoulder_index, elbow_index, wrist_index, hand_index = (
            _PICO_ARM_CHAINS[side]
        )
        start = 0
        while start < len(active):
            end = start + 1
            while end < len(active) and active[end] == active[start]:
                end += 1

            is_bounded_island = (
                start > 0
                and end < len(active)
                and bool(active[start])
                and active[start - 1] == active[end]
                and active[start] != active[start - 1]
            )
            if is_bounded_island:
                duration_s = float(timestamps[end] - timestamps[start - 1])
                boundary_frames = (start, end)
                endpoint_jumps: list[float] = []
                upstream_steps: list[float] = []
                for boundary in boundary_frames:
                    endpoint_jumps.append(
                        max(
                            _quaternion_step_deg(
                                source[boundary - 1, joint, 3:],
                                source[boundary, joint, 3:],
                            )
                            for joint in (wrist_index, hand_index)
                        )
                    )
                    upstream_steps.append(
                        max(
                            _quaternion_step_deg(
                                source[boundary - 1, joint, 3:],
                                source[boundary, joint, 3:],
                            )
                            for joint in (
                                collar_index,
                                shoulder_index,
                                elbow_index,
                            )
                        )
                    )

                should_repair = (
                    duration_s <= float(max_island_duration_s)
                    and min(endpoint_jumps) >= float(min_endpoint_jump_deg)
                    and max(upstream_steps) <= float(max_upstream_step_deg)
                )
                if should_repair:
                    sample_times = timestamps[start:end]
                    anchor_times = timestamps[[start - 1, end]]
                    for joint in (wrist_index, hand_index):
                        anchors = sRot.from_quat(
                            source[[start - 1, end], joint, 3:]
                        )
                        repaired[start:end, joint, 3:] = Slerp(
                            anchor_times, anchors
                        )(sample_times).as_quat()
                    repair_mask[start:end, side_index] = True
                    repaired_steps = [
                        max(
                            _quaternion_step_deg(
                                repaired[frame - 1, joint, 3:],
                                repaired[frame, joint, 3:],
                            )
                            for joint in (wrist_index, hand_index)
                        )
                        for frame in range(start, end + 1)
                    ]
                    events.append(
                        {
                            "side": side,
                            "side_index": side_index,
                            "start_frame": start,
                            "end_frame_exclusive": end,
                            "island_state": bool(active[start]),
                            "duration_s": duration_s,
                            "source_boundary_endpoint_jump_deg": endpoint_jumps,
                            "source_boundary_upstream_step_deg": upstream_steps,
                            "repaired_max_endpoint_step_deg": float(
                                max(repaired_steps, default=0.0)
                            ),
                        }
                    )
            start = end

    report = {
        "policy": PICO_FALSE_GRASP_WRIST_REPAIR_POLICY,
        "max_island_duration_s": float(max_island_duration_s),
        "min_endpoint_jump_deg": float(min_endpoint_jump_deg),
        "max_upstream_step_deg": float(max_upstream_step_deg),
        "repair_mask": repair_mask,
        "events": events,
        "event_count": len(events),
        "repaired_frame_count": int(np.sum(np.any(repair_mask, axis=1))),
        "repaired_side_frame_count": {
            "left": int(np.sum(repair_mask[:, 0])),
            "right": int(np.sum(repair_mask[:, 1])),
        },
    }
    return repaired, report


def _normalize_sign_continuous_xyzw(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0:
        raise ValueError(f"quaternions must have non-empty shape (N, 4), got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-8):
        raise ValueError("quaternions must be finite with non-zero norm")
    values /= norms
    for index in range(1, len(values)):
        if float(np.dot(values[index - 1], values[index])) < 0.0:
            values[index] *= -1.0
    return values


def _quaternion_distance_deg(
    left_xyzw: np.ndarray,
    right_xyzw: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left_xyzw, dtype=np.float64)
    right = np.asarray(right_xyzw, dtype=np.float64)
    if left.shape != right.shape or left.shape[-1] != 4:
        raise ValueError("quaternion distance inputs must have matching (..., 4) shape")
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    cosine = np.abs(np.sum(left * right, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)))


def _slerp_quaternion_span(
    quaternions: np.ndarray,
    timestamps_s: np.ndarray,
    start: int,
    end_inclusive: int,
) -> np.ndarray:
    if start < 0 or end_inclusive >= len(quaternions) or start >= end_inclusive:
        raise ValueError("SLERP span must contain two ordered in-range anchors")
    rotations = sRot.from_quat(quaternions[[start, end_inclusive]])
    values = Slerp(
        timestamps_s[[start, end_inclusive]], rotations
    )(timestamps_s[start : end_inclusive + 1]).as_quat()
    return _normalize_sign_continuous_xyzw(values)


def _zero_phase_quaternion_lowpass(
    quaternions: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Filter a sign-continuous quaternion path without adding time lag."""

    values = _normalize_sign_continuous_xyzw(quaternions)
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if len(values) != len(timestamps):
        raise ValueError("quaternion path and timestamps must have matching rows")
    if len(values) < 2:
        return values
    if cutoff_hz <= 0.0 or order <= 0:
        raise ValueError("wrist low-pass cutoff and order must be positive")

    median_dt = float(np.median(np.diff(timestamps)))
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        raise ValueError("timestamps must be strictly increasing")
    uniform_rows = max(2, int(round((timestamps[-1] - timestamps[0]) / median_dt)) + 1)
    uniform_timestamps = np.linspace(
        timestamps[0], timestamps[-1], uniform_rows, dtype=np.float64
    )
    sample_rate_hz = float((uniform_rows - 1) / (timestamps[-1] - timestamps[0]))
    if cutoff_hz >= 0.5 * sample_rate_hz:
        raise ValueError(
            f"wrist low-pass cutoff {cutoff_hz} Hz must be below Nyquist "
            f"{0.5 * sample_rate_hz} Hz"
        )

    uniform_values = Slerp(timestamps, sRot.from_quat(values))(
        uniform_timestamps
    ).as_quat()
    uniform_values = _normalize_sign_continuous_xyzw(uniform_values)
    sos = butter(
        int(order),
        float(cutoff_hz),
        btype="lowpass",
        fs=sample_rate_hz,
        output="sos",
    )
    # Match scipy's default SOS padding formula, but remain deterministic for
    # the tiny synthetic trajectories used by contract tests.
    ntaps = 2 * len(sos) + 1
    ntaps -= min(
        int(np.sum(sos[:, 2] == 0.0)),
        int(np.sum(sos[:, 5] == 0.0)),
    )
    padlen = min(3 * ntaps, len(uniform_values) - 1)
    filtered = sosfiltfilt(sos, uniform_values, axis=0, padlen=padlen)
    filtered = _normalize_sign_continuous_xyzw(filtered)
    if len(uniform_values) == len(values) and np.array_equal(
        uniform_timestamps, timestamps
    ):
        return filtered
    restored = Slerp(uniform_timestamps, sRot.from_quat(filtered))(
        timestamps
    ).as_quat()
    return _normalize_sign_continuous_xyzw(restored)


def _grasp_hysteresis_runs(
    score: np.ndarray,
    *,
    enter_threshold: float,
    exit_threshold: float,
) -> list[tuple[int, int]]:
    if not 0.0 <= exit_threshold < enter_threshold <= 1.0:
        raise ValueError("grasp thresholds must satisfy 0 <= exit < enter <= 1")
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(score, dtype=np.float64).reshape(-1)):
        if start is None and value >= enter_threshold:
            start = index
        elif start is not None and value <= exit_threshold:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(score)))
    return runs


def filter_pico_wrist_motion_artifacts(
    body_pose: np.ndarray,
    left_hand_command: np.ndarray,
    right_hand_command: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    cutoff_hz: float = PICO_WRIST_FILTER_CUTOFF_HZ,
    filter_order: int = PICO_WRIST_FILTER_ORDER,
    grasp_enter_threshold: float = PICO_WRIST_GRASP_ENTER_THRESHOLD,
    grasp_exit_threshold: float = PICO_WRIST_GRASP_EXIT_THRESHOLD,
    grasp_padding_s: float = PICO_WRIST_GRASP_PADDING_S,
    grasp_max_duration_s: float = PICO_WRIST_GRASP_MAX_DURATION_S,
    grasp_min_residual_deg: float = PICO_WRIST_GRASP_MIN_RESIDUAL_DEG,
    grasp_max_upstream_residual_deg: float = (
        PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG
    ),
) -> tuple[np.ndarray, dict]:
    """Remove Pico-only wrist motion while preserving the upstream arm.

    The source wrist is first expressed relative to its elbow. Finger commands
    provide a grasp cue, but never change the hand command themselves. A
    bounded grasp interval is replaced by local-wrist SLERP only when its wrist
    residual is large and collar/shoulder/elbow residuals stay small. The full
    local wrist trajectory is then filtered with a zero-phase Butterworth
    low-pass. The same rotation delta is applied to the optical hand root, so
    its wrist-to-hand orientation remains unchanged. Positions and every
    upstream joint remain byte-identical.
    """

    source = np.asarray(body_pose, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if source.ndim != 3 or source.shape[1:] != (24, 7) or len(source) == 0:
        raise ValueError(
            f"body_pose must have non-empty shape (N, 24, 7), got {source.shape}"
        )
    if len(timestamps) != len(source) or not np.isfinite(timestamps).all():
        raise ValueError("timestamps_s must be finite and match body_pose rows")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps_s must be strictly increasing")
    if not np.isfinite(source).all():
        raise ValueError("body_pose must be finite")
    if grasp_padding_s < 0.0 or grasp_max_duration_s <= 0.0:
        raise ValueError("grasp padding/duration thresholds are invalid")
    if grasp_min_residual_deg <= 0.0 or grasp_max_upstream_residual_deg < 0.0:
        raise ValueError("grasp rotation residual thresholds are invalid")

    commands = {
        "left": np.asarray(left_hand_command, dtype=np.float64),
        "right": np.asarray(right_hand_command, dtype=np.float64),
    }
    for side, command in commands.items():
        if command.ndim != 2 or command.shape[0] != len(source) or command.shape[1] < 6:
            raise ValueError(
                f"{side}_hand_command must have shape (N, >=6), got {command.shape}"
            )
        if not np.isfinite(command).all():
            raise ValueError(f"{side}_hand_command must be finite")

    repaired = source.copy()
    filter_mask = np.zeros((len(source), 2), dtype=bool)
    grasp_repair_mask = np.zeros((len(source), 2), dtype=bool)
    events: list[dict[str, object]] = []
    correction_by_side: dict[str, np.ndarray] = {}
    source_step_by_side: dict[str, np.ndarray] = {}
    filtered_step_by_side: dict[str, np.ndarray] = {}

    for side_index, side in enumerate(("left", "right")):
        collar_index, shoulder_index, elbow_index, wrist_index, hand_index = (
            _PICO_ARM_CHAINS[side]
        )
        elbow_rotation = sRot.from_quat(source[:, elbow_index, 3:])
        wrist_rotation = sRot.from_quat(source[:, wrist_index, 3:])
        hand_rotation = sRot.from_quat(source[:, hand_index, 3:])
        local_wrist = _normalize_sign_continuous_xyzw(
            (elbow_rotation.inv() * wrist_rotation).as_quat()
        )
        semantic_local_wrist = local_wrist.copy()
        grasp_score = np.max(commands[side][:, 2:6], axis=1)

        for raw_start, raw_end in _grasp_hysteresis_runs(
            grasp_score,
            enter_threshold=grasp_enter_threshold,
            exit_threshold=grasp_exit_threshold,
        ):
            event: dict[str, object] = {
                "side": side,
                "side_index": side_index,
                "start_frame": int(raw_start),
                "end_frame_exclusive": int(raw_end),
                "peak_grasp_score": float(
                    np.max(grasp_score[raw_start:raw_end], initial=0.0)
                ),
                "accepted": False,
            }
            if raw_start == 0 or raw_end >= len(source):
                event["reason"] = "open_boundary"
                events.append(event)
                continue
            padded_start = int(
                np.searchsorted(
                    timestamps,
                    timestamps[raw_start] - float(grasp_padding_s),
                    side="left",
                )
            )
            padded_end = int(
                np.searchsorted(
                    timestamps,
                    timestamps[raw_end - 1] + float(grasp_padding_s),
                    side="right",
                )
                - 1
            )
            padded_start = max(0, padded_start)
            padded_end = min(len(source) - 1, padded_end)
            event["padded_start_frame"] = padded_start
            event["padded_end_frame_exclusive"] = padded_end + 1
            duration_s = float(timestamps[padded_end] - timestamps[padded_start])
            event["padded_duration_s"] = duration_s
            if padded_start >= padded_end:
                event["reason"] = "insufficient_anchors"
                events.append(event)
                continue

            interpolated_wrist = _slerp_quaternion_span(
                local_wrist, timestamps, padded_start, padded_end
            )
            wrist_residual_deg = float(
                np.max(
                    _quaternion_distance_deg(
                        local_wrist[padded_start : padded_end + 1],
                        interpolated_wrist,
                    ),
                    initial=0.0,
                )
            )
            upstream_residual_deg = 0.0
            for upstream_index in (collar_index, shoulder_index, elbow_index):
                upstream = _normalize_sign_continuous_xyzw(
                    source[:, upstream_index, 3:]
                )
                interpolated_upstream = _slerp_quaternion_span(
                    upstream, timestamps, padded_start, padded_end
                )
                upstream_residual_deg = max(
                    upstream_residual_deg,
                    float(
                        np.max(
                            _quaternion_distance_deg(
                                upstream[padded_start : padded_end + 1],
                                interpolated_upstream,
                            ),
                            initial=0.0,
                        )
                    ),
                )
            event["wrist_slerp_residual_max_deg"] = wrist_residual_deg
            event["upstream_slerp_residual_max_deg"] = upstream_residual_deg

            should_repair = (
                duration_s <= float(grasp_max_duration_s)
                and wrist_residual_deg >= float(grasp_min_residual_deg)
                and upstream_residual_deg
                <= float(grasp_max_upstream_residual_deg)
            )
            if should_repair:
                semantic_local_wrist[padded_start : padded_end + 1] = (
                    interpolated_wrist
                )
                grasp_repair_mask[
                    padded_start : padded_end + 1, side_index
                ] = True
                event["accepted"] = True
                event["reason"] = "grasp_correlated_wrist_artifact"
            elif duration_s > float(grasp_max_duration_s):
                event["reason"] = "duration_exceeds_limit"
            elif wrist_residual_deg < float(grasp_min_residual_deg):
                event["reason"] = "wrist_residual_below_threshold"
            else:
                event["reason"] = "upstream_motion_exceeds_limit"
            events.append(event)

        filtered_local_wrist = _zero_phase_quaternion_lowpass(
            semantic_local_wrist,
            timestamps,
            cutoff_hz=float(cutoff_hz),
            order=int(filter_order),
        )
        filtered_wrist_rotation = elbow_rotation * sRot.from_quat(
            filtered_local_wrist
        )
        wrist_delta = filtered_wrist_rotation * wrist_rotation.inv()
        filtered_hand_rotation = wrist_delta * hand_rotation
        wrist_quaternion = filtered_wrist_rotation.as_quat()
        hand_quaternion = filtered_hand_rotation.as_quat()
        source_wrist_quaternion = _normalize_sign_continuous_xyzw(
            source[:, wrist_index, 3:]
        )
        source_hand_quaternion = _normalize_sign_continuous_xyzw(
            source[:, hand_index, 3:]
        )
        wrist_quaternion[
            np.sum(wrist_quaternion * source_wrist_quaternion, axis=1) < 0.0
        ] *= -1.0
        hand_quaternion[
            np.sum(hand_quaternion * source_hand_quaternion, axis=1) < 0.0
        ] *= -1.0
        repaired[:, wrist_index, 3:] = wrist_quaternion
        repaired[:, hand_index, 3:] = hand_quaternion

        correction_deg = _quaternion_distance_deg(
            local_wrist, filtered_local_wrist
        )
        correction_by_side[side] = correction_deg
        filter_mask[:, side_index] = correction_deg > 1e-7
        source_step_by_side[side] = _quaternion_distance_deg(
            local_wrist[:-1], local_wrist[1:]
        )
        filtered_step_by_side[side] = _quaternion_distance_deg(
            filtered_local_wrist[:-1], filtered_local_wrist[1:]
        )

    def _summary(values: np.ndarray) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        return {
            "mean_deg": float(np.mean(array)) if len(array) else 0.0,
            "p95_deg": float(np.percentile(array, 95)) if len(array) else 0.0,
            "max_deg": float(np.max(array, initial=0.0)),
        }

    report = {
        "policy": PICO_WRIST_MOTION_FILTER_POLICY,
        "cutoff_hz": float(cutoff_hz),
        "filter_order": int(filter_order),
        "grasp_enter_threshold": float(grasp_enter_threshold),
        "grasp_exit_threshold": float(grasp_exit_threshold),
        "grasp_padding_s": float(grasp_padding_s),
        "grasp_max_duration_s": float(grasp_max_duration_s),
        "grasp_min_residual_deg": float(grasp_min_residual_deg),
        "grasp_max_upstream_residual_deg": float(
            grasp_max_upstream_residual_deg
        ),
        "filter_mask": filter_mask,
        "grasp_repair_mask": grasp_repair_mask,
        "events": events,
        "accepted_grasp_event_count": int(
            sum(bool(event["accepted"]) for event in events)
        ),
        "filtered_frame_count": int(np.sum(np.any(filter_mask, axis=1))),
        "filtered_side_frame_count": {
            "left": int(np.sum(filter_mask[:, 0])),
            "right": int(np.sum(filter_mask[:, 1])),
        },
        "correction": {
            side: _summary(values) for side, values in correction_by_side.items()
        },
        "source_local_wrist_step": {
            side: _summary(values) for side, values in source_step_by_side.items()
        },
        "filtered_local_wrist_step": {
            side: _summary(values) for side, values in filtered_step_by_side.items()
        },
    }
    return repaired, report


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternion(s) (wxyz) to 6D rotation representation."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1 and q.shape[0] > 4:
        assert q.shape[0] % 4 == 0
        q = q.reshape(-1, 4)
        return quat_to_rot6d(q).ravel().astype(np.float32)

    single = q.ndim == 1
    q = np.atleast_2d(q)
    q_xyzw = q[:, [1, 2, 3, 0]]
    rot_mat = sRot.from_quat(q_xyzw).as_matrix()
    rot_6d = rot_mat[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    if single:
        return rot_6d[0].astype(np.float32)
    return rot_6d.astype(np.float32)


def compute_projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """Compute projected gravity vector from base quaternion (wxyz)."""
    base_quat = np.asarray(base_quat, dtype=np.float64)
    gravity_vec_world = np.array([0.0, 0.0, -1.0])
    base_rotation = sRot.from_quat(base_quat, scalar_first=True)
    return base_rotation.inv().apply(gravity_vec_world).astype(np.float32)


def _quaternion_multiply_np(a, b):
    """Hamilton product of quaternions (wxyz format), batched."""
    aw, ax, ay, az = np.split(a, 4, axis=1)
    bw, bx, by, bz = np.split(b, 4, axis=1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return np.concatenate([ow, ox, oy, oz], axis=1)


def _decompose_rotation_aa(rotation_aa, v2):
    """Decompose axis-angle rotation into twist and swing components."""
    angle = np.linalg.norm(rotation_aa, axis=1)[:, None]
    angle = np.clip(angle, 1e-8, None)
    w = np.cos(angle / 2)
    v = np.sin(angle / 2) * rotation_aa / angle
    q = np.concatenate([w, v], axis=1)
    v_twist = np.dot(v, v2)[:, None] * v2
    q_twist = np.concatenate([w, v_twist], axis=1)
    q_twist = q_twist / np.linalg.norm(q_twist, axis=1)[:, None]
    q_twist_inv = q_twist * np.array([1, -1, -1, -1])
    q_swing = _quaternion_multiply_np(q_twist_inv, q)
    return q_twist, q_swing


def compute_smpl_from_body_pose(
    body_pose_24x7: np.ndarray, device: torch.device = None,
) -> dict:
    """
    Convert raw PICO body_pose (24, 7) to SMPL format.

    Args:
        body_pose_24x7: (24, 7) array, each row [x,y,z, qx,qy,qz,qw] (Unity frame, scalar-last)

    Returns:
        dict with:
          smpl_joints: (72,) float32 — local joint positions flattened (24×3)
          smpl_pose: (63,) float32 — body pose axis-angle (21×3)
          body_quat_w: (4,) float32 — global orientation quaternion (wxyz)
    """
    if device is None:
        device = torch.device("cpu")

    positions = body_pose_24x7[:, :3]
    # xyzw → wxyz
    global_quats = body_pose_24x7[:, [6, 3, 4, 5]]

    global_rots = sRot.from_quat(global_quats, scalar_first=True)
    global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

    local_rots = []
    for i in range(24):
        if PARENT_INDICES[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rot = global_rots[PARENT_INDICES[i]].inv() * global_rots[i]
            local_rots.append(local_rot)

    pose_aa = np.array([rot.as_rotvec() for rot in local_rots])

    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)
    transl = torch.from_numpy(positions[0]).float().to(device).unsqueeze(0)

    # SMPL forward pass
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    if smpl_root_ytoz_up is not None:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
        human_joints_info_path=str(human_skeleton_path()),
    )

    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)

    smpl_joints = smpl_joints_local[0].detach().cpu().numpy().astype(np.float32).flatten()
    smpl_pose = body_pose[0, :63].detach().cpu().numpy().astype(np.float32)
    body_quat_w = global_orient_quat[0].detach().cpu().numpy().astype(np.float32)

    return {
        "smpl_joints": smpl_joints,
        "smpl_pose": smpl_pose,
        "body_quat_w": body_quat_w,
    }


def compute_smpl_wrist_rotvecs(body_pose_nx24x7: np.ndarray) -> np.ndarray:
    """Vectorized Pico-to-SMPL local wrist conversion for contract auditing.

    Returns ``(N, 2, 3)`` in left/right order, matching SMPL body-pose slots
    19/20. This is the wrist-only equivalent of ``compute_smpl_from_body_pose``
    and avoids running an unnecessary SMPL forward pass for every filtered row.
    """

    body = np.asarray(body_pose_nx24x7, dtype=np.float64)
    if body.ndim != 3 or body.shape[1:] != (24, 7) or len(body) == 0:
        raise ValueError(
            f"body pose must have non-empty shape (N, 24, 7), got {body.shape}"
        )
    if not np.isfinite(body).all():
        raise ValueError("body pose must be finite")
    adjustment = sRot.from_euler("y", 180.0, degrees=True)
    result: list[np.ndarray] = []
    for elbow_index, wrist_index in ((18, 20), (19, 21)):
        elbow = sRot.from_quat(body[:, elbow_index, 3:]) * adjustment
        wrist = sRot.from_quat(body[:, wrist_index, 3:]) * adjustment
        result.append((elbow.inv() * wrist).as_rotvec())
    return np.stack(result, axis=1).astype(np.float32)


def _compute_rel_transform(pose, world_frame, scalar_first=False):
    """Transform pose from Unity frame to robot frame."""
    pose = pose.copy()
    world_frame = world_frame.copy()
    pose[:3] = _Q @ pose[:3]
    world_frame[:3] = _Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(_Q @ (rot_base.T @ rot) @ _Q.T)
    rel_pos = sRot.from_matrix(_Q @ rot_base.T @ _Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def compute_vr3_root_relative_position(
    pose_7: np.ndarray,
    root_pose_7: np.ndarray,
) -> np.ndarray:
    """Express one world-space PICO point in SONIC's VR3 root frame.

    This includes the fixed root-device rotation used by ``compute_vr_3pt``.
    Omitting that rotation produces a 90-degree XY error even though the
    ordinary Unity-to-robot transform is otherwise correct.
    """

    pose = np.asarray(pose_7, dtype=np.float64).reshape(7)
    root = np.asarray(root_pose_7, dtype=np.float64).reshape(7)
    if not (np.isfinite(pose).all() and np.isfinite(root).all()):
        raise ValueError("PICO pose/root must be finite 7D values")
    root_rotation_robot = sRot.from_matrix(
        _Q @ sRot.from_quat(root[3:], scalar_first=False).as_matrix() @ _Q.T
    )
    sonic_vr3_root_rotation = root_rotation_robot * _OFFSETS[0]
    robot_displacement = _Q @ (pose[:3] - root[:3])
    return sonic_vr3_root_rotation.inv().apply(robot_displacement).astype(np.float32)


def compute_vr_3pt(body_pose_24x7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract VR 3-point pose (L-Wrist, R-Wrist, Neck) from body_pose.

    Args:
        body_pose_24x7: (24, 7) [x,y,z, qx,qy,qz,qw] Unity frame (scalar-last)

    Returns:
        position: (9,) float32 — [lwrist_xyz, rwrist_xyz, neck_xyz] relative to root
        orientation: (18,) float32 — rot6d of [lwrist, rwrist, neck] relative to root
    """
    body_poses = np.zeros((body_pose_24x7.shape[0], 7), dtype=np.float32)
    for i in range(body_pose_24x7.shape[0]):
        pos, orn = _compute_rel_transform(
            body_pose_24x7[i], np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
            scalar_first=False,
        )
        body_poses[i, :3] = pos
        body_poses[i, 3:] = orn

    kp_poses = np.zeros((4, 7), dtype=np.float32)
    joint_indices = [0, 22, 23, 12]
    for i in range(body_pose_24x7.shape[0]):
        if i not in joint_indices:
            continue
        rel_i = joint_indices.index(i)
        quat = body_poses[i, 3:]
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * _OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )
        kp_poses[rel_i, 3:] = rot_quat
        kp_poses[rel_i, :3] = body_poses[i, :3]

    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()
    for i in range(1, 4):
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    result = kp_poses[1:]  # (3, 7) [lwrist, rwrist, neck], quat is scalar-first
    position = result[:, :3].flatten().astype(np.float32)
    orientation = quat_to_rot6d(result[:, 3:]).flatten().astype(np.float32)
    return position, orientation


def compute_vr3_wrist_orientation_blocks(
    body_pose_nx24x7: np.ndarray,
) -> np.ndarray:
    """Vectorized left/right VR3 wrist orientation conversion.

    Returns ``(N, 2, 6)`` and matches the first twelve values produced by
    ``compute_vr_3pt``. Positions and the head orientation are intentionally
    omitted because wrist filtering is not allowed to change them.
    """

    body = np.asarray(body_pose_nx24x7, dtype=np.float64)
    if body.ndim != 3 or body.shape[1:] != (24, 7) or len(body) == 0:
        raise ValueError(
            f"body pose must have non-empty shape (N, 24, 7), got {body.shape}"
        )
    if not np.isfinite(body).all():
        raise ValueError("body pose must be finite")

    def _robot_rotation(joint_index: int) -> sRot:
        matrices = sRot.from_quat(body[:, joint_index, 3:]).as_matrix()
        transformed = _Q[None, :, :] @ matrices @ _Q.T[None, :, :]
        return sRot.from_matrix(transformed)

    root = _robot_rotation(0) * _OFFSETS[0]
    left = root.inv() * (_robot_rotation(22) * _OFFSETS[1])
    right = root.inv() * (_robot_rotation(23) * _OFFSETS[2])
    left_rot6d = quat_to_rot6d(left.as_quat(scalar_first=True))
    right_rot6d = quat_to_rot6d(right.as_quat(scalar_first=True))
    return np.stack((left_rot6d, right_rot6d), axis=1).astype(np.float32)


def compute_wrist_joints(smpl_pose_63: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute G1 wrist joint angles from SMPL body pose.
    Matches pico_manager_thread_server.py lines 1406-1464.

    Args:
        smpl_pose_63: (63,) axis-angle body pose (21 joints × 3)

    Returns:
        left_wrist: (3,) [roll, pitch, yaw]
        right_wrist: (3,) [roll, pitch, yaw]
    """
    body_pose = smpl_pose_63.reshape(1, 21, 3)

    SMPL_L_ELBOW_IDX = 17
    SMPL_L_WRIST_IDX = 19
    SMPL_R_ELBOW_IDX = 18
    SMPL_R_WRIST_IDX = 20

    smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
    smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
    smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
    smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

    g1_l_elbow_axis = np.array([0, 1, 0])
    _, g1_l_elbow_q_swing = _decompose_rotation_aa(smpl_l_elbow_aa, g1_l_elbow_axis)

    g1_r_elbow_axis = np.array([0, 1, 0])
    _, g1_r_elbow_q_swing = _decompose_rotation_aa(smpl_r_elbow_aa, g1_r_elbow_axis)

    l_elbow_swing_euler = sRot.from_quat(
        g1_l_elbow_q_swing[:, [1, 2, 3, 0]]
    ).as_euler("XYZ", degrees=False)
    r_elbow_swing_euler = sRot.from_quat(
        g1_r_elbow_q_swing[:, [1, 2, 3, 0]]
    ).as_euler("XYZ", degrees=False)

    l_wrist_euler = sRot.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = sRot.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    g1_l_wrist_pitch = -l_wrist_euler[:, 1]
    g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

    g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    g1_r_wrist_pitch = -r_wrist_euler[:, 1]
    g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

    # pico_manager negates l_wrist_pitch when writing to joint_pos
    left_wrist = np.array([
        g1_l_wrist_roll[0], -g1_l_wrist_pitch[0], g1_l_wrist_yaw[0],
    ], dtype=np.float32)
    right_wrist = np.array([
        g1_r_wrist_roll[0], g1_r_wrist_pitch[0], g1_r_wrist_yaw[0],
    ], dtype=np.float32)

    return left_wrist, right_wrist


def hand_status_to_joints(status: int) -> np.ndarray:
    """Convert binary hand status (0=open, 1=close) to brainco joint cmd (6,)."""
    return BRAINCO_CLOSE.copy() if status else BRAINCO_OPEN.copy()
