#!/usr/bin/env python3
"""Replace synthetic SONIC state with measured C++/MuJoCo feedback.

The raw-Pico converter can compute motion tokens without a robot, but its
historical ``zero``/``closed_loop`` decoder modes do not provide measured
robot feedback.  This post-processing stage consumes telemetry produced by a
real SONIC C++ + MuJoCo replay and rebuilds the state-conditioned fields.

The legacy ``velocity_gate`` mode reuses only the 30 Hz simulator telemetry and
therefore preserves the source motion token.  It is diagnostic-only for new
datasets.  The trainable ``captured_cpp`` mode consumes the per-cycle C++
control capture, writes the captured 64D token into ``action.motion_token``,
and preserves only the source 6D+6D BrainCo action fields.

Measured body state uses the ``isaaclab_offsets`` schema by default: IsaacLab joint order and
default-relative offsets in the 29 body slots, with the 14 hand slots unchanged.

The optional ``captured_cpp`` mode consumes newer ``.control.npz`` telemetry
and uses the policy action captured from the same C++ control tick:

* ``observation.state[t]`` is the simulated robot state captured before the
  action for replay row ``t`` is applied.
* ``action.wbc[t]`` uses either the next measured state (default lightweight
  mode) or the policy action captured by C++ (diagnostic mode).
* ``action.motion_token[t]`` uses the token from the same captured C++ control
  tick in ``captured_cpp`` mode; no manual 10/11-row shift is applied.
* The ONNX decoder is never run a second time with invented startup history.

Videos and unchanged parquet columns are hard-linked into a new dataset.  The
source dataset is never modified.
"""

from __future__ import annotations

from egoalign.io_safety import validate_output_root

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egoalign.datasets.features import get_g1_robot_model  # noqa: E402
from egoalign.adapters.sonic.rollout_contract import (  # noqa: E402
    MIN_VALID_ROOT_HEIGHT_M,
    sha256_file,
    SIM_EPISODE_LIFECYCLE_CONTRACT,
    validate_episode_rollout_contract,
)
from egoalign.geometry.numpy_transforms import (  # noqa: E402
    compute_projected_gravity,
)
from egoalign.adapters.sonic.state_space import (  # noqa: E402
    G1_BODY_DIM as POLICY_ACTION_DIM,
    G1_DEFAULT_ANGLES_MUJOCO as DEFAULT_ANGLES_MUJOCO,
    G1_MUJOCO_TO_ISAACLAB as MUJOCO_TO_ISAACLAB,
    physical_qpos_to_isaaclab_offset_state,
)
from scripts.human_data.processing.wrist_contract import (  # noqa: E402
    POSE_WRIST_CONTRACT,
)


BODY_STATE_INDICES = np.asarray(
    [*range(0, 22), *range(29, 36)], dtype=np.int64
)
LEFT_HAND_STATE_INDICES = np.arange(22, 29, dtype=np.int64)
RIGHT_HAND_STATE_INDICES = np.arange(36, 43, dtype=np.int64)
CHANGED_COLUMNS = (
    "observation.state",
    "observation.eef_state",
    "action.motion_token",
    "action.wbc",
    "observation.root_orientation",
    "observation.projected_gravity",
    "observation.init_base_quat",
    "teleop.g1_joint_pos",
    "teleop.g1_joint_vel",
)
LEARNED_ACTION_COLUMNS = (
    ("action.motion_token", 64),
    ("teleop.left_hand_joints", 6),
    ("teleop.right_hand_joints", 6),
)
HAND_ACTION_COLUMNS = (
    ("teleop.left_hand_joints", 6),
    ("teleop.right_hand_joints", 6),
)
FEEDBACK_MODES = ("velocity_gate", "captured_cpp")
STATE_SCHEMAS = ("isaaclab_offsets", "physical_qpos")
# Stage B owns this SONIC/MuJoCo rollout contract.  The velocity gate consumes
# validated telemetry downstream; it does not define the conversion lifecycle.
STAGE_B_ROLLOUT_CONFIG = {
    "replay_protocol": "pose",
    "ramp_seconds": 5.0,
    "release_elastic_band": True,
    "reset_sim": False,
    "sim_prepared_before_deploy": True,
    "planner_settle_ticks": 10,
    "sim_control_settle_seconds": 1.0,
    "sim_stability_timeout": 15.0,
    "sim_stability_poll_hz": 10.0,
    "sim_stability_hold_seconds": 0.5,
    "sim_max_root_linear_speed": 0.10,
    "sim_max_root_angular_speed": 0.35,
    "sim_max_body_joint_speed": 1.0,
}
# Formal replay is 50 Hz.  The source/replay/control indices below are the
# authoritative correspondence proof; host scheduling jitter changes the
# send-to-capture *spread* without changing which C++ control tick belongs to
# a row.  Keep a hard absolute cap of 1.5 replay periods so a genuinely late
# capture still fails closed.
MAX_SEND_TO_CAPTURE_LATENCY_S = 0.030


def _stack_fixed_list(table: pa.Table, key: str, size: int, dtype=np.float64) -> np.ndarray:
    column = table[key].combine_chunks()
    values = np.asarray(column.values, dtype=dtype)
    return values.reshape(len(table), size)


def _fixed_list(values: np.ndarray, value_type: pa.DataType) -> pa.FixedSizeListArray:
    values = np.ascontiguousarray(values)
    flat = pa.array(values.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _replace_fixed_list(table: pa.Table, key: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(key)
    if index < 0:
        raise KeyError(f"missing parquet column: {key}")
    field = table.schema.field(index)
    if pa.types.is_fixed_size_list(field.type):
        list_size = field.type.list_size
        replacement = _fixed_list(values, field.type.value_type)
    elif pa.types.is_list(field.type):
        lengths = np.asarray(column_lengths(table[key]), dtype=np.int64)
        if len(lengths) != len(table) or len(set(lengths.tolist())) != 1:
            raise TypeError(f"{key} has non-uniform list lengths: {field.type}")
        list_size = int(lengths[0])
        replacement = pa.array(values.tolist(), type=field.type)
    else:
        raise TypeError(f"{key} is not a list: {field.type}")
    if values.shape != (len(table), list_size):
        raise ValueError(
            f"{key} replacement shape {values.shape} != "
            f"({len(table)}, {list_size})"
        )
    return table.set_column(index, field, replacement)


def column_lengths(column: pa.ChunkedArray) -> list[int]:
    """Return row lengths without changing replay-only variable-list schema."""
    return [len(value) if value is not None else -1 for value in column.to_pylist()]


def _interpolate_linear(
    source_time_ns: np.ndarray,
    values: np.ndarray,
    target_time_ns: np.ndarray,
) -> np.ndarray:
    source_time = np.asarray(source_time_ns, dtype=np.float64)
    target_time = np.asarray(target_time_ns, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if source_time.ndim != 1 or len(source_time) != len(values):
        raise ValueError("telemetry timestamps and values have incompatible shapes")
    if len(source_time) < 2 or np.any(np.diff(source_time) <= 0):
        raise ValueError("telemetry timestamps must be strictly increasing")
    return np.column_stack(
        [np.interp(target_time, source_time, values[:, dim]) for dim in range(values.shape[1])]
    )


def _interpolate_quaternion_wxyz(
    source_time_ns: np.ndarray,
    quaternions: np.ndarray,
    target_time_ns: np.ndarray,
) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"quaternion telemetry must be [N, 4], got {quaternions.shape}")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("telemetry contains a zero quaternion")
    quaternions /= norms
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    result = _interpolate_linear(source_time_ns, quaternions, target_time_ns)
    result /= np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-8)
    return result


def _telemetry_dir(root: Path) -> Path:
    return root / "telemetry" if (root / "telemetry").is_dir() else root


def _resolve_telemetry_stem(
    root: Path,
    episode_index: int,
    *,
    require_control: bool,
) -> tuple[Path, str]:
    """Resolve one coherent Stage-B telemetry triplet for an episode.

    Ordinary full rollouts use ``episode_XXXXXX``.  A bounded smoke rollout
    uses the same source episode but names its replay artefacts
    ``episode_XXXXXX_first_YYYYYY`` so the video makes the truncation explicit.
    Treat that suffix as part of one telemetry stem instead of falling back to
    a different episode or silently mixing files from separate attempts.
    """

    telemetry_dir = _telemetry_dir(root)
    prefix = f"episode_{episode_index:06d}"
    reference_suffix = ".reference.npz"
    candidate_stems: list[str] = []
    for reference_path in sorted(telemetry_dir.glob(f"{prefix}*.reference.npz")):
        stem = reference_path.name[: -len(reference_suffix)]
        # Do not let episode 000001 accidentally match a similarly-prefixed
        # name such as episode_0000010.  Supported bounded-rollout suffixes
        # always start with an underscore.
        if stem != prefix and not stem.startswith(f"{prefix}_"):
            continue
        sim_path = telemetry_dir / f"{stem}.sim.npz"
        control_path = telemetry_dir / f"{stem}.control.npz"
        if not sim_path.is_file() or (require_control and not control_path.is_file()):
            continue
        candidate_stems.append(stem)

    if not candidate_stems:
        required = "sim/reference/control" if require_control else "sim/reference"
        raise FileNotFoundError(
            f"episode {episode_index}: missing coherent rollout telemetry "
            f"({required}) under {telemetry_dir}"
        )
    if len(candidate_stems) != 1:
        raise ValueError(
            f"episode {episode_index}: ambiguous rollout telemetry stems under "
            f"{telemetry_dir}: {candidate_stems}"
        )
    return telemetry_dir, candidate_stems[0]


def _telemetry_paths(
    root: Path,
    episode_index: int,
    *,
    require_control: bool = True,
) -> tuple[Path, Path, Path]:
    telemetry_dir, stem = _resolve_telemetry_stem(
        root,
        episode_index,
        require_control=require_control,
    )
    return (
        telemetry_dir / f"{stem}.sim.npz",
        telemetry_dir / f"{stem}.reference.npz",
        telemetry_dir / f"{stem}.control.npz",
    )


def _rollout_contract_path(
    root: Path,
    episode_index: int,
    *,
    require_control: bool = True,
) -> Path:
    telemetry_dir, stem = _resolve_telemetry_stem(
        root,
        episode_index,
        require_control=require_control,
    )
    return telemetry_dir / f"{stem}.rollout.json"


def validate_stage_b_rollout(
    telemetry_root: Path,
    episode_indices: Iterable[int],
    *,
    require_real_wrist_contract: bool = False,
) -> tuple[Path, dict]:
    """Require one complete, deterministic Stage-B conversion rollout."""

    candidates = [telemetry_root / "summary.json"]
    if telemetry_root.name == "telemetry":
        candidates.append(telemetry_root.parent / "summary.json")
    summary_path = next((path for path in candidates if path.is_file()), None)
    if summary_path is None:
        raise FileNotFoundError(
            "rollout summary.json not found; expected Stage-B batch replay "
            f"output under {telemetry_root}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("sim_episode_lifecycle_contract")
        != SIM_EPISODE_LIFECYCLE_CONTRACT
    ):
        raise ValueError(
            "rollout is missing the deterministic per-episode lifecycle "
            f"contract {SIM_EPISODE_LIFECYCLE_CONTRACT!r}"
        )
    if (
        require_real_wrist_contract
        and summary.get("pose_wrist_contract") != POSE_WRIST_CONTRACT
    ):
        raise ValueError(
            "rollout is missing the canonical real-wrist pose contract; "
            "zero-wrist or legacy telemetry cannot produce trainable tokens"
        )
    config = summary.get("config", {})
    mismatches: list[str] = []
    for key, expected in STAGE_B_ROLLOUT_CONFIG.items():
        actual = config.get(key)
        if key == "replay_protocol" and actual == "auto":
            resolved = summary.get("resolved_protocols", {})
            matches = set(resolved) == {expected} and int(resolved[expected]) > 0
        elif isinstance(expected, float):
            matches = actual is not None and np.isclose(float(actual), expected)
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ValueError(
            "rollout did not use the required Stage-B SONIC startup strategy: "
            + ", ".join(mismatches)
        )

    expected_episodes = {int(index) for index in episode_indices}
    successful_episodes = {
        int(index) for index in summary.get("successful_episodes", [])
    }
    successful_episodes.update(
        int(index) for index in summary.get("skipped_episodes", [])
    )
    failed_episodes = [int(index) for index in summary.get("failed_episodes", [])]
    missing = sorted(expected_episodes - successful_episodes)
    if failed_episodes or missing:
        raise ValueError(
            "rollout is incomplete: "
            f"failed={failed_episodes}, missing_successful={missing}"
        )
    return summary_path.resolve(), summary


# Compatibility for older one-off scripts.  New conversion code uses the
# Stage-B name so rollout ownership is not confused with velocity scoring.
validate_velocity_gate_rollout = validate_stage_b_rollout


def load_feedback(
    telemetry_root: Path,
    episode_index: int,
    frame_count: int,
    *,
    expected_smpl_joints: np.ndarray | None = None,
    expected_motion_tokens: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    sim_path, reference_path, control_path = _telemetry_paths(
        telemetry_root, episode_index
    )
    if not sim_path.is_file() or not reference_path.is_file() or not control_path.is_file():
        raise FileNotFoundError(
            f"episode {episode_index}: missing rollout telemetry: "
            f"sim={sim_path}, reference={reference_path}, control={control_path}"
        )
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        control_path, allow_pickle=False
    ) as control:
        replay_index = np.asarray(reference["replay_frame_index"], dtype=np.int64)
        expected_index = np.arange(frame_count, dtype=np.int64)
        if replay_index.shape != expected_index.shape or not np.array_equal(
            replay_index, expected_index
        ):
            raise ValueError(
                f"episode {episode_index}: reference telemetry does not cover exactly "
                f"0..{frame_count - 1} (got {len(replay_index)} rows)"
            )
        if expected_smpl_joints is not None:
            expected_smpl_joints = np.asarray(expected_smpl_joints, dtype=np.float64)
            recorded_smpl_joints = np.asarray(
                reference["reference_smpl_joints"], dtype=np.float64
            )
            if (
                recorded_smpl_joints.shape != expected_smpl_joints.shape
                or not np.array_equal(recorded_smpl_joints, expected_smpl_joints)
            ):
                raise ValueError(
                    f"episode {episode_index}: replay telemetry belongs to a "
                    "different SMPL trajectory"
                )

        control_replay_index = np.asarray(
            control["replay_frame_index"], dtype=np.int64
        )
        if not np.array_equal(control_replay_index, expected_index):
            raise ValueError(
                f"episode {episode_index}: control telemetry does not cover exactly "
                f"0..{frame_count - 1} (got {len(control_replay_index)} rows)"
            )

        control_index = np.asarray(control["index"], dtype=np.float64)
        if control_index.shape != (frame_count,) or not np.isfinite(control_index).all():
            raise ValueError(
                f"episode {episode_index}: invalid C++ control index telemetry"
            )
        if frame_count > 1 and not np.all(np.diff(control_index) == 1.0):
            unique_steps, counts = np.unique(
                np.diff(control_index), return_counts=True
            )
            steps = {float(step): int(count) for step, count in zip(unique_steps, counts)}
            raise ValueError(
                f"episode {episode_index}: control telemetry is not one C++ tick "
                f"per replay row: steps={steps}"
            )

        local_frame_index = np.asarray(control["local_frame_index"], dtype=np.int64)
        if local_frame_index.shape != (frame_count,):
            raise ValueError(
                f"episode {episode_index}: invalid sender frame-index telemetry"
            )
        if frame_count > 1 and not np.all(np.diff(local_frame_index) == 1):
            raise ValueError(
                f"episode {episode_index}: sender frame index is not consecutive"
                )

        send_monotonic_ns = np.asarray(
            reference["send_monotonic_ns"], dtype=np.int64
        )
        capture_monotonic_ns = np.asarray(
            control["capture_monotonic_ns"], dtype=np.int64
        )
        if (
            send_monotonic_ns.shape != (frame_count,)
            or capture_monotonic_ns.shape != (frame_count,)
        ):
            raise ValueError(
                f"episode {episode_index}: invalid send/capture timestamp telemetry"
            )
        send_to_capture_latency_s = (
            capture_monotonic_ns - send_monotonic_ns
        ).astype(np.float64) * 1e-9
        if (
            not np.isfinite(send_to_capture_latency_s).all()
            or np.any(send_to_capture_latency_s < 0.0)
        ):
            raise ValueError(
                f"episode {episode_index}: invalid send-to-capture latency"
            )
        max_latency_s = float(np.max(send_to_capture_latency_s))
        if max_latency_s > MAX_SEND_TO_CAPTURE_LATENCY_S:
            raise ValueError(
                f"episode {episode_index}: send-to-capture latency exceeds "
                f"{MAX_SEND_TO_CAPTURE_LATENCY_S:.3f}s "
                f"(max={max_latency_s:.6f}s)"
            )

        feedback = {
            "body_q": np.asarray(control["body_q"], dtype=np.float64),
            "body_dq": np.asarray(control["body_dq"], dtype=np.float64),
            "base_quat": np.asarray(control["base_quat"], dtype=np.float64),
            "base_ang_vel": np.asarray(control["base_ang_vel"], dtype=np.float64),
            "policy_action": np.asarray(
                control["policy_action"], dtype=np.float64
            ),
            "left_hand_q": np.asarray(control["left_hand_q"], dtype=np.float64),
            "right_hand_q": np.asarray(control["right_hand_q"], dtype=np.float64),
            "token_state": np.asarray(control["token_state"], dtype=np.float64),
            "control_index": control_index,
            "local_frame_index": local_frame_index,
            "send_to_capture_latency_s": send_to_capture_latency_s,
        }

    expected_shapes = {
        "body_q": (frame_count, POLICY_ACTION_DIM),
        "body_dq": (frame_count, POLICY_ACTION_DIM),
        "base_quat": (frame_count, 4),
        "base_ang_vel": (frame_count, 3),
        "policy_action": (frame_count, POLICY_ACTION_DIM),
        "left_hand_q": (frame_count, 7),
        "right_hand_q": (frame_count, 7),
        "token_state": (frame_count, 64),
    }
    for key, shape in expected_shapes.items():
        if feedback[key].shape != shape:
            raise ValueError(
                f"episode {episode_index}: {key} shape {feedback[key].shape} != {shape}"
            )
    for key, value in feedback.items():
        if not np.isfinite(value).all():
            raise ValueError(f"episode {episode_index}: non-finite values in {key}")

    quaternion_norm = np.linalg.norm(feedback["base_quat"], axis=1)
    if np.any(quaternion_norm <= 1e-8):
        raise ValueError(f"episode {episode_index}: control telemetry has zero quaternion")
    feedback["base_quat"] /= quaternion_norm[:, None]

    if expected_motion_tokens is not None:
        expected_motion_tokens = np.asarray(expected_motion_tokens, dtype=np.float64)
        if expected_motion_tokens.shape != (frame_count, 64):
            raise ValueError(
                f"episode {episode_index}: expected motion-token shape "
                f"{expected_motion_tokens.shape} != ({frame_count}, 64)"
            )
        # The C++ encoder intentionally trails the newest streamed SMPL row to
        # retain its observation window.  Keep this as an audit metric rather
        # than incorrectly demanding row-wise token equality.
        feedback["token_mae_to_sent_rows"] = np.asarray(
            [np.mean(np.abs(feedback["token_state"] - expected_motion_tokens))],
            dtype=np.float64,
        )
    return feedback


def load_velocity_gate_feedback(
    telemetry_root: Path,
    episode_index: int,
    frame_count: int,
    *,
    expected_smpl_joints: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Load the existing 30 Hz MuJoCo gate and align it to 50 Hz replay rows.

    The reference recorder timestamps every source row immediately after it is
    sent.  The simulator recorder timestamps measured joint/root state.  Linear
    interpolation onto the send timestamps therefore reconstructs the causal
    state seen during the already validated replay without rerunning SONIC.
    """

    sim_path, reference_path, _ = _telemetry_paths(
        telemetry_root,
        episode_index,
        require_control=False,
    )
    if not sim_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(
            f"episode {episode_index}: missing velocity-gate telemetry: "
            f"sim={sim_path}, reference={reference_path}"
        )

    with np.load(reference_path, allow_pickle=False) as reference:
        replay_index = np.asarray(reference["replay_frame_index"], dtype=np.int64)
        expected_index = np.arange(frame_count, dtype=np.int64)
        if replay_index.shape != expected_index.shape or not np.array_equal(
            replay_index, expected_index
        ):
            raise ValueError(
                f"episode {episode_index}: reference telemetry does not cover exactly "
                f"0..{frame_count - 1} (got {len(replay_index)} rows)"
            )
        target_time_ns = np.asarray(reference["send_monotonic_ns"], dtype=np.int64)
        recorded_smpl_joints = np.asarray(
            reference["reference_smpl_joints"], dtype=np.float64
        )

    if expected_smpl_joints is not None:
        expected_smpl_joints = np.asarray(expected_smpl_joints, dtype=np.float64)
        if (
            recorded_smpl_joints.shape != expected_smpl_joints.shape
            or not np.array_equal(recorded_smpl_joints, expected_smpl_joints)
        ):
            raise ValueError(
                f"episode {episode_index}: replay telemetry belongs to a "
                "different SMPL trajectory"
            )

    with np.load(sim_path, allow_pickle=False) as sim:
        source_time_ns = np.asarray(sim["monotonic_ns"], dtype=np.int64)
        body_q_samples = np.asarray(sim["body_joint_position"], dtype=np.float64)
        body_dq_samples = np.asarray(sim["body_joint_velocity"], dtype=np.float64)
        base_quat_samples = np.asarray(
            sim["root_quaternion_wxyz"], dtype=np.float64
        )
        base_ang_vel_samples = np.asarray(
            sim["root_angular_velocity"], dtype=np.float64
        )
        root_position_samples = np.asarray(sim["root_position"], dtype=np.float64)
        recorded_joint_names = [str(name) for name in np.asarray(sim["joint_names"])]

    expected_joint_names = list(
        get_g1_robot_model().supplemental_info.body_actuated_joints
    )
    if recorded_joint_names != expected_joint_names:
        raise ValueError(
            f"episode {episode_index}: MuJoCo joint order does not match RobotModel"
        )
    if len(source_time_ns) < 2 or np.any(np.diff(source_time_ns) <= 0):
        raise ValueError(
            f"episode {episode_index}: simulator timestamps are not strictly increasing"
        )
    if target_time_ns.shape != (frame_count,) or (
        frame_count > 1 and np.any(np.diff(target_time_ns) <= 0)
    ):
        raise ValueError(
            f"episode {episode_index}: replay send timestamps are invalid"
        )

    # The 30 Hz recorder may begin a few milliseconds after row zero or stop a
    # few milliseconds before the final 50 Hz row.  np.interp clamps those edge
    # rows; reject anything larger than 100 ms so a truncated rollout cannot be
    # mistaken for valid feedback.
    leading_gap_s = max(0.0, float(source_time_ns[0] - target_time_ns[0]) * 1e-9)
    trailing_gap_s = max(0.0, float(target_time_ns[-1] - source_time_ns[-1]) * 1e-9)
    if max(leading_gap_s, trailing_gap_s) > 0.1:
        raise ValueError(
            f"episode {episode_index}: simulator telemetry does not cover replay "
            f"timestamps (leading={leading_gap_s:.3f}s, trailing={trailing_gap_s:.3f}s)"
        )

    feedback = {
        "feedback_mode": np.asarray("velocity_gate"),
        "body_q": _interpolate_linear(
            source_time_ns, body_q_samples, target_time_ns
        ),
        "body_dq": _interpolate_linear(
            source_time_ns, body_dq_samples, target_time_ns
        ),
        "base_quat": _interpolate_quaternion_wxyz(
            source_time_ns, base_quat_samples, target_time_ns
        ),
        "base_ang_vel": _interpolate_linear(
            source_time_ns, base_ang_vel_samples, target_time_ns
        ),
        "root_position": _interpolate_linear(
            source_time_ns, root_position_samples, target_time_ns
        ),
        "edge_gap_s": np.asarray([leading_gap_s, trailing_gap_s], dtype=np.float64),
    }
    expected_shapes = {
        "body_q": (frame_count, POLICY_ACTION_DIM),
        "body_dq": (frame_count, POLICY_ACTION_DIM),
        "base_quat": (frame_count, 4),
        "base_ang_vel": (frame_count, 3),
        "root_position": (frame_count, 3),
    }
    for key, shape in expected_shapes.items():
        if feedback[key].shape != shape:
            raise ValueError(
                f"episode {episode_index}: {key} shape {feedback[key].shape} != {shape}"
            )
        if not np.isfinite(feedback[key]).all():
            raise ValueError(f"episode {episode_index}: non-finite values in {key}")
    minimum_root_height = float(np.min(feedback["root_position"][:, 2]))
    if minimum_root_height < MIN_VALID_ROOT_HEIGHT_M:
        raise ValueError(
            f"episode {episode_index}: velocity-gate replay contains a fallen robot "
            f"(minimum root height {minimum_root_height:.3f}m < "
            f"{MIN_VALID_ROOT_HEIGHT_M:.3f}m)"
        )
    feedback["minimum_root_height_m"] = np.asarray(
        [minimum_root_height], dtype=np.float64
    )
    return feedback


def _physical_action_to_reference_offset(physical: np.ndarray) -> np.ndarray:
    physical = np.asarray(physical, dtype=np.float64)
    if physical.ndim != 2 or physical.shape[1] != POLICY_ACTION_DIM:
        raise ValueError(
            f"physical policy action must be [N, {POLICY_ACTION_DIM}], got {physical.shape}"
        )
    return (
        physical[:, MUJOCO_TO_ISAACLAB]
        - DEFAULT_ANGLES_MUJOCO[MUJOCO_TO_ISAACLAB]
    ).astype(np.float32)


def _compute_eef_state(
    body_q: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
) -> np.ndarray:
    robot_model = get_g1_robot_model()
    output = np.empty((len(body_q), 14), dtype=np.float64)
    for frame_index in range(len(body_q)):
        whole_q = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_q[frame_index],
            left_hand_actuated_joint_values=left_hand_q[frame_index],
            right_hand_actuated_joint_values=right_hand_q[frame_index],
        )
        robot_model.cache_forward_kinematics(whole_q)
        parts = []
        for side in ("left", "right"):
            placement = robot_model.frame_placement(
                robot_model.supplemental_info.hand_frame_names[side]
            )
            parts.append(
                np.concatenate(
                    [
                        np.asarray(placement.translation[:3], dtype=np.float64),
                        R.from_matrix(placement.rotation).as_quat(scalar_first=True),
                    ]
                )
            )
        output[frame_index] = np.concatenate(parts)
    return output


def _causal_hand_state_from_source(
    table: pa.Table,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate unrecorded BrainCo feedback with the prior source command."""

    source_state = _stack_fixed_list(table, "observation.state", 43).astype(
        np.float64, copy=False
    )
    left_command = source_state[:, LEFT_HAND_STATE_INDICES]
    right_command = source_state[:, RIGHT_HAND_STATE_INDICES]
    left_state = left_command.copy()
    right_state = right_command.copy()
    if len(table) > 1:
        left_state[1:] = left_command[:-1]
        right_state[1:] = right_command[:-1]
    return left_state, right_state


def _next_measured_body_state(body_q: np.ndarray) -> np.ndarray:
    """Use q[t+1] as a causal auxiliary target and hold the final row."""

    body_q = np.asarray(body_q, dtype=np.float64)
    result = body_q.copy()
    if len(result) > 1:
        result[:-1] = body_q[1:]
    return result


def _learned_action_snapshot(table: pa.Table) -> dict[str, np.ndarray]:
    return {
        key: _stack_fixed_list(table, key, size).copy()
        for key, size in LEARNED_ACTION_COLUMNS
    }


def _encode_observation_state(
    physical_state: np.ndarray,
    state_schema: str,
) -> np.ndarray:
    """Encode measured 43D qpos without changing its causal timing."""

    physical_state = np.asarray(physical_state, dtype=np.float64)
    if physical_state.ndim != 2 or physical_state.shape[1] != 43:
        raise ValueError(f"physical observation state must be [N, 43], got {physical_state.shape}")
    if state_schema == "isaaclab_offsets":
        return physical_qpos_to_isaaclab_offset_state(physical_state)
    if state_schema == "physical_qpos":
        return physical_state.copy()
    raise ValueError(f"unsupported state schema: {state_schema}")


def rebuild_episode(
    table: pa.Table,
    feedback: dict[str, np.ndarray],
    *,
    state_schema: str = "isaaclab_offsets",
) -> tuple[pa.Table, dict[str, np.ndarray]]:
    frame_count = len(table)
    body_q = np.asarray(feedback["body_q"], dtype=np.float64)
    body_dq = np.asarray(feedback["body_dq"], dtype=np.float64)
    base_quat = np.asarray(feedback["base_quat"], dtype=np.float64)
    feedback_mode = str(np.asarray(feedback.get("feedback_mode", "captured_cpp")))
    if feedback_mode == "velocity_gate":
        action_body_physical = _next_measured_body_state(body_q)
        left_hand_state, right_hand_state = _causal_hand_state_from_source(table)
    elif feedback_mode == "captured_cpp":
        action_body_physical = np.asarray(
            feedback["policy_action"], dtype=np.float64
        )
        left_hand_state = np.asarray(feedback["left_hand_q"], dtype=np.float64)
        right_hand_state = np.asarray(feedback["right_hand_q"], dtype=np.float64)
    else:
        raise ValueError(f"unsupported feedback mode: {feedback_mode}")

    learned_action_before = _learned_action_snapshot(table)
    projected_gravity = np.stack(
        [compute_projected_gravity(quat) for quat in base_quat], axis=0
    ).astype(np.float64)
    reference_q = _physical_action_to_reference_offset(action_body_physical)
    reference_dq = np.zeros_like(reference_q)
    if frame_count > 1:
        reference_dq[1:] = np.diff(reference_q, axis=0) * 50.0
        reference_dq[0] = reference_dq[1]

    old_action = _stack_fixed_list(table, "action.wbc", 43)

    physical_state = _stack_fixed_list(table, "observation.state", 43).copy()
    physical_state[:, BODY_STATE_INDICES] = body_q
    physical_state[:, LEFT_HAND_STATE_INDICES] = left_hand_state
    physical_state[:, RIGHT_HAND_STATE_INDICES] = right_hand_state
    state = _encode_observation_state(physical_state, state_schema)

    action = old_action.copy()
    action[:, BODY_STATE_INDICES] = action_body_physical
    eef_state = _compute_eef_state(
        body_q,
        left_hand_state,
        right_hand_state,
    )
    init_base_quat = np.repeat(base_quat[:1], frame_count, axis=0)

    replacements = {
        "observation.state": state,
        "observation.eef_state": eef_state,
        "action.wbc": action,
        "observation.root_orientation": base_quat,
        "observation.projected_gravity": projected_gravity,
        "observation.init_base_quat": init_base_quat,
        "teleop.g1_joint_pos": reference_q,
        "teleop.g1_joint_vel": reference_dq,
    }
    if feedback_mode == "captured_cpp":
        replacements["action.motion_token"] = np.asarray(
            feedback["token_state"], dtype=np.float64
        )
    for key, values in replacements.items():
        table = _replace_fixed_list(table, key, values)
    learned_action_after = _learned_action_snapshot(table)
    for key, _ in HAND_ACTION_COLUMNS:
        if not np.array_equal(learned_action_before[key], learned_action_after[key]):
            raise AssertionError(f"BrainCo action field changed unexpectedly: {key}")
    if feedback_mode == "captured_cpp":
        if not np.array_equal(
            learned_action_after["action.motion_token"],
            replacements["action.motion_token"],
        ):
            raise AssertionError("captured C++ motion token was not written exactly")
    elif not np.array_equal(
        learned_action_before["action.motion_token"],
        learned_action_after["action.motion_token"],
    ):
        raise AssertionError("legacy velocity-gate mode changed the source motion token")
    return table, replacements


def _array_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(len(values))],
    }


def _copy_or_link(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _parse_episode_selection(
    expression: str,
    available_indices: Iterable[int],
) -> list[int]:
    """Select source episode indices without silently renumbering them.

    Stage-B rollout contracts are bound to the original Stage-A parquet path
    and hash.  A feedback-only recovery may therefore intentionally skip a
    failed source episode and must keep the surviving source indices intact
    until the copied output is compacted.  Support the same half-open slice
    notation as the dataset replay runner (for example ``0:80,81:103``), as
    well as the older inclusive ``0-9`` spelling used by this shell wrapper.
    """

    available = sorted({int(index) for index in available_indices})
    if not available:
        raise ValueError("source dataset has no episodes")
    if available[0] < 0:
        raise ValueError(f"source dataset has negative episode index: {available[0]}")

    text = expression.strip().lower()
    if text in {"", "all", "*"}:
        return available

    upper_bound = available[-1] + 1
    selected: set[int] = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            fields = token.split(":")
            if len(fields) not in {2, 3}:
                raise ValueError(f"invalid episode range: {token}")
            start = int(fields[0]) if fields[0] else 0
            stop = int(fields[1]) if fields[1] else upper_bound
            step = int(fields[2]) if len(fields) == 3 and fields[2] else 1
            if step <= 0:
                raise ValueError(f"episode range step must be positive: {token}")
            selected.update(range(start, stop, step))
        elif "-" in token:
            fields = token.split("-")
            if len(fields) != 2 or not all(field.strip() for field in fields):
                raise ValueError(f"invalid episode range: {token}")
            start, stop_inclusive = (int(field) for field in fields)
            if stop_inclusive < start:
                raise ValueError(f"episode range must be ascending: {token}")
            selected.update(range(start, stop_inclusive + 1))
        else:
            selected.add(int(token))

    if not selected:
        raise ValueError("episode selection is empty")
    unavailable = sorted(selected - set(available))
    if unavailable:
        raise ValueError(
            "episode selection is not present in source dataset: "
            f"{unavailable}"
        )
    return sorted(selected)


def _episode_indices(dataset_root: Path, selection: str = "all") -> list[int]:
    records = [
        json.loads(line)
        for line in (dataset_root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(record["episode_index"]) for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate episode_index in {dataset_root / 'meta/episodes.jsonl'}")
    return _parse_episode_selection(selection, indices)


def process_dataset(args: argparse.Namespace) -> None:
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    telemetry_root = args.telemetry_root.expanduser().resolve()
    validate_output_root(output_root, [source_root, telemetry_root])
    if not (source_root / "meta/info.json").is_file():
        raise FileNotFoundError(f"missing source metadata: {source_root}")
    all_episode_indices = _episode_indices(source_root)
    episode_indices = _episode_indices(source_root, args.episodes)
    summary_path, rollout_summary = validate_stage_b_rollout(
        telemetry_root,
        episode_indices,
        require_real_wrist_contract=args.feedback_mode == "captured_cpp",
    )

    replay_rate = float(rollout_summary.get("replay_rate", 0.0))
    if not np.isfinite(replay_rate) or replay_rate <= 0:
        raise ValueError(f"rollout summary has invalid replay_rate: {replay_rate}")
    # Fail before creating OUTPUT_ROOT if even one episode belongs to a
    # different Stage-A parquet or predates the per-episode rollout stamp.
    for episode_index in episode_indices:
        parquet_path = source_root / (
            f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
        )
        source_rows = pq.ParquetFile(parquet_path).metadata.num_rows
        require_control = args.feedback_mode == "captured_cpp"
        telemetry_dir, telemetry_stem = _resolve_telemetry_stem(
            telemetry_root,
            episode_index,
            require_control=require_control,
        )
        contract_path = telemetry_dir / f"{telemetry_stem}.rollout.json"
        try:
            rollout_contract = json.loads(
                contract_path.read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f"episode {episode_index}: missing Stage-B rollout contract "
                f"{contract_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"episode {episode_index}: invalid Stage-B rollout contract "
                f"{contract_path}: {exc}"
            ) from exc
        validate_episode_rollout_contract(
            rollout_contract,
            episode_index=episode_index,
            source_sha256=sha256_file(parquet_path),
            source_rows=source_rows,
            replay_rows=source_rows,
            protocol="pose",
            replay_rate=replay_rate,
            video_name=f"{telemetry_stem}.mp4",
            pose_wrist_contract=POSE_WRIST_CONTRACT,
        )

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root, copy_function=_copy_or_link)

    # The LeRobot state schema keeps the fixed 7-DOF ThreeFinger layout while
    # the teleop action modality carries BrainCo's native six values.
    os.environ["HAND_TYPE"] = "brainco"
    stats_path = output_root / "meta/episodes_stats.jsonl"
    existing_stats = {
        int(record["episode_index"]): record
        for record in (
            json.loads(line)
            for line in stats_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    missing_stats = sorted(set(all_episode_indices) - set(existing_stats))
    if missing_stats:
        raise ValueError(
            "source episodes_stats.jsonl is missing records for: "
            f"{missing_stats}"
        )

    for ordinal, episode_index in enumerate(episode_indices, start=1):
        parquet_path = output_root / (
            f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(parquet_path)
        expected_smpl_joints = _stack_fixed_list(
            table, "teleop.smpl_joints", 72, np.float64
        )
        if args.feedback_mode == "velocity_gate":
            feedback = load_velocity_gate_feedback(
                telemetry_root,
                episode_index,
                len(table),
                expected_smpl_joints=expected_smpl_joints,
            )
        else:
            feedback = load_feedback(
                telemetry_root,
                episode_index,
                len(table),
                expected_smpl_joints=expected_smpl_joints,
                expected_motion_tokens=_stack_fixed_list(
                    table, "action.motion_token", 64, np.float64
                ),
            )
            feedback["feedback_mode"] = np.asarray("captured_cpp")
        table, replacements = rebuild_episode(
            table,
            feedback,
            state_schema=args.state_schema,
        )
        temporary = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="snappy")
        os.replace(temporary, parquet_path)
        record = existing_stats[episode_index]
        for key, values in replacements.items():
            record["stats"][key] = _array_stats(values)
        print(
            f"[{ordinal}/{len(episode_indices)}] episode {episode_index:06d}: "
            f"{len(table)} causal MuJoCo frames "
            f"({args.feedback_mode}, {args.state_schema})",
            flush=True,
        )

    _write_jsonl_atomic(
        stats_path,
        [existing_stats[index] for index in all_episode_indices],
    )
    info_path = output_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    script_config = info.setdefault("script_config", {})
    common_config = {
        "pipeline": "raw_pico_to_smpl_to_sonic_encoder_to_mujoco_feedback_to_gr00t",
        "rollout_telemetry_root": str(telemetry_root),
        "rollout_manifest": str(summary_path),
        "rollout_source_dataset": rollout_summary.get("dataset"),
        "rollout_startup_strategy": "sonic_stage_b_conversion",
        "rollout_startup": {
            key: rollout_summary.get("config", {}).get(key)
            for key in STAGE_B_ROLLOUT_CONFIG
        },
        "state_schema": args.state_schema,
        "state_source_schema": "physical_qpos_mujoco_absolute",
        "state_conversion": (
            "MuJoCo absolute/grouped order -> IsaacLab order/default-relative "
            "body slots; hand slots unchanged"
            if args.state_schema == "isaaclab_offsets"
            else "none"
        ),
        "action_wbc_schema": "physical_q_target_mujoco_absolute",
        "learned_action_schema": (
            "76D = action.motion_token[64] + teleop.left_hand_joints[6] + "
            "teleop.right_hand_joints[6]"
        ),
        "learned_action_policy": (
            "64D motion token replaced from same captured C++ control tick; "
            "6D+6D BrainCo actions preserved from source"
            if args.feedback_mode == "captured_cpp"
            else "legacy diagnostic: preserved exactly from source dataset"
        ),
        "motion_token_source": (
            "telemetry/episode_XXXXXX.control.npz:token_state"
            if args.feedback_mode == "captured_cpp"
            else "source staging action.motion_token"
        ),
        "pose_wrist_contract": rollout_summary.get("pose_wrist_contract"),
        "sim_episode_lifecycle_contract": rollout_summary.get(
            "sim_episode_lifecycle_contract"
        ),
        "feedback_source_episode_selection": args.episodes,
        "feedback_source_episode_indices": episode_indices,
        "feedback_unmodified_source_episode_indices": [
            index for index in all_episode_indices if index not in set(episode_indices)
        ],
    }
    overrides_manifest = telemetry_root / "lightweight_overrides.json"
    if overrides_manifest.is_file():
        common_config["rollout_overrides_manifest"] = str(
            overrides_manifest.resolve()
        )
    if args.feedback_mode == "velocity_gate":
        mode_config = {
            "decoder_history_mode": "not_redecoded_existing_mujoco_feedback",
            "decoder_history_note": (
                "lightweight route: observation.state is interpolated from the "
                "validated Stage-B SONIC/MuJoCo rollout; no new C++ replay "
                "or offline decoder warmup is performed"
            ),
            "state_alignment": "linear interpolation at replay send_monotonic_ns",
            "hand_state_timing": (
                "initial command at t=0, previous command for t>0; gate telemetry "
                "does not record BrainCo feedback"
            ),
            "body_action_timing": (
                "auxiliary action.wbc and teleop.g1_joint_pos use measured q[t+1]; "
                "final row holds q[t]"
            ),
            "control_tick_gate": "not required; existing 30Hz sim state aligned to 50Hz rows",
        }
    else:
        mode_config = {
            "decoder_history_mode": "captured_cpp_mujoco_feedback",
            "decoder_history_note": (
                "state[t] and policy action[t] are captured from the same SONIC "
                "C++ control tick after the standard planner-default hold and "
                "5-second SMPL ramp; no offline decoder warmup is synthesized"
            ),
            "hand_state_timing": "captured pre-action BrainCo feedback",
            "body_action_timing": "same C++ control tick as observation.state",
            "control_tick_gate": "exactly one consecutive C++ tick per replay row",
            "motion_token_alignment": (
                "same captured C++ control tick; no manual source-row shift"
            ),
        }
    script_config.update({**common_config, **mode_config})
    script_config.pop("decoder_warmup_steps", None)
    _write_json_atomic(info_path, info)
    for stale_name in ("stats.json", "relative_stats.json"):
        stale_path = output_root / "meta" / stale_name
        if stale_path.exists():
            stale_path.unlink()
    print(f"Done: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply causal SONIC/MuJoCo feedback to a converted LeRobot dataset."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        default="all",
        help=(
            "Source episode selector. Supports all, 0,2,4, 0:80,81:103, "
            "or 0-9. A subset keeps untouched source rows until a later "
            "explicit filter/compaction step."
        ),
    )
    parser.add_argument(
        "--feedback-mode",
        choices=FEEDBACK_MODES,
        default="captured_cpp",
        help=(
            "captured_cpp requires real-wrist per-cycle .control.npz telemetry "
            "and writes the captured token (default); velocity_gate is a legacy "
            "state-only diagnostic that preserves the source token"
        ),
    )
    parser.add_argument(
        "--state-schema",
        choices=STATE_SCHEMAS,
        default="isaaclab_offsets",
        help=(
            "isaaclab_offsets uses IsaacLab-order default-relative body angles "
            "(default); physical_qpos uses absolute model joint angles"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    process_dataset(parse_args())
