"""Pure trajectory helpers for metric SMPL-to-G1 endpoint retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from egoalign.geometry.torch_transforms import angle_axis_to_rotation_matrix
from scripts.human_data.endpoint_geometry import (
    G1_LEFT_FOOT_ENDPOINT_OFFSET_M,
    G1_LEFT_PALM_ENDPOINT_OFFSET_M,
    G1_RIGHT_FOOT_ENDPOINT_OFFSET_M,
    G1_RIGHT_PALM_ENDPOINT_OFFSET_M,
)
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
    filter_pico_wrist_motion_artifacts,
    repair_pico_false_grasp_wrist_artifacts,
)


OUTPUT_FROM_SMPL_INTERNAL = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
ENDPOINT_RETARGET_ALGORITHM = (
    "endpoint_retarget_egoalign_arm_first_gated_spine_locked_clavicles"
)
ENDPOINT_RETARGET_CANDIDATE_CONTRACT_VERSION = 4
ENDPOINT_DEFINITION = "four_non_thumb_mcp_root_mean"
ENDPOINT_TARGET_CONTRACT = (
    "direct per-frame height-scaled canonical big-SMPL four-MCP palm "
    "centers after bounded Pico frame-switch repair and zero-phase elbow-local "
    "wrist denoising, horizontal pelvis centering, and feet grounding; "
    "EgoAlign Staged IK arm-first IK with original per-frame clavicles locked; input spine preserved unless arm-stage residual exceeds release threshold; bounded spine assistance"
)
WRIST_LOCAL_ROTATION_POLICY = (
    "locked_exactly_to_denoised_wrist_baseline"
)
VR3PT_ORIENTATION_POLICY = (
    "preserved_exactly_from_denoised_wrist_baseline"
)

# Bilateral palm alignment includes explicit Pico wrist denoising.
ENDPOINT_PARTS_HANDS = "hands"
ENDPOINT_PART_CHOICES = (
    ENDPOINT_PARTS_HANDS,
)
HAND_ENDPOINT_SIDE_BOTH = "both"
HAND_ENDPOINT_SIDE_CHOICES = (HAND_ENDPOINT_SIDE_BOTH,)

DEFAULT_CANONICAL_HEAD_HEIGHT_M = 1.5549421608448029
LEFT_ENDPOINT_OUTPUT_INDEX = 22
RIGHT_ENDPOINT_OUTPUT_INDEX = 23
# Staged IK first preserves input spine, then permits bounded assistance on high-error frames.
# Wrists and all joints outside the spine/arm chains remain unchanged.
ARM_OPTIMIZED_JOINT_INDICES = (13, 14, 16, 17, 18, 19)
DEFAULT_OPTIMIZED_JOINT_INDICES = (3, 6, 9) + ARM_OPTIMIZED_JOINT_INDICES
# Used only by the production auditor to report preserved lower-body motion.
FOOT_CHAIN_JOINT_INDICES = (1, 2, 4, 5, 7, 8)
# The previous endpoint used thumb tips 39/54, turning anatomical thumb
# abduction into apparent wrist extension. The mean of the four long-finger
# MCP roots is a stable differentiable palm-center proxy.
LEFT_PALM_MCP_INDICES = (25, 28, 31, 34)
RIGHT_PALM_MCP_INDICES = (40, 43, 46, 49)
LEFT_FOOT_INDEX = 10
RIGHT_FOOT_INDEX = 11
LEFT_WRIST_INDEX = 20
RIGHT_WRIST_INDEX = 21
ROOT_INDEX = 0
HEAD_INDEX = 15

@dataclass(frozen=True)
class EndpointRetargetContract:
    endpoint_parts: str
    algorithm: str
    candidate_contract_version: int
    endpoint_definition: str
    endpoint_target_contract: str
    optimized_joint_indices: tuple[int, ...]
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH


def endpoint_retarget_contract(
    endpoint_parts: str,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
    *,
    experimental_lock_spine: bool = False,
) -> EndpointRetargetContract:
    """Validate the public bilateral-palm contract, including archived metadata."""
    parts = str(endpoint_parts).strip().lower()
    hand_side = str(hand_endpoint_side).strip().lower()
    if parts != "hands" or hand_side != "both" or experimental_lock_spine:
        raise ValueError(
            "EgoAlign supports only bilateral hands with arm-first gated spine assistance; "
            "legacy FEET/ARM_PRIOR and diagnostic flags are not supported"
        )
    return EndpointRetargetContract(
        endpoint_parts=parts,
        algorithm=ENDPOINT_RETARGET_ALGORITHM,
        candidate_contract_version=ENDPOINT_RETARGET_CANDIDATE_CONTRACT_VERSION,
        endpoint_definition=ENDPOINT_DEFINITION,
        endpoint_target_contract=ENDPOINT_TARGET_CONTRACT,
        optimized_joint_indices=DEFAULT_OPTIMIZED_JOINT_INDICES,
        hand_endpoint_side=hand_side,
    )


def endpoint_parts_from_archive(archive: Any) -> str:
    """Return the explicit endpoint selection or infer hands for diagnostics."""

    if "endpoint_parts" not in _archive_keys(archive):
        return ENDPOINT_PARTS_HANDS
    value = _archive_scalar(archive, "endpoint_parts")
    parts = str(value)
    endpoint_retarget_contract(parts)
    return parts


def hand_endpoint_side_from_archive(archive: Any) -> str:
    """Return the explicit hand-side contract; legacy palm archives mean both."""

    if "hand_endpoint_side" not in _archive_keys(archive):
        return HAND_ENDPOINT_SIDE_BOTH
    side = str(_archive_scalar(archive, "hand_endpoint_side"))
    if side not in HAND_ENDPOINT_SIDE_CHOICES:
        raise ValueError(f"unsupported hand_endpoint_side in archive: {side!r}")
    return side


@dataclass(frozen=True)
class OptimizationWeights:
    position: float = 500.0
    pose_prior: float = 0.6
    velocity: float = 18.0
    acceleration: float = 8.0
    first_frame: float = 0.0


@dataclass(frozen=True)
class OptimizationResult:
    pose: np.ndarray
    virtual_joints: np.ndarray
    losses: list[dict[str, float]]
    optimized_joint_indices: tuple[int, ...]
    joint_delta_limits_rad: np.ndarray
    staged_proof: dict[str, np.ndarray] | None = None


def propagate_endpoint_delta_to_vr3pt(
    baseline_vr3pt_position: np.ndarray,
    baseline_canonical_joints: np.ndarray,
    candidate_canonical_joints: np.ndarray,
    *,
    propagate_head: bool = False,
) -> np.ndarray:
    """Keep SONIC's VR3 position input consistent with the retargeted SMPL.

    POSE packets carry both SMPL features and explicit VR3 wrist/head targets.
    The original VR3 values include tracker-specific offsets, so preserve those
    values and add only the retargeted canonical displacement.  Arm-only
    callers leave the head unchanged; whole-body morphology scaling must also
    propagate the canonical head displacement so SONIC does not receive two
    inconsistent body scales.
    """

    vr3 = np.asarray(baseline_vr3pt_position, dtype=np.float64)
    baseline = np.asarray(baseline_canonical_joints, dtype=np.float64)
    candidate = np.asarray(candidate_canonical_joints, dtype=np.float64)
    if vr3.ndim != 2 or vr3.shape[1] != 9:
        raise ValueError(f"baseline_vr3pt_position must have shape (N, 9), got {vr3.shape}")
    if baseline.shape != candidate.shape or baseline.ndim != 3 or baseline.shape[1:] != (24, 3):
        raise ValueError(
            "baseline/candidate canonical joints must have matching shape (N, 24, 3)"
        )
    if len(vr3) != len(baseline):
        raise ValueError("VR3 and canonical-joint frame counts differ")
    if not (np.isfinite(vr3).all() and np.isfinite(baseline).all() and np.isfinite(candidate).all()):
        raise ValueError("VR3/canonical endpoint propagation inputs must be finite")

    result = vr3.copy()
    result[:, 0:3] += (
        candidate[:, LEFT_ENDPOINT_OUTPUT_INDEX]
        - baseline[:, LEFT_ENDPOINT_OUTPUT_INDEX]
    )
    result[:, 3:6] += (
        candidate[:, RIGHT_ENDPOINT_OUTPUT_INDEX]
        - baseline[:, RIGHT_ENDPOINT_OUTPUT_INDEX]
    )
    if propagate_head:
        result[:, 6:9] += candidate[:, HEAD_INDEX] - baseline[:, HEAD_INDEX]
    return result.astype(np.float32)


def canonical_head_height_m(rest_joints: np.ndarray) -> float:
    """Return the canonical foot-joint to head-joint height in meters."""

    rest = np.asarray(rest_joints, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] != 3 or len(rest) <= HEAD_INDEX:
        raise ValueError("rest_joints must contain canonical SMPL joints through head index 15")
    if not np.isfinite(rest).all():
        raise ValueError("rest_joints must be finite")
    output = rest @ OUTPUT_FROM_SMPL_INTERNAL.T
    ground_z = min(float(output[LEFT_FOOT_INDEX, 2]), float(output[RIGHT_FOOT_INDEX, 2]))
    height = float(output[HEAD_INDEX, 2] - ground_z)
    if height <= 0.0 or not np.isfinite(height):
        raise ValueError(f"canonical SMPL head height must be positive, got {height}")
    return height


def scale_canonical_rest_to_head_height(
    rest_joints: np.ndarray,
    target_head_height_m: float = DEFAULT_CANONICAL_HEAD_HEIGHT_M,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Uniformly scale canonical SMPL bones to an exact head-joint height.

    Scaling is root-relative, so the canonical pelvis anchor is unchanged.
    Translation of each recorded trajectory remains owned by
    :func:`canonical_output_joints`.
    """

    rest = np.asarray(rest_joints, dtype=np.float64)
    target = float(target_head_height_m)
    if target <= 0.0 or not np.isfinite(target):
        raise ValueError(f"target_head_height_m must be positive, got {target}")
    source = canonical_head_height_m(rest)
    scale = target / source
    root = rest[ROOT_INDEX].copy()
    scaled = root + (rest - root) * scale
    achieved = canonical_head_height_m(scaled)
    return scaled, {
        "method": "uniform root-relative canonical SMPL bone scaling",
        "source_head_height_m": source,
        "target_head_height_m": target,
        "achieved_head_height_m": achieved,
        "uniform_scale": scale,
    }


def relative_offsets(rest_joints: np.ndarray, parents: list[int]) -> np.ndarray:
    rest = np.asarray(rest_joints, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] != 3 or len(parents) != len(rest):
        raise ValueError("rest_joints must be (J, 3) and match parents")
    offsets = rest.copy()
    for index, parent in enumerate(parents):
        if parent >= 0:
            offsets[index] -= rest[parent]
    return offsets


def absolute_rest_joints(offsets: np.ndarray, parents: list[int]) -> np.ndarray:
    rel = np.asarray(offsets, dtype=np.float64)
    if rel.ndim != 2 or rel.shape[1] != 3 or len(parents) != len(rel):
        raise ValueError("offsets must be (J, 3) and match parents")
    rest = np.zeros_like(rel)
    for index, parent in enumerate(parents):
        rest[index] = rel[index] if parent < 0 else rest[parent] + rel[index]
    return rest


def _set_chain_length(offsets: np.ndarray, chain: tuple[int, ...], length: float) -> None:
    current = float(sum(np.linalg.norm(offsets[index]) for index in chain))
    if current <= 1e-9 or not np.isfinite(current):
        raise ValueError(f"cannot scale degenerate morphology chain {chain}")
    scale = float(length) / current
    for index in chain:
        offsets[index] *= scale


def _set_palm_center_length(
    offsets: np.ndarray,
    mcp_indices: tuple[int, ...],
    length: float,
) -> None:
    """Scale direct wrist-to-MCP offsets to one requested palm-center length."""

    indices = np.asarray(mcp_indices, dtype=np.int64)
    center_offset = np.mean(offsets[indices], axis=0)
    current = float(np.linalg.norm(center_offset))
    if current <= 1e-9 or not np.isfinite(current):
        raise ValueError(f"cannot scale degenerate palm center {mcp_indices}")
    offsets[indices] *= float(length) / current


def build_virtual_g1_sized_rest_joints(
    canonical_rest_joints: np.ndarray,
    parents: list[int],
    morphology: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the full SMPL skeleton to measured G1 morphology.

    The virtual skeleton is an optimization-only body.  Start by scaling every
    non-root bone with the shoulder/torso ratio, then replace both complete leg
    chains and both arm/palm chains with lengths measured from the bound G1
    MuJoCo model.  This prevents the misleading half-human/half-G1 morphology
    produced by the original upper-body-only implementation.
    """

    canonical = np.asarray(canonical_rest_joints, dtype=np.float64)
    offsets = relative_offsets(canonical, parents)
    sides = morphology["sides"]

    canonical_root = canonical[0] @ OUTPUT_FROM_SMPL_INTERNAL.T
    shoulder_ratios: list[float] = []
    for side, joint_index in (("left", 16), ("right", 17)):
        canonical_shoulder = canonical[joint_index] @ OUTPUT_FROM_SMPL_INTERNAL.T
        canonical_radius = float(np.linalg.norm(canonical_shoulder - canonical_root))
        g1_shoulder = np.asarray(
            sides[side]["shoulder_pitch_anchor_from_root_m"], dtype=np.float64
        )
        shoulder_ratios.append(float(np.linalg.norm(g1_shoulder)) / canonical_radius)
    torso_scale = float(np.mean(shoulder_ratios))
    offsets[1:] *= torso_scale

    required_leg_fields = (
        "hip_pitch_anchor_distance_m",
        "upper_leg_chain_length_m",
        "lower_leg_chain_length_m",
        "foot_offset_length_m",
    )
    for side in ("left", "right"):
        missing = [name for name in required_leg_fields if name not in sides[side]]
        if missing:
            raise ValueError(
                f"G1 morphology for {side} is upper-body-only; missing {missing}. "
                "Regenerate it with calibrate_g1_endpoint_morphology.py format v2."
            )

    for side, hip, knee, ankle, foot in (
        ("left", 1, 4, 7, 10),
        ("right", 2, 5, 8, 11),
    ):
        _set_chain_length(
            offsets,
            (hip,),
            float(sides[side]["hip_pitch_anchor_distance_m"]),
        )
        _set_chain_length(
            offsets,
            (knee,),
            float(sides[side]["upper_leg_chain_length_m"]),
        )
        _set_chain_length(
            offsets,
            (ankle,),
            float(sides[side]["lower_leg_chain_length_m"]),
        )
        _set_chain_length(
            offsets,
            (foot,),
            float(sides[side]["foot_offset_length_m"]),
        )

    _set_chain_length(offsets, (18,), float(sides["left"]["upper_arm_chain_length_m"]))
    _set_chain_length(offsets, (19,), float(sides["right"]["upper_arm_chain_length_m"]))
    _set_chain_length(offsets, (20,), float(sides["left"]["forearm_chain_length_m"]))
    _set_chain_length(offsets, (21,), float(sides["right"]["forearm_chain_length_m"]))
    _set_palm_center_length(
        offsets,
        LEFT_PALM_MCP_INDICES,
        float(sides["left"]["palm_offset_length_m"]),
    )
    _set_palm_center_length(
        offsets,
        RIGHT_PALM_MCP_INDICES,
        float(sides["right"]["palm_offset_length_m"]),
    )

    virtual = absolute_rest_joints(offsets, parents)
    details = {
        "fit_method": (
            "SMPL topology; whole-body base scale plus side-specific "
            "hip/upper-leg/lower-leg/foot and arm/palm chain lengths"
        ),
        "whole_body_scaled": True,
        "torso_scale": torso_scale,
        "shoulder_scale_candidates": shoulder_ratios,
        "left_hip_offset_length_m": float(np.linalg.norm(offsets[1])),
        "right_hip_offset_length_m": float(np.linalg.norm(offsets[2])),
        "left_upper_leg_length_m": float(np.linalg.norm(offsets[4])),
        "right_upper_leg_length_m": float(np.linalg.norm(offsets[5])),
        "left_lower_leg_length_m": float(np.linalg.norm(offsets[7])),
        "right_lower_leg_length_m": float(np.linalg.norm(offsets[8])),
        "left_foot_length_m": float(np.linalg.norm(offsets[10])),
        "right_foot_length_m": float(np.linalg.norm(offsets[11])),
        "left_upper_arm_length_m": float(np.linalg.norm(offsets[18])),
        "right_upper_arm_length_m": float(np.linalg.norm(offsets[19])),
        "left_forearm_length_m": float(np.linalg.norm(offsets[20])),
        "right_forearm_length_m": float(np.linalg.norm(offsets[21])),
        "left_palm_center_length_m": float(
            np.linalg.norm(np.mean(offsets[np.asarray(LEFT_PALM_MCP_INDICES)], axis=0))
        ),
        "right_palm_center_length_m": float(
            np.linalg.norm(np.mean(offsets[np.asarray(RIGHT_PALM_MCP_INDICES)], axis=0))
        ),
    }
    return virtual, details


def forward_joints_torch(
    body_pose: torch.Tensor,
    rest_joints: torch.Tensor,
    parents: list[int],
) -> torch.Tensor:
    """Differentiable SMPL FK with identity root orientation."""

    if body_pose.ndim != 2 or body_pose.shape[1] != 63:
        raise ValueError(f"body_pose must have shape (N, 63), got {tuple(body_pose.shape)}")
    if rest_joints.ndim != 2 or rest_joints.shape[1] != 3:
        raise ValueError("rest_joints must have shape (J, 3)")
    frames = body_pose.shape[0]
    joint_count = rest_joints.shape[0]
    remaining = joint_count * 3 - 3 - 63
    if remaining < 0:
        raise ValueError("rest skeleton is too small for 21 SMPL body joints")
    full_pose = torch.cat(
        [
            torch.zeros((frames, 3), dtype=body_pose.dtype, device=body_pose.device),
            body_pose,
            torch.zeros((frames, remaining), dtype=body_pose.dtype, device=body_pose.device),
        ],
        dim=-1,
    )
    rotations = angle_axis_to_rotation_matrix(full_pose.reshape(frames, joint_count, 3))
    rest = rest_joints.to(dtype=body_pose.dtype, device=body_pose.device)
    rel = rest.clone()
    parent_tensor = torch.as_tensor(parents[1:], dtype=torch.long, device=body_pose.device)
    rel[1:] = rest[1:] - rest[parent_tensor]

    transforms: list[torch.Tensor] = []
    for index, parent in enumerate(parents):
        transform = torch.zeros(
            (frames, 4, 4), dtype=body_pose.dtype, device=body_pose.device
        )
        transform[:, :3, :3] = rotations[:, index]
        transform[:, :3, 3] = rel[index]
        transform[:, 3, 3] = 1.0
        transforms.append(transform if parent < 0 else transforms[parent] @ transform)
    return torch.stack(transforms, dim=1)[:, :, :3, 3]


def output_frame_joints_torch(internal_joints: torch.Tensor) -> torch.Tensor:
    transform = torch.as_tensor(
        OUTPUT_FROM_SMPL_INTERNAL,
        dtype=internal_joints.dtype,
        device=internal_joints.device,
    )
    return internal_joints @ transform.T


def palm_centers_torch(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left/right four-MCP palm centers from a 55-joint trajectory."""

    required = max(*LEFT_PALM_MCP_INDICES, *RIGHT_PALM_MCP_INDICES)
    if joints.ndim != 3 or joints.shape[2] != 3 or joints.shape[1] <= required:
        raise ValueError(
            "joints must have shape (N, J, 3) and contain all palm MCP roots"
        )
    left = torch.mean(joints[:, list(LEFT_PALM_MCP_INDICES)], dim=1)
    right = torch.mean(joints[:, list(RIGHT_PALM_MCP_INDICES)], dim=1)
    return left, right


def smpl_output_joints_torch(internal_joints: torch.Tensor) -> torch.Tensor:
    """Build SONIC's 24 joints with palm centers in output slots 22/23."""

    left, right = palm_centers_torch(internal_joints)
    selected = torch.cat(
        [internal_joints[:, :22], left[:, None, :], right[:, None, :]],
        dim=1,
    )
    return output_frame_joints_torch(selected)


def root_centered_grounded_joints(
    joints: np.ndarray,
    *,
    root_index: int = ROOT_INDEX,
    foot_indices: tuple[int, int] = (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
) -> tuple[np.ndarray, np.ndarray]:
    """Put a trajectory in the comparison frame used by the pose renderer.

    Horizontal position is expressed relative to the pelvis while height is
    expressed relative to the lower of the two feet.  Keeping ground height
    instead of pelvis height is intentional: the endpoint-retargeting goal is
    for the G1-sized body to put its hands at the same physical, ground-based
    locations as the human-sized SMPL when both bodies stand at the same
    horizontal origin.

    Returns the normalized joints and the per-frame anchor that was removed.
    """

    values = np.asarray(joints, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"joints must have shape (N, J, 3), got {values.shape}")
    required = max(int(root_index), *(int(index) for index in foot_indices))
    if values.shape[1] <= required:
        raise ValueError(
            f"joints has {values.shape[1]} joints but normalization needs index {required}"
        )
    if not np.isfinite(values).all():
        raise ValueError("joints must be finite")

    ground_z = np.minimum(
        values[:, int(foot_indices[0]), 2],
        values[:, int(foot_indices[1]), 2],
    )
    anchor = np.stack(
        [
            values[:, int(root_index), 0],
            values[:, int(root_index), 1],
            ground_z,
        ],
        axis=1,
    )
    return values - anchor[:, None, :], anchor


def selected_endpoint_names(endpoint_parts: str) -> tuple[str, ...]:
    endpoint_retarget_contract(endpoint_parts)
    return ("hand_left", "hand_right")


def _archive_keys(archive: Any) -> set[str]:
    files = getattr(archive, "files", None)
    return set(files if files is not None else archive.keys())


def _archive_scalar(archive: Any, key: str) -> Any:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"candidate contract field {key!r} must be scalar")
    return value.item()


def _validate_palm_endpoint_candidate_archive(archive: Any) -> dict[str, Any]:
    """Fail closed unless a candidate proves the current palm/wrist contract.

    This validator is intentionally shared by the evaluator, production
    auditor, visualizers, resume path, and promotion path.  Merely renaming an
    old thumb-tip candidate to the current algorithm is therefore insufficient.
    """

    hand_side = hand_endpoint_side_from_archive(archive)
    experimental_lock_spine = (
        bool(_archive_scalar(archive, "experimental_lock_spine"))
        if "experimental_lock_spine" in _archive_keys(archive)
        else False
    )
    required = {
        "algorithm",
        "candidate_contract_version",
        "endpoint_parts",
        "endpoint_definition",
        "endpoint_target_contract",
        "left_palm_mcp_indices",
        "right_palm_mcp_indices",
        "optimized_joint_indices",
        "wrist_local_rotation_policy",
        "wrist_local_rotation_max_abs_delta_rad",
        "vr3pt_orientation_policy",
        "wrist_false_grasp_repair_policy",
        "wrist_false_grasp_max_island_duration_s",
        "wrist_false_grasp_min_endpoint_jump_deg",
        "wrist_false_grasp_max_upstream_step_deg",
        "wrist_false_grasp_repair_mask",
        "wrist_false_grasp_repair_event_bounds",
        "wrist_motion_filter_policy",
        "wrist_filter_cutoff_hz",
        "wrist_filter_order",
        "wrist_grasp_enter_threshold",
        "wrist_grasp_exit_threshold",
        "wrist_grasp_padding_s",
        "wrist_grasp_max_duration_s",
        "wrist_grasp_min_residual_deg",
        "wrist_grasp_max_upstream_residual_deg",
        "wrist_motion_filter_mask",
        "wrist_grasp_repair_mask",
        "wrist_grasp_repair_event_bounds",
        "pico_timestamp_s",
        "pico_left_hand_active",
        "pico_right_hand_active",
        "baseline_left_hand_command",
        "baseline_right_hand_command",
        "source_pico_body_pose",
        "false_grasp_repaired_pico_body_pose",
        "repaired_pico_body_pose",
        "source_baseline_smpl_pose",
        "baseline_vr3pt_orientation",
        "source_baseline_vr3pt_orientation",
        "candidate_vr3pt_orientation",
        "baseline_smpl_pose",
        "candidate_smpl_pose",
        "parent_candidate_sha256",
        "correction_npz_sha256",
        "canonical_baseline_full_joints",
        "canonical_candidate_full_joints",
        "canonical_baseline_joints",
        "canonical_candidate_joints",
        "virtual_baseline_joints",
        "virtual_candidate_joints",
        "target_left_grounded_m",
        "target_right_grounded_m",
    }
    missing = sorted(required - _archive_keys(archive))
    if missing:
        raise ValueError(
            "candidate is missing current palm/wrist contract fields: "
            + ", ".join(missing)
        )

    contract = endpoint_retarget_contract(
        ENDPOINT_PARTS_HANDS,
        hand_side,
        experimental_lock_spine=experimental_lock_spine,
    )
    scalar_contract = {
        "algorithm": contract.algorithm,
        "candidate_contract_version": contract.candidate_contract_version,
        "endpoint_parts": ENDPOINT_PARTS_HANDS,
        "endpoint_definition": contract.endpoint_definition,
        "endpoint_target_contract": contract.endpoint_target_contract,
        "wrist_local_rotation_policy": WRIST_LOCAL_ROTATION_POLICY,
        "vr3pt_orientation_policy": VR3PT_ORIENTATION_POLICY,
        "wrist_false_grasp_repair_policy": (
            PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
        ),
        "wrist_motion_filter_policy": PICO_WRIST_MOTION_FILTER_POLICY,
    }
    for key, expected in scalar_contract.items():
        observed = _archive_scalar(archive, key)
        if observed != expected:
            raise ValueError(
                f"candidate contract mismatch for {key}: "
                f"observed={observed!r}, expected={expected!r}"
            )

    parent_candidate_sha256 = str(
        _archive_scalar(archive, "parent_candidate_sha256")
    )
    correction_npz_sha256 = str(_archive_scalar(archive, "correction_npz_sha256"))
    if bool(parent_candidate_sha256) != bool(correction_npz_sha256):
        raise ValueError(
            "candidate lineage must contain both parent and correction hashes or neither"
        )
    for key, value in (
        ("parent_candidate_sha256", parent_candidate_sha256),
        ("correction_npz_sha256", correction_npz_sha256),
    ):
        if value and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"candidate lineage field {key} is not a SHA256 digest")

    index_contract = {
        "left_palm_mcp_indices": LEFT_PALM_MCP_INDICES,
        "right_palm_mcp_indices": RIGHT_PALM_MCP_INDICES,
        "optimized_joint_indices": contract.optimized_joint_indices,
    }
    for key, expected in index_contract.items():
        observed = tuple(int(value) for value in np.asarray(archive[key]).reshape(-1))
        if observed != tuple(expected):
            raise ValueError(
                f"candidate contract mismatch for {key}: "
                f"observed={observed}, expected={tuple(expected)}"
            )

    scalar_thresholds = {
        "wrist_false_grasp_max_island_duration_s": (
            PICO_FALSE_GRASP_MAX_ISLAND_DURATION_S
        ),
        "wrist_false_grasp_min_endpoint_jump_deg": (
            PICO_FALSE_GRASP_MIN_ENDPOINT_JUMP_DEG
        ),
        "wrist_false_grasp_max_upstream_step_deg": (
            PICO_FALSE_GRASP_MAX_UPSTREAM_STEP_DEG
        ),
        "wrist_filter_cutoff_hz": PICO_WRIST_FILTER_CUTOFF_HZ,
        "wrist_filter_order": PICO_WRIST_FILTER_ORDER,
        "wrist_grasp_enter_threshold": PICO_WRIST_GRASP_ENTER_THRESHOLD,
        "wrist_grasp_exit_threshold": PICO_WRIST_GRASP_EXIT_THRESHOLD,
        "wrist_grasp_padding_s": PICO_WRIST_GRASP_PADDING_S,
        "wrist_grasp_max_duration_s": PICO_WRIST_GRASP_MAX_DURATION_S,
        "wrist_grasp_min_residual_deg": PICO_WRIST_GRASP_MIN_RESIDUAL_DEG,
        "wrist_grasp_max_upstream_residual_deg": (
            PICO_WRIST_GRASP_MAX_UPSTREAM_RESIDUAL_DEG
        ),
    }
    for key, expected in scalar_thresholds.items():
        observed = float(_archive_scalar(archive, key))
        if observed != float(expected):
            raise ValueError(
                f"candidate wrist repair/filter threshold mismatch for {key}: "
                f"observed={observed}, expected={expected}"
            )

    source_pose = np.asarray(
        archive["source_baseline_smpl_pose"], dtype=np.float64
    )
    baseline_pose = np.asarray(archive["baseline_smpl_pose"], dtype=np.float64)
    candidate_pose = np.asarray(archive["candidate_smpl_pose"], dtype=np.float64)
    if (
        source_pose.shape != baseline_pose.shape
        or baseline_pose.shape != candidate_pose.shape
        or baseline_pose.ndim != 2
        or baseline_pose.shape[1] != 63
        or len(baseline_pose) == 0
    ):
        raise ValueError(
            "baseline/candidate SMPL pose must have non-empty matching shape (N, 63)"
        )
    rows = len(baseline_pose)
    pico_timestamps = np.asarray(
        archive["pico_timestamp_s"], dtype=np.float64
    ).reshape(-1)
    pico_left_active = np.asarray(
        archive["pico_left_hand_active"], dtype=bool
    ).reshape(-1)
    pico_right_active = np.asarray(
        archive["pico_right_hand_active"], dtype=bool
    ).reshape(-1)
    left_hand_command = np.asarray(
        archive["baseline_left_hand_command"], dtype=np.float64
    )
    right_hand_command = np.asarray(
        archive["baseline_right_hand_command"], dtype=np.float64
    )
    source_pico = np.asarray(archive["source_pico_body_pose"], dtype=np.float64)
    false_grasp_repaired_pico = np.asarray(
        archive["false_grasp_repaired_pico_body_pose"], dtype=np.float64
    )
    repaired_pico = np.asarray(
        archive["repaired_pico_body_pose"], dtype=np.float64
    )
    repair_mask = np.asarray(
        archive["wrist_false_grasp_repair_mask"], dtype=bool
    )
    event_bounds = np.asarray(
        archive["wrist_false_grasp_repair_event_bounds"], dtype=np.int64
    )
    filter_mask = np.asarray(
        archive["wrist_motion_filter_mask"], dtype=bool
    )
    grasp_repair_mask = np.asarray(
        archive["wrist_grasp_repair_mask"], dtype=bool
    )
    grasp_event_bounds = np.asarray(
        archive["wrist_grasp_repair_event_bounds"], dtype=np.int64
    )
    if (
        pico_timestamps.shape != (rows,)
        or pico_left_active.shape != (rows,)
        or pico_right_active.shape != (rows,)
        or left_hand_command.shape != (rows, 6)
        or right_hand_command.shape != (rows, 6)
        or source_pico.shape != (rows, 24, 7)
        or false_grasp_repaired_pico.shape != (rows, 24, 7)
        or repaired_pico.shape != (rows, 24, 7)
        or repair_mask.shape != (rows, 2)
        or filter_mask.shape != (rows, 2)
        or grasp_repair_mask.shape != (rows, 2)
        or event_bounds.ndim != 2
        or event_bounds.shape[1:] != (3,)
        or grasp_event_bounds.ndim != 2
        or grasp_event_bounds.shape[1:] != (3,)
    ):
        raise ValueError("candidate wrist repair/filter proof has invalid shapes")
    expected_false_grasp_pico, repair_report = (
        repair_pico_false_grasp_wrist_artifacts(
            source_pico,
            pico_left_active,
            pico_right_active,
            pico_timestamps,
        )
    )
    expected_pico, filter_report = filter_pico_wrist_motion_artifacts(
        expected_false_grasp_pico,
        left_hand_command,
        right_hand_command,
        pico_timestamps,
    )
    expected_mask = np.asarray(repair_report["repair_mask"], dtype=bool)
    expected_bounds = np.asarray(
        [
            [
                int(event["side_index"]),
                int(event["start_frame"]),
                int(event["end_frame_exclusive"]),
            ]
            for event in repair_report["events"]
        ],
        dtype=np.int64,
    ).reshape(-1, 3)
    expected_filter_mask = np.asarray(filter_report["filter_mask"], dtype=bool)
    expected_grasp_repair_mask = np.asarray(
        filter_report["grasp_repair_mask"], dtype=bool
    )
    expected_grasp_bounds = np.asarray(
        [
            [
                int(event["side_index"]),
                int(event["padded_start_frame"]),
                int(event["padded_end_frame_exclusive"]),
            ]
            for event in filter_report["events"]
            if bool(event["accepted"])
        ],
        dtype=np.int64,
    ).reshape(-1, 3)
    if not np.array_equal(repair_mask, expected_mask):
        raise ValueError("candidate false-grasp repair mask is not reproducible")
    if not np.array_equal(event_bounds, expected_bounds):
        raise ValueError("candidate false-grasp repair event bounds are not reproducible")
    if not np.allclose(
        false_grasp_repaired_pico,
        expected_false_grasp_pico,
        atol=2e-6,
        rtol=0.0,
    ):
        raise ValueError("candidate false-grasp-repaired Pico pose is not reproducible")
    if not np.array_equal(filter_mask, expected_filter_mask):
        raise ValueError("candidate wrist motion filter mask is not reproducible")
    if not np.array_equal(grasp_repair_mask, expected_grasp_repair_mask):
        raise ValueError("candidate grasp-correlated wrist repair mask is not reproducible")
    if not np.array_equal(grasp_event_bounds, expected_grasp_bounds):
        raise ValueError("candidate grasp-correlated wrist event bounds are not reproducible")
    if not np.allclose(repaired_pico, expected_pico, atol=2e-6, rtol=0.0):
        raise ValueError("candidate filtered Pico body pose is not reproducible")

    pose_change = baseline_pose.reshape(rows, 21, 3) - source_pose.reshape(
        rows, 21, 3
    )
    combined_mask = repair_mask | filter_mask
    allowed_pose_change = np.zeros((rows, 21, 3), dtype=bool)
    allowed_pose_change[combined_mask[:, 0], 19, :] = True
    allowed_pose_change[combined_mask[:, 1], 20, :] = True
    if np.any(pose_change[~allowed_pose_change] != 0.0):
        raise ValueError(
            "palm wrist filtering modified SMPL pose outside the affected wrist"
        )
    source_wrist_pose = compute_smpl_wrist_rotvecs(source_pico)
    repaired_wrist_pose = compute_smpl_wrist_rotvecs(repaired_pico)
    expected_baseline_wrist_pose = source_pose.reshape(rows, 21, 3)[:, 19:21].copy()
    for side_index in (0, 1):
        expected_baseline_wrist_pose[combined_mask[:, side_index], side_index] = (
            repaired_wrist_pose[combined_mask[:, side_index], side_index]
        )
    source_wrist_matches = all(
        np.allclose(
            source_pose.reshape(rows, 21, 3)[combined_mask[:, side_index], 19 + side_index],
            source_wrist_pose[combined_mask[:, side_index], side_index],
            atol=2e-6,
            rtol=0.0,
        )
        for side_index in (0, 1)
    )
    if not source_wrist_matches or not np.allclose(
        baseline_pose.reshape(rows, 21, 3)[:, 19:21],
        expected_baseline_wrist_pose,
        atol=2e-6,
        rtol=0.0,
    ):
        raise ValueError("filtered SMPL wrist does not match Pico conversion")

    wrist_delta = candidate_pose[:, 57:63] - baseline_pose[:, 57:63]
    wrist_delta_max = float(np.max(np.abs(wrist_delta), initial=0.0))
    declared_wrist_delta = float(
        _archive_scalar(archive, "wrist_local_rotation_max_abs_delta_rad")
    )
    if wrist_delta_max > 1e-7 or abs(declared_wrist_delta - wrist_delta_max) > 1e-9:
        raise ValueError(
            "candidate modified or misreported Pico wrist-local rotation: "
            f"measured={wrist_delta_max}, declared={declared_wrist_delta}"
        )

    from egoalign.staged_ik import validate_proof
    validate_proof(archive, baseline_pose, candidate_pose,
                   np.stack([archive["target_left_m"], archive["target_right_m"]], axis=1))

    candidate_pose_change = candidate_pose.reshape(rows, 21, 3) - baseline_pose.reshape(
        rows, 21, 3
    )
    mutable_pose_indices = {
        joint_index - 1 for joint_index in contract.optimized_joint_indices
    }
    locked_pose_indices = [
        pose_index for pose_index in range(21) if pose_index not in mutable_pose_indices
    ]
    locked_pose_delta_max = float(
        np.max(
            np.abs(candidate_pose_change[:, locked_pose_indices]),
            initial=0.0,
        )
    )
    if locked_pose_delta_max > 1e-7:
        raise ValueError(
            "candidate modified a locked SMPL joint outside the palm natural-pose "
            f"whitelist: max_abs_delta_rad={locked_pose_delta_max}"
        )

    source_orientation = np.asarray(
        archive["source_baseline_vr3pt_orientation"], dtype=np.float64
    )
    baseline_orientation = np.asarray(
        archive["baseline_vr3pt_orientation"], dtype=np.float64
    )
    candidate_orientation = np.asarray(
        archive["candidate_vr3pt_orientation"], dtype=np.float64
    )
    if (
        source_orientation.shape != baseline_orientation.shape
        or baseline_orientation.shape != candidate_orientation.shape
        or baseline_orientation.ndim != 2
        or baseline_orientation.shape[1] != 18
        or not np.array_equal(baseline_orientation, candidate_orientation)
    ):
        raise ValueError(
            "candidate must preserve denoised-baseline vr_3pt_orientation exactly"
        )

    orientation_change = baseline_orientation - source_orientation
    allowed_orientation_change = np.zeros((rows, 18), dtype=bool)
    allowed_orientation_change[combined_mask[:, 0], 0:6] = True
    allowed_orientation_change[combined_mask[:, 1], 6:12] = True
    if np.any(orientation_change[~allowed_orientation_change] != 0.0):
        raise ValueError(
            "palm wrist filtering modified VR3 orientation outside the affected wrist"
        )
    source_vr3_wrist = compute_vr3_wrist_orientation_blocks(source_pico)
    repaired_vr3_wrist = compute_vr3_wrist_orientation_blocks(repaired_pico)
    expected_baseline_vr3_wrist = source_orientation[:, :12].reshape(
        rows, 2, 6
    ).copy()
    for side_index in (0, 1):
        expected_baseline_vr3_wrist[
            combined_mask[:, side_index], side_index
        ] = repaired_vr3_wrist[combined_mask[:, side_index], side_index]
    source_orientation_matches = all(
        np.allclose(
            source_orientation[:, :12].reshape(rows, 2, 6)[
                combined_mask[:, side_index], side_index
            ],
            source_vr3_wrist[combined_mask[:, side_index], side_index],
            atol=2e-6,
            rtol=0.0,
        )
        for side_index in (0, 1)
    )
    if not source_orientation_matches or not np.allclose(
        baseline_orientation[:, :12].reshape(rows, 2, 6),
        expected_baseline_vr3_wrist,
        atol=2e-6,
        rtol=0.0,
    ):
        raise ValueError(
            "filtered VR3 wrist orientation does not match Pico conversion"
        )

    canonical_baseline_full = np.asarray(
        archive["canonical_baseline_full_joints"], dtype=np.float64
    )
    canonical_candidate_full = np.asarray(
        archive["canonical_candidate_full_joints"], dtype=np.float64
    )
    canonical_baseline = np.asarray(archive["canonical_baseline_joints"], dtype=np.float64)
    canonical_candidate = np.asarray(archive["canonical_candidate_joints"], dtype=np.float64)
    virtual_baseline = np.asarray(archive["virtual_baseline_joints"], dtype=np.float64)
    virtual_candidate = np.asarray(archive["virtual_candidate_joints"], dtype=np.float64)
    target_left = np.asarray(archive["target_left_grounded_m"], dtype=np.float64)
    target_right = np.asarray(archive["target_right_grounded_m"], dtype=np.float64)
    left_correction = (
        np.asarray(archive["left_target_correction_m"], dtype=np.float64)
        if "left_target_correction_m" in _archive_keys(archive)
        else np.zeros((rows, 3), dtype=np.float64)
    )
    right_correction = (
        np.asarray(archive["right_target_correction_m"], dtype=np.float64)
        if "right_target_correction_m" in _archive_keys(archive)
        else np.zeros((rows, 3), dtype=np.float64)
    )
    if (
        canonical_baseline_full.shape != (rows, 55, 3)
        or canonical_candidate_full.shape != (rows, 55, 3)
    ):
        raise ValueError("full canonical candidate contract requires shape (N, 55, 3)")
    if (
        canonical_baseline.shape != (rows, 24, 3)
        or canonical_candidate.shape != (rows, 24, 3)
    ):
        raise ValueError("canonical candidate contract requires shape (N, 24, 3)")
    if (
        virtual_baseline.shape != (rows, 55, 3)
        or virtual_candidate.shape != (rows, 55, 3)
    ):
        raise ValueError("virtual candidate contract requires shape (N, 55, 3)")
    if target_left.shape != (rows, 3) or target_right.shape != (rows, 3):
        raise ValueError("grounded palm targets must have shape (N, 3)")
    if left_correction.shape != (rows, 3) or right_correction.shape != (rows, 3):
        raise ValueError("palm corrections must have shape (N, 3)")
    arrays = (
        source_pose,
        baseline_pose,
        candidate_pose,
        source_pico,
        false_grasp_repaired_pico,
        repaired_pico,
        pico_timestamps,
        left_hand_command,
        right_hand_command,
        canonical_baseline_full,
        canonical_candidate_full,
        canonical_baseline,
        canonical_candidate,
        virtual_baseline,
        virtual_candidate,
        target_left,
        target_right,
        left_correction,
        right_correction,
        source_orientation,
        baseline_orientation,
        candidate_orientation,
    )
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("candidate palm/wrist contract arrays must be finite")

    def output_24_from_full(full: np.ndarray) -> np.ndarray:
        left = np.mean(full[:, list(LEFT_PALM_MCP_INDICES)], axis=1)
        right = np.mean(full[:, list(RIGHT_PALM_MCP_INDICES)], axis=1)
        return np.concatenate(
            [full[:, :22], left[:, None, :], right[:, None, :]], axis=1
        )

    reconstructed_baseline = output_24_from_full(canonical_baseline_full)
    reconstructed_candidate = output_24_from_full(canonical_candidate_full)
    if not np.allclose(
        reconstructed_baseline, canonical_baseline, atol=2e-6, rtol=0.0
    ) or not np.allclose(
        reconstructed_candidate, canonical_candidate, atol=2e-6, rtol=0.0
    ):
        raise ValueError(
            "canonical slots 22/23 are not the mean of the four non-thumb MCP roots"
        )

    grounded_baseline, _ = root_centered_grounded_joints(canonical_baseline)
    if not np.allclose(
        grounded_baseline[:, LEFT_ENDPOINT_OUTPUT_INDEX], target_left, atol=2e-6, rtol=0.0
    ) or not np.allclose(
        grounded_baseline[:, RIGHT_ENDPOINT_OUTPUT_INDEX], target_right, atol=2e-6, rtol=0.0
    ):
        raise ValueError("grounded targets do not match canonical palm-center slots 22/23")

    left_virtual = np.mean(virtual_candidate[:, list(LEFT_PALM_MCP_INDICES)], axis=1)
    right_virtual = np.mean(virtual_candidate[:, list(RIGHT_PALM_MCP_INDICES)], axis=1)
    return {
        "algorithm": contract.algorithm,
        "candidate_contract_version": contract.candidate_contract_version,
        "endpoint_parts": ENDPOINT_PARTS_HANDS,
        "hand_endpoint_side": hand_side,
        "experimental_lock_spine": experimental_lock_spine,
        "endpoint_definition": contract.endpoint_definition,
        "left_palm_mcp_indices": list(LEFT_PALM_MCP_INDICES),
        "right_palm_mcp_indices": list(RIGHT_PALM_MCP_INDICES),
        "optimized_joint_indices": list(contract.optimized_joint_indices),
        "wrist_local_rotation_max_abs_delta_rad": wrist_delta_max,
        "locked_joint_max_abs_delta_rad": locked_pose_delta_max,
        "wrist_false_grasp_repair_policy": (
            PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
        ),
        "wrist_false_grasp_repair_event_count": int(len(event_bounds)),
        "wrist_false_grasp_repaired_frame_count": int(
            np.sum(np.any(repair_mask, axis=1))
        ),
        "wrist_motion_filter_policy": PICO_WRIST_MOTION_FILTER_POLICY,
        "wrist_filtered_frame_count": int(
            np.sum(np.any(filter_mask, axis=1))
        ),
        "wrist_grasp_repair_event_count": int(len(grasp_event_bounds)),
        "wrist_grasp_repaired_frame_count": int(
            np.sum(np.any(grasp_repair_mask, axis=1))
        ),
        "wrist_filter_correction": filter_report["correction"],
        "wrist_source_local_step": filter_report["source_local_wrist_step"],
        "wrist_filtered_local_step": filter_report["filtered_local_wrist_step"],
        "vr3pt_orientation_exactly_preserved": True,
        "has_closed_loop_parent": bool(parent_candidate_sha256),
        "rows": rows,
        "virtual_candidate_palms_finite": bool(
            np.isfinite(left_virtual).all() and np.isfinite(right_virtual).all()
        ),
    }


def validate_endpoint_candidate_archive(archive: Any) -> dict[str, Any]:
    """Validate measured-target bilateral-palm candidate provenance and invariants."""
    endpoint_parts_from_archive(archive)
    return _validate_palm_endpoint_candidate_archive(archive)


def _validate_palm_endpoint_correction_archive(
    archive: Any,
    *,
    expected_parent_candidate_sha256: str,
    expected_shape: tuple[int, int],
    expected_hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> dict[str, Any]:
    """Validate that a correction belongs to one current-contract candidate."""

    required = {
        "algorithm",
        "candidate_contract_version",
        "endpoint_parts",
        "endpoint_definition",
        "wrist_local_rotation_policy",
        "wrist_false_grasp_repair_policy",
        "wrist_motion_filter_policy",
        "source_candidate_sha256",
        "left_target_correction_m",
        "right_target_correction_m",
    }
    missing = sorted(required - _archive_keys(archive))
    if missing:
        raise ValueError(
            "correction is missing current palm/wrist contract fields: "
            + ", ".join(missing)
        )
    contract = endpoint_retarget_contract(
        ENDPOINT_PARTS_HANDS, expected_hand_endpoint_side
    )
    scalar_contract = {
        "algorithm": contract.algorithm,
        "candidate_contract_version": contract.candidate_contract_version,
        "endpoint_parts": ENDPOINT_PARTS_HANDS,
        "endpoint_definition": contract.endpoint_definition,
        "wrist_local_rotation_policy": WRIST_LOCAL_ROTATION_POLICY,
        "wrist_false_grasp_repair_policy": (
            PICO_FALSE_GRASP_WRIST_REPAIR_POLICY
        ),
        "wrist_motion_filter_policy": PICO_WRIST_MOTION_FILTER_POLICY,
        "source_candidate_sha256": expected_parent_candidate_sha256,
    }
    for key, expected in scalar_contract.items():
        observed = _archive_scalar(archive, key)
        if observed != expected:
            raise ValueError(
                f"correction handoff has incompatible {key}: "
                f"observed={observed!r}, expected={expected!r}"
            )
    left = np.asarray(archive["left_target_correction_m"], dtype=np.float64)
    right = np.asarray(archive["right_target_correction_m"], dtype=np.float64)
    if left.shape != expected_shape or right.shape != expected_shape:
        raise ValueError(
            "correction arrays must match target shape: "
            f"left={left.shape}, right={right.shape}, expected={expected_shape}"
        )
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("correction arrays must be finite")
    return {
        "algorithm": contract.algorithm,
        "candidate_contract_version": contract.candidate_contract_version,
        "endpoint_parts": ENDPOINT_PARTS_HANDS,
        "hand_endpoint_side": expected_hand_endpoint_side,
        "endpoint_definition": contract.endpoint_definition,
        "source_candidate_sha256": expected_parent_candidate_sha256,
        "shape": list(expected_shape),
    }


def validate_endpoint_correction_archive(
    archive: Any,
    *,
    expected_parent_candidate_sha256: str,
    expected_shape: tuple[int, int],
    expected_endpoint_parts: str = ENDPOINT_PARTS_HANDS,
    expected_hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> dict[str, Any]:
    """Validate a hash-linked bilateral-palm residual correction handoff."""
    endpoint_retarget_contract(expected_endpoint_parts, expected_hand_endpoint_side)
    return _validate_palm_endpoint_correction_archive(
        archive,
        expected_parent_candidate_sha256=expected_parent_candidate_sha256,
        expected_shape=expected_shape,
        expected_hand_endpoint_side=expected_hand_endpoint_side,
    )


def direct_big_smpl_endpoint_targets(
    big_smpl_joints: np.ndarray,
    virtual_baseline_joints: np.ndarray,
    *,
    big_left_endpoint_index: int = LEFT_ENDPOINT_OUTPUT_INDEX,
    big_right_endpoint_index: int = RIGHT_ENDPOINT_OUTPUT_INDEX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map big-SMPL hand locations into the virtual-small-SMPL FK frame.

    The desired positions are *absolute positions in the shared comparison
    frame*, not motion deltas anchored at frame zero.  Both bodies are centered
    horizontally at their pelvis and placed on the ground.  The normalized big
    endpoints are then re-anchored in the virtual skeleton's raw FK frame so
    the optimizer can compare them directly with ``forward_joints_torch``.

    Returns raw-FK left/right targets followed by the normalized left/right
    big-SMPL endpoints used for visual and replay evaluation.
    """

    big = np.asarray(big_smpl_joints, dtype=np.float64)
    virtual = np.asarray(virtual_baseline_joints, dtype=np.float64)
    if big.ndim != 3 or big.shape[2] != 3:
        raise ValueError(f"big_smpl_joints must have shape (N, J, 3), got {big.shape}")
    if virtual.ndim != 3 or virtual.shape[2] != 3:
        raise ValueError(
            "virtual_baseline_joints must have shape (N, J, 3), "
            f"got {virtual.shape}"
        )
    if len(big) != len(virtual):
        raise ValueError("big and virtual SMPL trajectories must have equal frame counts")
    if big.shape[1] <= max(big_left_endpoint_index, big_right_endpoint_index):
        raise ValueError("big SMPL trajectory is missing palm-center endpoint joints")

    big_normalized, _ = root_centered_grounded_joints(big)
    _, virtual_anchor = root_centered_grounded_joints(virtual)
    left_grounded = big_normalized[:, int(big_left_endpoint_index)]
    right_grounded = big_normalized[:, int(big_right_endpoint_index)]
    target_left = virtual_anchor + left_grounded
    target_right = virtual_anchor + right_grounded
    return target_left, target_right, left_grounded, right_grounded


def canonical_output_joints(
    pose: np.ndarray,
    canonical_rest_joints: np.ndarray,
    parents: list[int],
    baseline_output_joints: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Rebuild canonical 24-joint SONIC input while preserving the root transform."""

    output, _ = canonical_output_joints_with_full(
        pose,
        canonical_rest_joints,
        parents,
        baseline_output_joints,
        device=device,
    )
    return output


def canonical_output_joints_with_full(
    pose: np.ndarray,
    canonical_rest_joints: np.ndarray,
    parents: list[int],
    baseline_output_joints: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return palm-slot SONIC joints plus the translated full 55-joint FK.

    Keeping the full FK trajectory in the candidate archive lets downstream
    consumers independently prove that output slots 22/23 were synthesized
    from the four non-thumb MCP roots instead of trusting an algorithm label.
    """

    pose_t = torch.as_tensor(pose, dtype=torch.float32, device=device)
    rest_t = torch.as_tensor(canonical_rest_joints, dtype=torch.float32, device=device)
    with torch.no_grad():
        internal = forward_joints_torch(pose_t, rest_t, parents)
        full = output_frame_joints_torch(internal)
        left, right = palm_centers_torch(full)
        output = torch.cat(
            [full[:, :22], left[:, None, :], right[:, None, :]], dim=1
        )
        full_np = full.cpu().numpy().astype(np.float64)
        output_np = output.cpu().numpy().astype(np.float64)
    baseline = np.asarray(baseline_output_joints, dtype=np.float64)
    if baseline.shape != output_np.shape:
        raise ValueError(
            f"baseline_output_joints shape mismatch: {baseline.shape} vs {output_np.shape}"
        )
    translation = baseline[:, :1, :] - output_np[:, :1, :]
    return (
        (output_np + translation).astype(np.float32),
        (full_np + translation).astype(np.float32),
    )


def smoothstep_ramp(frames: int, ramp_frames: int) -> np.ndarray:
    values = np.ones(int(frames), dtype=np.float64)
    if int(ramp_frames) <= 0:
        return values
    ramp_frames = max(1, min(int(ramp_frames), int(frames)))
    t = np.linspace(0.0, 1.0, ramp_frames, dtype=np.float64)
    values[:ramp_frames] = t * t * (3.0 - 2.0 * t)
    return values


def _optimize_arm_trajectory(
    *,
    baseline_pose: np.ndarray,
    virtual_rest_joints: np.ndarray,
    parents: list[int],
    target_left: np.ndarray,
    target_right: np.ndarray,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
    initial_pose: np.ndarray | None = None,
    optimized_joint_indices: tuple[int, ...] = ARM_OPTIMIZED_JOINT_INDICES,
    joint_delta_limits_rad: np.ndarray | None = None,
    weights: OptimizationWeights = OptimizationWeights(),
    ramp_frames: int = 0,
    iterations: int = 350,
    learning_rate: float = 0.035,
    device: str | None = None,
    log_every: int = 50,
) -> OptimizationResult:
    baseline_np = np.asarray(baseline_pose, dtype=np.float32).reshape(-1, 21, 3)
    initial_np = baseline_np if initial_pose is None else np.asarray(initial_pose, dtype=np.float32).reshape(-1, 21, 3)
    if initial_np.shape != baseline_np.shape:
        raise ValueError("initial_pose must match baseline_pose")
    frames = len(baseline_np)
    target_left_np = np.asarray(target_left, dtype=np.float32).reshape(frames, 3)
    target_right_np = np.asarray(target_right, dtype=np.float32).reshape(frames, 3)
    side = str(hand_endpoint_side).strip().lower()
    if side not in HAND_ENDPOINT_SIDE_CHOICES:
        raise ValueError(f"unsupported hand_endpoint_side: {hand_endpoint_side!r}")
    active = tuple(int(index) for index in optimized_joint_indices)
    if active != ARM_OPTIMIZED_JOINT_INDICES:
        raise ValueError("EgoAlign requires bilateral arm-only optimization in every round")
    if any(index < 1 or index > 21 for index in active):
        raise ValueError("optimized joints must be SMPL body indices in [1, 21]")

    if joint_delta_limits_rad is None:
        limits = np.array(
            [0.35 if index <= 14 else 0.90 if index <= 17 else 0.75 if index <= 19 else 0.50 for index in active],
            dtype=np.float32,
        )
    else:
        limits = np.asarray(joint_delta_limits_rad, dtype=np.float32).reshape(len(active))
    if np.any(limits <= 0.0) or not np.isfinite(limits).all():
        raise ValueError("joint_delta_limits_rad must be finite and positive")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(selected_device)
    baseline = torch.as_tensor(baseline_np, device=torch_device)
    initial = torch.as_tensor(initial_np, device=torch_device)
    rest = torch.as_tensor(virtual_rest_joints, dtype=torch.float32, device=torch_device)
    target = torch.as_tensor(
        np.concatenate([target_left_np, target_right_np], axis=1),
        dtype=torch.float32,
        device=torch_device,
    )
    limit_t = torch.as_tensor(limits, dtype=torch.float32, device=torch_device)[None, :, None]
    ramp = torch.as_tensor(
        smoothstep_ramp(frames, ramp_frames), dtype=torch.float32, device=torch_device
    )[:, None, None]

    initial_delta = initial[:, np.asarray(active) - 1] - baseline[:, np.asarray(active) - 1]
    normalized = torch.clamp(initial_delta / (limit_t * ramp.clamp_min(1e-3)), -0.95, 0.95)
    parameter = torch.nn.Parameter(torch.atanh(normalized))
    optimizer = torch.optim.Adam([parameter], lr=float(learning_rate))
    losses: list[dict[str, float]] = []

    for iteration in range(max(1, int(iterations))):
        optimizer.zero_grad(set_to_none=True)
        candidate = baseline.clone()
        delta = limit_t * torch.tanh(parameter) * ramp
        # Retain six slots and the original loss normalization; freeze clavicles.
        delta = delta * torch.tensor([0, 0, 1, 1, 1, 1], device=delta.device)[None, :, None]
        candidate[:, np.asarray(active) - 1] = baseline[:, np.asarray(active) - 1] + delta
        internal = forward_joints_torch(candidate.reshape(frames, 63), rest, parents)
        output = output_frame_joints_torch(internal)
        left_palm, right_palm = palm_centers_torch(output)
        endpoints = torch.cat([left_palm, right_palm], dim=1)
        position_loss = torch.mean((endpoints - target) ** 2)
        pose_loss = torch.mean(delta**2)
        velocity_loss = (
            torch.mean((delta[1:] - delta[:-1]) ** 2) if frames > 1 else delta.sum() * 0.0
        )
        acceleration_loss = (
            torch.mean((delta[2:] - 2.0 * delta[1:-1] + delta[:-2]) ** 2)
            if frames > 2
            else delta.sum() * 0.0
        )
        first_loss = torch.mean(delta[0] ** 2)
        total = (
            weights.position * position_loss
            + weights.pose_prior * pose_loss
            + weights.velocity * velocity_loss
            + weights.acceleration * acceleration_loss
            + weights.first_frame * first_loss
        )
        total.backward()
        optimizer.step()

        if iteration == 0 or (iteration + 1) % max(1, int(log_every)) == 0 or iteration + 1 == iterations:
            record = {
                "iteration": float(iteration + 1),
                "total": float(total.detach().cpu()),
                "position": float(position_loss.detach().cpu()),
                "pose_prior": float(pose_loss.detach().cpu()),
                "velocity": float(velocity_loss.detach().cpu()),
                "acceleration": float(acceleration_loss.detach().cpu()),
            }
            losses.append(record)
            print(
                "endpoint-retarget "
                f"iter={iteration + 1}/{iterations} total={record['total']:.6f} "
                f"position_rmse_m={record['position'] ** 0.5:.5f}",
                flush=True,
            )

    with torch.no_grad():
        delta = limit_t * torch.tanh(parameter) * ramp
        # Retain six slots and the original loss normalization; freeze clavicles.
        delta = delta * torch.tensor([0, 0, 1, 1, 1, 1], device=delta.device)[None, :, None]
        candidate = baseline.clone()
        candidate[:, np.asarray(active) - 1] = baseline[:, np.asarray(active) - 1] + delta
        virtual = output_frame_joints_torch(
            forward_joints_torch(candidate.reshape(frames, 63), rest, parents)
        )
    return OptimizationResult(
        pose=candidate.cpu().numpy().reshape(frames, 63).astype(np.float32),
        virtual_joints=virtual.cpu().numpy().astype(np.float32),
        losses=losses,
        optimized_joint_indices=active,
        joint_delta_limits_rad=limits,
    )


def optimize_virtual_smpl_trajectory(
    *, baseline_pose, virtual_rest_joints, parents, target_left, target_right,
    hand_endpoint_side=HAND_ENDPOINT_SIDE_BOTH, initial_pose=None,
    optimized_joint_indices=DEFAULT_OPTIMIZED_JOINT_INDICES,
    joint_delta_limits_rad=None, weights=OptimizationWeights(), ramp_frames=0,
    iterations=350, learning_rate=0.035, device=None, log_every=50,
):
    from egoalign.staged_ik import optimize_staged
    if hand_endpoint_side != "both" or tuple(optimized_joint_indices) != DEFAULT_OPTIMIZED_JOINT_INDICES:
        raise ValueError("Staged IK requires the bilateral arm-first gated-spine contract")
    if joint_delta_limits_rad is not None:
        raise ValueError("Staged IK bounds are part of its fixed config identity")
    return optimize_staged(
        baseline_pose=baseline_pose, virtual_rest_joints=virtual_rest_joints,
        parents=parents, target_left=target_left, target_right=target_right,
        initial_pose=initial_pose, arm_solver=_optimize_arm_trajectory,
        weights=weights, ramp_frames=ramp_frames, iterations=iterations,
        learning_rate=learning_rate, device=device, log_every=log_every,
    )


def endpoint_error_summary(
    left_actual: np.ndarray,
    right_actual: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    hand_endpoint_side: str = HAND_ENDPOINT_SIDE_BOTH,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    all_errors: list[np.ndarray] = []
    for side, actual, target in (
        ("left", left_actual, left_target),
        ("right", right_actual, right_target),
    ):
        residual = np.asarray(target, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
        errors = np.linalg.norm(residual, axis=1)
        all_errors.append(errors)
        summary[side] = {
            "mean_m": float(np.mean(errors)),
            "p95_m": float(np.percentile(errors, 95)),
            "max_m": float(np.max(errors)),
            "axis_mean_abs_m": np.mean(np.abs(residual), axis=0).tolist(),
        }
    side = str(hand_endpoint_side).strip().lower()
    if side not in HAND_ENDPOINT_SIDE_CHOICES:
        raise ValueError(f"unsupported hand_endpoint_side: {hand_endpoint_side!r}")
    combined = (np.concatenate(all_errors))
    summary["combined"] = {
        "mean_m": float(np.mean(combined)),
        "p95_m": float(np.percentile(combined, 95)),
        "max_m": float(np.max(combined)),
    }
    return summary
