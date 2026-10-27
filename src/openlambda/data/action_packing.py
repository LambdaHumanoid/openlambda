"""Strict tensor packing for the production action-space contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from lambda0.models.action_specs import (
    FINGER_GEOMETRY_DIM,
    REVO2_HAND_DIM,
    SIMPLE_G1_ARM_DIM,
    SIMPLE_G1_HAND_DIM,
    SIMPLE_G1_NAVIGATION_DIM,
    SIMPLE_G1_TORSO_POSE_DIM,
    SONIC_MOTION_DIM,
    WRIST_POSITION_DIM,
    WRIST_ROTATION_6D_DIM,
    ActionSpaceSpec,
    get_action_space_spec,
)


def _component(name: str, value: torch.Tensor, width: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim == 0 or value.shape[-1] != width:
        raise ValueError(
            f"{name} must end with {width} channels, got {tuple(value.shape)}."
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")
    return value


def _pack(components: Mapping[str, tuple[torch.Tensor, int]]) -> torch.Tensor:
    validated = [
        _component(name, value, width)
        for name, (value, width) in components.items()
    ]
    leading_shape = validated[0].shape[:-1]
    for name, value in zip(components, validated, strict=True):
        if value.shape[:-1] != leading_shape:
            raise ValueError(
                f"{name} has leading shape {tuple(value.shape[:-1])}, "
                f"expected {tuple(leading_shape)}."
            )
    return torch.cat(validated, dim=-1)


def pack_human_hand_geometry(
    left_wrist_position: torch.Tensor,
    left_wrist_rotation_6d: torch.Tensor,
    left_finger_geometry: torch.Tensor,
    right_wrist_position: torch.Tensor,
    right_wrist_rotation_6d: torch.Tensor,
    right_finger_geometry: torch.Tensor,
) -> torch.Tensor:
    """Pack the 138D Human-Ego action or current proprioception vector."""
    return _pack(
        {
            "left_wrist_position": (
                left_wrist_position,
                WRIST_POSITION_DIM,
            ),
            "left_wrist_rotation_6d": (
                left_wrist_rotation_6d,
                WRIST_ROTATION_6D_DIM,
            ),
            "left_finger_geometry": (
                left_finger_geometry,
                FINGER_GEOMETRY_DIM,
            ),
            "right_wrist_position": (
                right_wrist_position,
                WRIST_POSITION_DIM,
            ),
            "right_wrist_rotation_6d": (
                right_wrist_rotation_6d,
                WRIST_ROTATION_6D_DIM,
            ),
            "right_finger_geometry": (
                right_finger_geometry,
                FINGER_GEOMETRY_DIM,
            ),
        }
    )


def pack_human_fullbody_sonic_geometry(
    sonic_motion: torch.Tensor,
    left_finger_geometry: torch.Tensor,
    right_finger_geometry: torch.Tensor,
) -> torch.Tensor:
    """Pack the 184D Human-Fullbody action vector."""
    return _pack(
        {
            "sonic_motion": (sonic_motion, SONIC_MOTION_DIM),
            "left_finger_geometry": (
                left_finger_geometry,
                FINGER_GEOMETRY_DIM,
            ),
            "right_finger_geometry": (
                right_finger_geometry,
                FINGER_GEOMETRY_DIM,
            ),
        }
    )


def pack_human_fullbody_proprio(
    left_finger_geometry: torch.Tensor,
    right_finger_geometry: torch.Tensor,
) -> torch.Tensor:
    """Pack current left/right finger geometry without a future SONIC token."""
    return _pack(
        {
            "left_finger_geometry": (
                left_finger_geometry,
                FINGER_GEOMETRY_DIM,
            ),
            "right_finger_geometry": (
                right_finger_geometry,
                FINGER_GEOMETRY_DIM,
            ),
        }
    )


def pack_human_fullbody_hand_state(
    left_wrist_position: torch.Tensor,
    left_wrist_rotation_6d: torch.Tensor,
    left_finger_geometry: torch.Tensor,
    right_wrist_position: torch.Tensor,
    right_wrist_rotation_6d: torch.Tensor,
    right_finger_geometry: torch.Tensor,
) -> torch.Tensor:
    """Pack the 138D current camera-frame hand state for SONIC mid-training."""
    return pack_human_hand_geometry(
        left_wrist_position,
        left_wrist_rotation_6d,
        left_finger_geometry,
        right_wrist_position,
        right_wrist_rotation_6d,
        right_finger_geometry,
    )


def pack_robot_sonic_revo2(
    sonic_motion: torch.Tensor,
    left_revo2: torch.Tensor,
    right_revo2: torch.Tensor,
) -> torch.Tensor:
    """Pack the 76D executable robot action vector."""
    return _pack(
        {
            "sonic_motion": (sonic_motion, SONIC_MOTION_DIM),
            "left_revo2": (left_revo2, REVO2_HAND_DIM),
            "right_revo2": (right_revo2, REVO2_HAND_DIM),
        }
    )


def pack_robot_proprio(
    body_joints: torch.Tensor,
    left_revo2: torch.Tensor,
    right_revo2: torch.Tensor,
    projected_gravity: torch.Tensor,
) -> torch.Tensor:
    """Pack 29D body joints, 12D Revo2 state, and 3D projected gravity."""
    return _pack(
        {
            "body_joints": (body_joints, 29),
            "left_revo2": (left_revo2, REVO2_HAND_DIM),
            "right_revo2": (right_revo2, REVO2_HAND_DIM),
            "projected_gravity": (projected_gravity, 3),
        }
    )


def pack_robot_g1_simple_sim(
    left_hand: torch.Tensor,
    right_hand: torch.Tensor,
    left_arm: torch.Tensor,
    right_arm: torch.Tensor,
    torso_pose: torch.Tensor,
    torso_navigation: torch.Tensor,
) -> torch.Tensor:
    """Pack the 36D executable G1 action used by the SIMPLE simulator."""
    return _pack(
        {
            "left_hand": (left_hand, SIMPLE_G1_HAND_DIM),
            "right_hand": (right_hand, SIMPLE_G1_HAND_DIM),
            "left_arm": (left_arm, SIMPLE_G1_ARM_DIM),
            "right_arm": (right_arm, SIMPLE_G1_ARM_DIM),
            "torso_pose": (torso_pose, SIMPLE_G1_TORSO_POSE_DIM),
            "torso_navigation": (
                torso_navigation,
                SIMPLE_G1_NAVIGATION_DIM,
            ),
        }
    )


def pack_robot_g1_simple_proprio(
    left_hand: torch.Tensor,
    right_hand: torch.Tensor,
    left_arm: torch.Tensor,
    right_arm: torch.Tensor,
    torso_pose: torch.Tensor,
) -> torch.Tensor:
    """Pack the 32D current G1 state used by the SIMPLE simulator."""
    return _pack(
        {
            "left_hand": (left_hand, SIMPLE_G1_HAND_DIM),
            "right_hand": (right_hand, SIMPLE_G1_HAND_DIM),
            "left_arm": (left_arm, SIMPLE_G1_ARM_DIM),
            "right_arm": (right_arm, SIMPLE_G1_ARM_DIM),
            "torso_pose": (torso_pose, SIMPLE_G1_TORSO_POSE_DIM),
        }
    )


def split_action_blocks(
    action: torch.Tensor,
    action_space: int | str | ActionSpaceSpec,
) -> dict[str, torch.Tensor]:
    """Return named action views after validating the unpadded channel width."""
    spec = get_action_space_spec(action_space)
    if not isinstance(action, torch.Tensor):
        raise TypeError("action must be a torch.Tensor.")
    if action.ndim == 0 or action.shape[-1] != spec.action_dim:
        raise ValueError(
            f"{spec.name} action must end with {spec.action_dim} channels, "
            f"got {tuple(action.shape)}."
        )
    return {
        block.name: action[..., block.start : block.end]
        for block in spec.blocks
    }


def slice_action_trajectory(
    absolute_trajectory: torch.Tensor,
    delta_indices: Sequence[int] = (),
    masked_proprio_indices: Sequence[int] = (),
) -> dict[str, torch.Tensor]:
    """Split `[current,future...]` and apply configured delta/mask fields."""
    if not isinstance(absolute_trajectory, torch.Tensor):
        raise TypeError("absolute_trajectory must be a torch.Tensor.")
    if (
        absolute_trajectory.ndim != 2
        or absolute_trajectory.shape[0] < 2
    ):
        raise ValueError("absolute_trajectory must be [H+1,D] with H>=1.")

    proprio = absolute_trajectory[0].clone()
    action = absolute_trajectory[1:].clone()
    if delta_indices:
        indices = torch.as_tensor(
            delta_indices,
            dtype=torch.long,
            device=absolute_trajectory.device,
        )
        action[:, indices] -= proprio[indices]
    if masked_proprio_indices:
        indices = torch.as_tensor(
            masked_proprio_indices,
            dtype=torch.long,
            device=absolute_trajectory.device,
        )
        proprio[indices] = 0.0
    return {"proprio": proprio, "action": action}


__all__ = [
    "pack_human_hand_geometry",
    "pack_human_fullbody_sonic_geometry",
    "pack_human_fullbody_proprio",
    "pack_human_fullbody_hand_state",
    "pack_robot_sonic_revo2",
    "pack_robot_proprio",
    "pack_robot_g1_simple_sim",
    "pack_robot_g1_simple_proprio",
    "split_action_blocks",
    "slice_action_trajectory",
]
