"""Conversions between physical SONIC state and model training state.

The training representation stores 29 body values in SONIC's IsaacLab order,
relative to default joint angles. The surrounding 43-D vector uses RobotModel
joint slots. The live ``g1_debug`` path emits absolute joint angles in
MuJoCo/grouped order, so it requires the corresponding state conversion.

Hand values are deliberately left untouched by the legacy body conversion.
Final feedback-generated datasets store each BrainCo hand in seven-value wire
order directly in the fixed 43-D hand slots.  Both physical feedback and C++'s
final OPEN/CLAMP/CURL command use that wire order. ``RobotModel`` uses a
different actuated-joint order, so live observations must restore the wire
order after constructing the physical qpos; see
``restore_brainco_wire_hand_slots``.
"""

from __future__ import annotations

import numpy as np


G1_FULL_STATE_DIM = 43
G1_BODY_DIM = 29
G1_BRAINCO_WIRE_DIM = 7
G1_LEFT_SHOULDER_PITCH_MUJOCO_INDEX = 15

# Body slots in the 43-D RobotModel configuration.  The left hand sits between
# the left and right arm groups in the flattened Pinocchio configuration.
G1_BODY_STATE_INDICES = np.asarray(
    [*range(0, 22), *range(29, 36)], dtype=np.int64
)
G1_LEFT_HAND_STATE_INDICES = np.arange(22, 29, dtype=np.int64)
G1_RIGHT_HAND_STATE_INDICES = np.arange(36, 43, dtype=np.int64)

# Index a MuJoCo/grouped-order vector with this array to obtain IsaacLab order.
G1_MUJOCO_TO_ISAACLAB = np.asarray(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)

# SONIC nominal pose in MuJoCo/grouped order.  These values must stay identical
# to the C++ g1_debug output conversion and the dataset converter.
G1_DEFAULT_ANGLES_MUJOCO = np.asarray(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def remove_left_shoulder_back_offset_from_vla_state(
    body_q: np.ndarray,
    offset_deg: float,
) -> np.ndarray:
    """Express measured body state in the pre-offset VLA coordinate frame.

    The deploy process adds ``offset_deg`` to the final command for the left
    shoulder pitch joint.  The policy was trained without that actuator-side
    adjustment, so subtract the same constant from only the copied VLA input.
    Raw ZMQ feedback and all controller-side state remain unchanged.
    """

    body_q = np.asarray(body_q, dtype=np.float32)
    if body_q.shape[-1:] != (G1_BODY_DIM,):
        raise ValueError(
            f"G1 body state must end in {G1_BODY_DIM} values, got {body_q.shape}"
        )
    if not np.isfinite(body_q).all():
        raise ValueError("G1 body state contains non-finite values")
    if not np.isfinite(offset_deg) or not 0.0 <= offset_deg <= 90.0:
        raise ValueError(
            "left-shoulder back offset must be finite and in [0, 90] degrees, "
            f"got {offset_deg}"
        )

    compensated = body_q.copy()
    compensated[..., G1_LEFT_SHOULDER_PITCH_MUJOCO_INDEX] -= np.float32(
        np.deg2rad(offset_deg)
    )
    return compensated


def apply_brainco_state_deadband(
    hand_wire: np.ndarray,
    deadband: float,
) -> np.ndarray:
    """Suppress small BrainCo feedback noise before policy conditioning.

    This affects measured state only. It must not be used on policy actions or
    final hand commands. Values whose absolute magnitude is strictly below
    ``deadband`` become zero; larger values are preserved exactly.
    """

    hand_wire = np.asarray(hand_wire, dtype=np.float32)
    if hand_wire.shape[-1:] != (G1_BRAINCO_WIRE_DIM,):
        raise ValueError(
            "BrainCo wire state must end in "
            f"{G1_BRAINCO_WIRE_DIM} values, got {hand_wire.shape}"
        )
    if not np.isfinite(hand_wire).all():
        raise ValueError("BrainCo wire state contains non-finite values")
    if not np.isfinite(deadband) or deadband < 0.0:
        raise ValueError(
            f"BrainCo state deadband must be finite and non-negative, got {deadband}"
        )

    filtered = hand_wire.copy()
    filtered[np.abs(filtered) < deadband] = 0.0
    return filtered


def restore_brainco_wire_hand_slots(
    qpos: np.ndarray,
    left_hand_wire: np.ndarray,
    right_hand_wire: np.ndarray,
) -> np.ndarray:
    """Restore the dataset/checkpoint BrainCo order in a physical 43-D qpos.

    SONIC feedback and final hand-command fields publish each hand as
    ``[thumb, thumb_aux, index, middle, ring, pinky, padding]``.  The final
    exported datasets copy those seven values directly into slots ``22:29`` and
    ``36:43``.  In contrast, ``RobotModel`` accepts and then groups hand joints
    in two different URDF orders; passing the wire vector through it permutes
    the policy state.  Replace only the hand slots after RobotModel assembly so
    the live numerical channels exactly match the trained checkpoint contract.

    The returned array is always an owned copy; body values are unchanged.
    """

    qpos = np.asarray(qpos, dtype=np.float32)
    left_hand_wire = np.asarray(left_hand_wire, dtype=np.float32)
    right_hand_wire = np.asarray(right_hand_wire, dtype=np.float32)
    if qpos.shape[-1:] != (G1_FULL_STATE_DIM,):
        raise ValueError(
            f"G1 full state must end in {G1_FULL_STATE_DIM} values, got {qpos.shape}"
        )
    for side, hand in (("left", left_hand_wire), ("right", right_hand_wire)):
        if hand.shape[-1:] != (G1_BRAINCO_WIRE_DIM,):
            raise ValueError(
                f"BrainCo {side} wire state must end in "
                f"{G1_BRAINCO_WIRE_DIM} values, got {hand.shape}"
            )
        if not np.isfinite(hand).all():
            raise ValueError(f"BrainCo {side} wire state contains non-finite values")
    if not np.isfinite(qpos).all():
        raise ValueError("G1 full state contains non-finite values")

    restored = qpos.copy()
    restored[..., G1_LEFT_HAND_STATE_INDICES] = left_hand_wire
    restored[..., G1_RIGHT_HAND_STATE_INDICES] = right_hand_wire
    return restored


def mujoco_absolute_to_isaaclab_offset(body_q: np.ndarray) -> np.ndarray:
    """Convert 29 absolute grouped angles to IsaacLab-order default offsets."""

    body_q = np.asarray(body_q, dtype=np.float32)
    if body_q.shape[-1:] != (G1_BODY_DIM,):
        raise ValueError(
            f"G1 body state must end in {G1_BODY_DIM} values, got {body_q.shape}"
        )
    if not np.isfinite(body_q).all():
        raise ValueError("G1 body state contains non-finite values")

    return (
        body_q[..., G1_MUJOCO_TO_ISAACLAB]
        - G1_DEFAULT_ANGLES_MUJOCO[G1_MUJOCO_TO_ISAACLAB]
    ).astype(np.float32, copy=False)


def physical_qpos_to_isaaclab_offset_state(qpos: np.ndarray) -> np.ndarray:
    """Map live 43-D physical qpos into the training-state representation.

    The returned vector exactly reproduces the representation stored in the
    exported parquet files: the 29 body slots contain IsaacLab-order offsets and
    the 14 hand slots retain the RobotModel/BrainCo values and ordering.
    """

    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape[-1:] != (G1_FULL_STATE_DIM,):
        raise ValueError(
            f"G1 full state must end in {G1_FULL_STATE_DIM} values, got {qpos.shape}"
        )
    if not np.isfinite(qpos).all():
        raise ValueError("G1 full state contains non-finite values")

    converted = qpos.copy()
    body_q_mujoco_absolute = qpos[..., G1_BODY_STATE_INDICES]
    converted[..., G1_BODY_STATE_INDICES] = mujoco_absolute_to_isaaclab_offset(
        body_q_mujoco_absolute
    )
    return converted
