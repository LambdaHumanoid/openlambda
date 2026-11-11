"""Hand/arm layout for the original AMO-based SIMPLE tasks."""

import numpy as np

# SIMPLE qpos -> policy: thumb, middle, index, right hand, left/right arm.
STATE_SLICES = ((29, 32), (34, 36), (32, 34), (36, 43), (15, 22), (22, 29))


def upper_state(joint_qpos):
    return np.concatenate([joint_qpos[start:end] for start, end in STATE_SLICES])


def upper_joint_targets(joint_names, action):
    # Policy -> SIMPLE: arms, left thumb/index/middle, right hand.
    joints = np.concatenate((action[14:28], action[0:3], action[5:7], action[3:5], action[7:14]))
    return dict(zip(joint_names[15:], joints, strict=True))


def waist_joint_targets(action):
    return {
        "waist_yaw_joint": action[30],
        "waist_roll_joint": action[28],
        "waist_pitch_joint": action[29],
    }


def require_instruction(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("SIMPLE evaluation requires a non-empty task instruction.")


def validate_action_chunk(actions):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 36:
        raise ValueError(f"Expected a non-empty (T, 36) action chunk, received {actions.shape}.")
    if actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
        raise ValueError("The action chunk must contain only finite real numbers.")
    return actions
