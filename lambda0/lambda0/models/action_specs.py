"""Canonical action/proprioception contracts for heterogeneous VLA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple, Union


SONIC_MOTION_DIM = 64
WRIST_POSITION_DIM = 3
WRIST_ROTATION_6D_DIM = 6
FINGER_JOINT_COUNT = 20
FINGER_GEOMETRY_DIM = FINGER_JOINT_COUNT * 3
REVO2_HAND_DIM = 6
SIMPLE_G1_HAND_DIM = 7
SIMPLE_G1_ARM_DIM = 7
SIMPLE_G1_TORSO_POSE_DIM = 4
SIMPLE_G1_NAVIGATION_DIM = 4


@dataclass(frozen=True)
class ActionBlockSpec:
    """A named contiguous slice of an action vector."""

    name: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Action block name must not be empty.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid action block range: {self.start}:{self.end}.")

    @property
    def width(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ActionSpaceSpec:
    """Runtime contract for one embodiment/action representation."""

    id: int
    name: str
    action_dim: int
    proprio_dim: int
    blocks: Tuple[ActionBlockSpec, ...]

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("Action-space id must be non-negative.")
        if not self.name:
            raise ValueError("Action-space name must not be empty.")
        if self.action_dim <= 0 or self.proprio_dim < 0:
            raise ValueError("Action/proprio dimensions are invalid.")
        previous_end = 0
        for block in self.blocks:
            if block.start != previous_end or block.end > self.action_dim:
                raise ValueError(
                    f"Blocks for {self.name} must be contiguous, sorted, and fit "
                    f"action_dim={self.action_dim}; expected start {previous_end}, "
                    f"got {block.start}."
                )
            previous_end = block.end
        if previous_end != self.action_dim:
            raise ValueError(
                f"Blocks for {self.name} end at {previous_end}, "
                f"but action_dim is {self.action_dim}."
            )

    @property
    def block_names(self) -> Tuple[str, ...]:
        return tuple(block.name for block in self.blocks)


def _block(name: str, start: int, end: int) -> ActionBlockSpec:
    return ActionBlockSpec(name=name, start=start, end=end)


# These are the only production contracts used by heterogeneous training.
HUMAN_HAND_GEOMETRY = ActionSpaceSpec(
    id=0,
    name="human_hand_geometry",
    action_dim=138,
    proprio_dim=138,
    blocks=(
        _block("left_wrist_pos", 0, 3),
        _block("left_wrist_rot6d", 3, 9),
        _block("left_finger_geometry", 9, 69),
        _block("right_wrist_pos", 69, 72),
        _block("right_wrist_rot6d", 72, 78),
        _block("right_finger_geometry", 78, 138),
    ),
)

HUMAN_FULLBODY_SONIC_GEOMETRY = ActionSpaceSpec(
    id=1,
    name="human_fullbody_sonic_geometry",
    action_dim=184,
    proprio_dim=120,
    blocks=(
        _block("sonic_motion", 0, 64),
        _block("left_finger_geometry", 64, 124),
        _block("right_finger_geometry", 124, 184),
    ),
)

HUMAN_FULLBODY_SONIC_HAND_STATE = ActionSpaceSpec(
    id=4,
    name="human_fullbody_sonic_hand_state",
    action_dim=184,
    proprio_dim=138,
    blocks=(
        _block("sonic_motion", 0, 64),
        _block("left_finger_geometry", 64, 124),
        _block("right_finger_geometry", 124, 184),
    ),
)

ROBOT_SONIC_REVO2 = ActionSpaceSpec(
    id=2,
    name="robot_sonic_revo2",
    action_dim=76,
    proprio_dim=44,
    blocks=(
        _block("sonic_motion", 0, 64),
        _block("left_revo2", 64, 70),
        _block("right_revo2", 70, 76),
    ),
)

ROBOT_G1_SIMPLE_SIM = ActionSpaceSpec(
    id=3,
    name="robot_g1_simple_sim",
    action_dim=36,
    proprio_dim=32,
    blocks=(
        _block("left_hand", 0, 7),
        _block("right_hand", 7, 14),
        _block("left_arm", 14, 21),
        _block("right_arm", 21, 28),
        _block("torso_pose", 28, 32),
        _block("torso_navigation", 32, 36),
    ),
)


ACTION_SPACE_SPECS: Dict[int, ActionSpaceSpec] = {
    spec.id: spec
    for spec in (
        HUMAN_HAND_GEOMETRY,
        HUMAN_FULLBODY_SONIC_GEOMETRY,
        ROBOT_SONIC_REVO2,
        ROBOT_G1_SIMPLE_SIM,
        HUMAN_FULLBODY_SONIC_HAND_STATE,
    )
}
ACTION_SPACE_NAME_TO_ID: Dict[str, int] = {
    spec.name: spec.id for spec in ACTION_SPACE_SPECS.values()
}

MAX_ACTION_DIM = max(spec.action_dim for spec in ACTION_SPACE_SPECS.values())
MAX_PROPRIO_DIM = max(spec.proprio_dim for spec in ACTION_SPACE_SPECS.values())


def get_action_space_spec(value: Union[int, str, ActionSpaceSpec]) -> ActionSpaceSpec:
    if isinstance(value, ActionSpaceSpec):
        return value
    if isinstance(value, str):
        key = value.lower()
        if key not in ACTION_SPACE_NAME_TO_ID:
            raise KeyError(f"Unknown action-space name: {value}.")
        return ACTION_SPACE_SPECS[ACTION_SPACE_NAME_TO_ID[key]]
    if isinstance(value, int) and value in ACTION_SPACE_SPECS:
        return ACTION_SPACE_SPECS[value]
    raise KeyError(f"Unknown action-space id: {value}.")


def action_space_config(
    specs: Mapping[int, ActionSpaceSpec] | None = None,
) -> Tuple[Mapping[str, object], ...]:
    """Return JSON-serializable config entries for model checkpoints."""
    runtime_specs = specs or ACTION_SPACE_SPECS
    return tuple(
        {
            "id": spec.id,
            "name": spec.name,
            "action_dim": spec.action_dim,
            "proprio_dim": spec.proprio_dim,
            "blocks": [
                {"name": block.name, "start": block.start, "end": block.end}
                for block in spec.blocks
            ],
        }
        for _, spec in sorted(runtime_specs.items())
    )


def action_space_specs_from_config(
    entries: Sequence[Mapping[str, object]] | None,
) -> Dict[int, ActionSpaceSpec]:
    """Build validated runtime specs from a checkpoint configuration."""
    if entries is None:
        return dict(ACTION_SPACE_SPECS)
    specs: Dict[int, ActionSpaceSpec] = {}
    names: Dict[str, int] = {}
    for entry in entries:
        blocks_value = entry.get("blocks")
        if not isinstance(blocks_value, Sequence) or isinstance(
            blocks_value, (str, bytes)
        ):
            raise TypeError("Each action-space config entry must contain a blocks sequence.")
        blocks = tuple(
            ActionBlockSpec(
                name=str(block["name"]),
                start=int(block["start"]),
                end=int(block["end"]),
            )
            for block in blocks_value
        )
        spec = ActionSpaceSpec(
            id=int(entry["id"]),
            name=str(entry["name"]).lower(),
            action_dim=int(entry["action_dim"]),
            proprio_dim=int(entry["proprio_dim"]),
            blocks=blocks,
        )
        if spec.id in specs:
            raise ValueError(f"Duplicate action-space id {spec.id}.")
        if spec.name in names:
            raise ValueError(
                f"Action-space name '{spec.name}' is shared by ids "
                f"{names[spec.name]} and {spec.id}."
            )
        names[spec.name] = spec.id
        specs[spec.id] = spec
    if not specs:
        raise ValueError("At least one action-space config entry is required.")
    return specs


__all__ = [
    "ActionBlockSpec",
    "ActionSpaceSpec",
    "ACTION_SPACE_SPECS",
    "ACTION_SPACE_NAME_TO_ID",
    "MAX_ACTION_DIM",
    "MAX_PROPRIO_DIM",
    "SONIC_MOTION_DIM",
    "WRIST_POSITION_DIM",
    "WRIST_ROTATION_6D_DIM",
    "FINGER_JOINT_COUNT",
    "FINGER_GEOMETRY_DIM",
    "REVO2_HAND_DIM",
    "SIMPLE_G1_HAND_DIM",
    "SIMPLE_G1_ARM_DIM",
    "SIMPLE_G1_TORSO_POSE_DIM",
    "SIMPLE_G1_NAVIGATION_DIM",
    "HUMAN_HAND_GEOMETRY",
    "HUMAN_FULLBODY_SONIC_GEOMETRY",
    "HUMAN_FULLBODY_SONIC_HAND_STATE",
    "ROBOT_SONIC_REVO2",
    "ROBOT_G1_SIMPLE_SIM",
    "get_action_space_spec",
    "action_space_config",
    "action_space_specs_from_config",
]
