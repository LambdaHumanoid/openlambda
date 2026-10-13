from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


CONTINUOUS_ACTION_EXPERT = "continuous_action_expert"
DISCRETE_VLM = "discrete_vlm"
STATE_CONDITIONING_MODES = (
    CONTINUOUS_ACTION_EXPERT,
    DISCRETE_VLM,
)
DEFAULT_STATE_DISCRETE_BINS = 256


def validate_state_conditioning(
    mode: str,
    num_bins: int = DEFAULT_STATE_DISCRETE_BINS,
) -> tuple[str, int]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in STATE_CONDITIONING_MODES:
        raise ValueError(
            f"Unsupported state conditioning mode {mode!r}; expected one of "
            f"{STATE_CONDITIONING_MODES}."
        )
    normalized_bins = int(num_bins)
    if normalized_bins != num_bins or normalized_bins < 2:
        raise ValueError("state_discrete_bins must be an integer of at least 2.")
    return normalized_mode, normalized_bins


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def discretize_normalized_state(
    proprio: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    num_bins: int = DEFAULT_STATE_DISCRETE_BINS,
) -> list[list[int]]:
    _, normalized_bins = validate_state_conditioning(DISCRETE_VLM, num_bins)
    values = np.asarray(_as_numpy(proprio), dtype=np.float32)
    mask = np.asarray(_as_numpy(valid_mask), dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError(
            "proprio and valid_mask must have the same [B,P] shape for "
            "discrete VLM conditioning."
        )
    selected = values[mask]
    if not np.isfinite(selected).all():
        raise ValueError("Valid proprio values must be finite.")
    result: list[list[int]] = []
    for row, row_mask in zip(values, mask, strict=True):
        indices = np.digitize(
            row[row_mask], np.linspace(-1, 1, normalized_bins + 1)[:-1]
        ) - 1
        result.append(indices.tolist())
    return result


def inject_discrete_state_into_instructions(
    instructions: Sequence[str],
    proprio: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    num_bins: int = DEFAULT_STATE_DISCRETE_BINS,
) -> list[str]:
    state_bins = discretize_normalized_state(
        proprio,
        valid_mask,
        num_bins=num_bins,
    )
    if len(instructions) != len(state_bins):
        raise ValueError("Instruction and proprio batch sizes must match.")
    conditioned: list[str] = []
    for instruction, bins in zip(instructions, state_bins, strict=True):
        task = str(instruction).strip().replace("\n", " ")
        if not task:
            raise ValueError("Language instruction must not be empty.")
        if not bins:
            raise ValueError("Every sample must have at least one valid state value.")
        state = " ".join(str(value) for value in bins)
        conditioned.append(f"Task: {task}, State: {state};\nAction:")
    return conditioned


__all__ = [
    "CONTINUOUS_ACTION_EXPERT",
    "DEFAULT_STATE_DISCRETE_BINS",
    "DISCRETE_VLM",
    "STATE_CONDITIONING_MODES",
    "discretize_normalized_state",
    "inject_discrete_state_into_instructions",
    "validate_state_conditioning",
]
