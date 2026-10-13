"""Pi0.5-style scalar binning for the Stage-I hand-state language task."""

from __future__ import annotations

import numpy as np


HAND_STATE_DIM = 138
HAND_STATE_HORIZON = 50
STATE_DISCRETE_BINS = 256


def _bins(values: np.ndarray, valid: np.ndarray, num_bins: int) -> np.ndarray:
    if values.shape != valid.shape or values.shape[-1] != HAND_STATE_DIM:
        raise ValueError("Hand states and masks must have matching 138D shapes.")
    if num_bins != STATE_DISCRETE_BINS:
        raise ValueError("Stage I uses the pi0.5 256-bin state discretization.")
    if not np.isfinite(values[valid]).all():
        raise ValueError("Valid hand-state coordinates must be finite.")
    # Match openpi's np.digitize rule, including values outside [-1, 1].
    return np.digitize(values, np.linspace(-1, 1, num_bins + 1)[:-1]) - 1


def hand_state_text(
    current: np.ndarray,
    current_valid: np.ndarray,
    future: np.ndarray,
    future_valid: np.ndarray,
    *,
    num_bins: int = STATE_DISCRETE_BINS,
) -> tuple[str, str, int]:
    """Return current-state text, future target text, and valid scalar count.

    Future frames are tagged, and each frame lists coordinates in the fixed
    138D order. Missing coordinates use '_', never a fabricated numeric value.
    """
    current = np.asarray(current, dtype=np.float32)
    current_valid = np.asarray(current_valid, dtype=bool)
    future = np.asarray(future, dtype=np.float32)
    future_valid = np.asarray(future_valid, dtype=bool)
    if current.shape != (HAND_STATE_DIM,) or current_valid.shape != current.shape:
        raise ValueError("Current hand state must have 138 values and masks.")
    if future.shape != (HAND_STATE_HORIZON, HAND_STATE_DIM) or future_valid.shape != future.shape:
        raise ValueError("Future hand states must have shape [50,138] and matching masks.")
    current_bins = _bins(current, current_valid, num_bins)
    future_bins = _bins(future, future_valid, num_bins)
    if not current_valid.any() or not future_valid.any():
        raise ValueError("Stage-I samples require valid current and future hand-state coordinates.")
    current_text = " ".join(
        str(int(value)) if valid else "_"
        for value, valid in zip(current_bins, current_valid, strict=True)
    )
    lines = []
    for frame_index, (values, valid) in enumerate(zip(future_bins, future_valid, strict=True), start=1):
        coordinates = " ".join(
            str(int(value)) if present else "_"
            for value, present in zip(values, valid, strict=True)
        )
        lines.append(f"t={frame_index}: {coordinates}")
    return current_text, "\n".join(lines), int(future_valid.sum())


__all__ = ["HAND_STATE_DIM", "HAND_STATE_HORIZON", "hand_state_text"]
