"""Release split/statistics primitives. No training or video decoding."""
import random
import math
import numpy as np


CONTRACT = "lambda0_velocity_v1"


def split_episodes(items, seed=0):
    """Stable episode holdout, independent of incoming file order."""
    identities = [str(item["episode_index"]) for item in items]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate episode identities.")
    order = sorted(range(len(items)), key=lambda i: identities[i])
    random.Random(seed).shuffle(order)
    count = min(len(items) - 1, max(1, math.ceil(len(items) * .01))) if len(items) > 1 else 0
    holdout = set(order[:count])
    return ([item for i, item in enumerate(items) if i not in holdout],
            [item for i, item in enumerate(items) if i in holdout])


def quantile_parameters(values, valid):
    values, valid = np.asarray(values), np.asarray(valid, dtype=bool)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("Statistics require equally shaped [N,D] values/masks.")
    if not np.isfinite(values[valid]).all():
        raise ValueError("Nonfinite valid labels.")
    q01, q99 = [], []
    for column in range(values.shape[1]):
        selected = values[valid[:, column], column]
        if not len(selected):
            raise ValueError(f"No training values for dimension {column}.")
        low, high = np.quantile(selected, [.01, .99])
        q01.append(float(low)); q99.append(float(high))
    low, high = np.asarray(q01), np.asarray(q99)
    # Exactly equivalent to 2*(x-q01)/(q99-q01+1e-6)-1.
    scale = (high - low + 1e-6) / 2
    return {"q01": q01, "q99": q99, "location": (low + scale).tolist(), "scale": scale.tolist()}


def valid_observations(state_mask, action_mask, horizon=50):
    length = len(state_mask)
    result = []
    for i in range(length):
        future = np.minimum(np.arange(i + 1, i + horizon + 1), length - 1)
        if state_mask[i].any() and action_mask[future].any():
            result.append(i)
    return result
