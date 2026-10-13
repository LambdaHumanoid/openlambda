"""Velocity-flow training primitives, shared by tests and the policy."""
import torch


def sample_flow_conditions(valid_mask, max_delay, no_prefix_probability=0.5, generator=None):
    if not 0 <= no_prefix_probability <= 1:
        raise ValueError("RTC no-prefix probability must be in [0,1].")
    batch, horizon, _ = valid_mask.shape
    if not 0 <= max_delay < horizon:
        raise ValueError("RTC delay must be smaller than the horizon.")
    valid = valid_mask.any(-1)
    if not bool(valid.any(-1).all()):
        raise ValueError("Every sample needs valid supervision.")
    def uniform():
        return torch.rand(batch, device=valid_mask.device, generator=generator)
    # Inverse CDF of Beta(1.5, 1), with a seedable Generator.
    time = uniform().pow(2 / 3) * 0.999 + 0.001
    positions = torch.arange(horizon, device=valid_mask.device)[None, :]
    last_valid = positions.expand(batch, -1).masked_fill(~valid, -1).max(-1).values
    maximum = last_valid.clamp(max=max_delay)
    delay = (uniform() * maximum).long() + 1
    delay = torch.where((uniform() >= no_prefix_probability) & (maximum > 0), delay, 0)
    prefix = (positions < delay[:, None]) & valid
    return time, prefix
