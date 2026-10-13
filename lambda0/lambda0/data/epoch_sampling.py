"""Deterministic no-replacement epoch construction for hierarchical datasets."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleRequest:
    """One sampled episode and optional explicit timestep request."""

    group_name: str
    dataset_name: str
    task_name: str | None
    episode_index: int
    timestep_index: int | None = None


@dataclass(frozen=True)
class EpochEpisode:
    """One episode leaf and its target probability and finite capacity."""

    group_name: str
    dataset_name: str
    task_name: str | None
    episode_index: int
    probability: float
    capacity: int
    candidate_count: int


def _seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    return random.Random(payload).getrandbits(64)


def _allocate_quotas(episodes: list[EpochEpisode]) -> list[int]:
    if not episodes:
        raise ValueError("An epoch requires at least one sampleable episode.")
    probability_sum = sum(episode.probability for episode in episodes)
    probabilities = [
        episode.probability / probability_sum for episode in episodes
    ]
    target_size = int(
        math.floor(
            min(
                episode.capacity / probability
                for episode, probability in zip(
                    episodes,
                    probabilities,
                    strict=True,
                )
            )
            + 1e-9
        )
    )
    if target_size < len(episodes):
        raise ValueError(
            "Sampling weights and episode capacities cannot allocate at least "
            "one timestep to every episode in an epoch."
        )

    exact = [target_size * probability for probability in probabilities]
    quotas = [
        min(episode.capacity, int(math.floor(value)))
        for episode, value in zip(episodes, exact, strict=True)
    ]
    remaining = target_size - sum(quotas)
    order = sorted(
        range(len(episodes)),
        key=lambda index: (exact[index] - math.floor(exact[index]), -index),
        reverse=True,
    )
    while remaining:
        progressed = False
        for index in order:
            if quotas[index] >= episodes[index].capacity:
                continue
            quotas[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("Unable to satisfy epoch sampling quotas.")
    if any(quota <= 0 for quota in quotas):
        raise ValueError(
            "Every episode must contribute at least one timestep per epoch."
        )
    return quotas


def _episode_timesteps(
    episode: EpochEpisode,
    quota: int,
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    size = episode.candidate_count
    permutation_rng = random.Random(
        _seed(seed, "permutation", episode.dataset_name, episode.episode_index)
    )
    if size == 1:
        multiplier, offset = 0, 0
    else:
        multiplier = permutation_rng.randrange(1, size)
        while math.gcd(multiplier, size) != 1:
            multiplier = permutation_rng.randrange(1, size)
        offset = permutation_rng.randrange(size)
    start = (epoch * quota) % size
    timesteps = [
        (multiplier * ((start + position) % size) + offset) % size
        if size > 1
        else 0
        for position in range(quota)
    ]
    random.Random(
        _seed(
            seed,
            "episode-order",
            epoch,
            episode.dataset_name,
            episode.episode_index,
        )
    ).shuffle(timesteps)
    return timesteps


def build_epoch_requests(
    episodes: list[EpochEpisode],
    *,
    seed: int,
    epoch: int,
) -> tuple[SampleRequest, ...]:
    """Build one weighted, deterministic, globally shardable epoch."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative.")
    quotas = _allocate_quotas(episodes)
    requests = []
    for episode, quota in zip(episodes, quotas, strict=True):
        timesteps = _episode_timesteps(
            episode,
            quota,
            seed=seed,
            epoch=epoch,
        )
        requests.extend(
            SampleRequest(
                group_name=episode.group_name,
                dataset_name=episode.dataset_name,
                task_name=episode.task_name,
                episode_index=episode.episode_index,
                timestep_index=timestep,
            )
            for timestep in timesteps
        )
    random.Random(_seed(seed, "global-order", epoch)).shuffle(requests)
    return tuple(requests)


__all__ = ["EpochEpisode", "SampleRequest", "build_epoch_requests"]
