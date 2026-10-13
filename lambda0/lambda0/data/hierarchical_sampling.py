"""Hierarchical sampling across action spaces, datasets, tasks, and episodes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lambda0.models.action_specs import get_action_space_spec

from .epoch_sampling import EpochEpisode, SampleRequest, build_epoch_requests


@dataclass(frozen=True)
class _TaskPool:
    name: str | None
    episode_indices: Sequence[int]


@dataclass(frozen=True)
class _DatasetPool:
    name: str
    weight: float
    tasks: tuple[_TaskPool, ...]
    task_weights: tuple[float, ...]


@dataclass(frozen=True)
class _GroupPool:
    name: str
    weight: float
    datasets: tuple[_DatasetPool, ...]
    timestep_mode: str
    samples_per_episode_per_epoch: int | None


def _positive_number(value: Any, context: str) -> float:
    result = float(value)
    if not result > 0:
        raise ValueError(f"{context} must be positive, got {value!r}.")
    return result


def _require_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    missing = expected.difference(value)
    extra = set(value).difference(expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if extra:
            details.append("extra=" + ",".join(sorted(extra)))
        raise KeyError(f"{context} has invalid keys: {'; '.join(details)}")


def _task_value(item: Mapping[str, Any], key: str, context: str) -> str:
    if key not in item:
        raise KeyError(f"{context} does not contain task key {key!r}.")
    value = item[key]
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                f"{context}.{key} must contain exactly one task, got {value!r}."
            )
        value = value[0]
    task = str(value).strip()
    if not task:
        raise ValueError(f"{context}.{key} must not be empty.")
    return task


class HierarchicalSampler:
    """Compile and sample a strict group/dataset/task/episode hierarchy."""

    def __init__(
        self,
        metas: Mapping[str, dict[str, Any]],
        plan: Mapping[str, Any] | None,
    ) -> None:
        if not metas:
            raise ValueError("At least one dataset is required for sampling.")
        if plan is None:
            plan = self._default_plan(metas)
        self.strategy = str(plan.get("strategy", ""))
        self.frame_leaves = []
        if self.strategy == "frame_uniform":
            if set(plan) != {"strategy"}:
                raise ValueError("frame_uniform does not accept mixture weights.")
            for name, meta in metas.items():
                for index, item in enumerate(meta["_episodes"]):
                    valid = item.get("valid_sample_indices")
                    if valid is None:
                        raise ValueError("frame_uniform requires prepared valid_sample_indices.")
                    if valid:
                        self.frame_leaves.append((name, index, valid))
            if not self.frame_leaves:
                raise ValueError("No valid observation frames.")
            self.frame_weights = [len(item[2]) for item in self.frame_leaves]
            self.groups = ()
            self.group_weights = ()
            return
        self.groups = self._compile(metas, plan)
        self.group_weights = tuple(group.weight for group in self.groups)

    @property
    def uses_epoch_sampling(self) -> bool:
        return self.strategy == "hierarchical_epoch"

    @staticmethod
    def _default_plan(
        metas: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any]:
        grouped: dict[str, dict[str, float]] = {}
        for name, meta in metas.items():
            action_space = str(meta["action_space"])
            grouped.setdefault(action_space, {})[name] = float(
                meta["sample_weight"]
            )
        groups = []
        for action_space, datasets in grouped.items():
            groups.append(
                {
                    "name": action_space,
                    "action_space": action_space,
                    "weight": sum(datasets.values()),
                    "datasets": {
                        "mode": "explicit",
                        "weights": datasets,
                    },
                    "task": {"mode": "none", "key": None},
                    "episode": {"mode": "uniform"},
                    "timestep": {"mode": "uniform"},
                }
            )
        return {"strategy": "hierarchical_iid", "groups": groups}

    def _compile(
        self,
        metas: Mapping[str, dict[str, Any]],
        plan: Mapping[str, Any],
    ) -> tuple[_GroupPool, ...]:
        _require_keys(plan, {"strategy", "groups"}, "sampling plan")
        if plan["strategy"] not in {"hierarchical_iid", "hierarchical_epoch"}:
            raise ValueError(
                "sampling.strategy must be 'hierarchical_iid' or "
                "'hierarchical_epoch'."
            )
        raw_groups = plan["groups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError("sampling.groups must be a non-empty list.")

        loaded = set(metas)
        assigned: set[str] = set()
        group_names: set[str] = set()
        groups = []
        for group_index, raw_group in enumerate(raw_groups):
            context = f"sampling.groups[{group_index}]"
            if not isinstance(raw_group, Mapping):
                raise TypeError(f"{context} must be an object.")
            _require_keys(
                raw_group,
                {
                    "name",
                    "action_space",
                    "weight",
                    "datasets",
                    "task",
                    "episode",
                    "timestep",
                },
                context,
            )
            group_name = str(raw_group["name"]).strip()
            if not group_name or group_name in group_names:
                raise ValueError(f"{context}.name must be non-empty and unique.")
            group_names.add(group_name)
            group_weight = _positive_number(raw_group["weight"], f"{context}.weight")
            action_space = get_action_space_spec(raw_group["action_space"]).name

            if raw_group["episode"] != {"mode": "uniform"}:
                raise ValueError(f"{context}.episode must be {{'mode': 'uniform'}}.")
            timestep_config = raw_group["timestep"]
            if not isinstance(timestep_config, Mapping):
                raise TypeError(f"{context}.timestep must be an object.")
            if plan["strategy"] == "hierarchical_iid":
                if timestep_config != {"mode": "uniform"}:
                    raise ValueError(
                        f"{context}.timestep must be {{'mode': 'uniform'}} for "
                        "hierarchical_iid."
                    )
                timestep_mode = "uniform"
                samples_per_episode = None
            else:
                _require_keys(
                    timestep_config,
                    {"mode", "samples_per_episode_per_epoch"},
                    f"{context}.timestep",
                )
                timestep_mode = str(timestep_config["mode"])
                if timestep_mode != "shuffled_without_replacement":
                    raise ValueError(
                        f"{context}.timestep.mode must be "
                        "'shuffled_without_replacement' for hierarchical_epoch."
                    )
                samples_per_episode = int(
                    timestep_config["samples_per_episode_per_epoch"]
                )
                if samples_per_episode <= 0:
                    raise ValueError(
                        f"{context}.timestep.samples_per_episode_per_epoch must "
                        "be positive."
                    )

            datasets_config = raw_group["datasets"]
            if not isinstance(datasets_config, Mapping):
                raise TypeError(f"{context}.datasets must be an object.")
            _require_keys(
                datasets_config,
                {"mode", "weights"},
                f"{context}.datasets",
            )
            if datasets_config["mode"] != "explicit":
                raise ValueError(f"{context}.datasets.mode must be 'explicit'.")
            dataset_weights = datasets_config["weights"]
            if not isinstance(dataset_weights, Mapping) or not dataset_weights:
                raise ValueError(f"{context}.datasets.weights must not be empty.")

            pools = []
            for dataset_name, raw_weight in dataset_weights.items():
                dataset_name = str(dataset_name)
                if dataset_name not in loaded:
                    raise KeyError(f"Sampling references unloaded dataset {dataset_name!r}.")
                if dataset_name in assigned:
                    raise ValueError(
                        f"Dataset {dataset_name!r} belongs to more than one sampling group."
                    )
                meta = metas[dataset_name]
                actual_space = get_action_space_spec(meta["action_space"]).name
                if actual_space != action_space:
                    raise ValueError(
                        f"Dataset {dataset_name!r} uses {actual_space!r}, but "
                        f"group {group_name!r} declares {action_space!r}."
                    )
                tasks, task_weights = self._compile_tasks(
                    dataset_name,
                    meta,
                    raw_group["task"],
                )
                pools.append(
                    _DatasetPool(
                        name=dataset_name,
                        weight=_positive_number(
                            raw_weight,
                            f"{context}.datasets.weights.{dataset_name}",
                        ),
                        tasks=tasks,
                        task_weights=task_weights,
                    )
                )
                assigned.add(dataset_name)
            groups.append(
                _GroupPool(
                    name=group_name,
                    weight=group_weight,
                    datasets=tuple(pools),
                    timestep_mode=timestep_mode,
                    samples_per_episode_per_epoch=samples_per_episode,
                )
            )

        missing = loaded.difference(assigned)
        if missing:
            raise ValueError(
                "Every loaded dataset must belong to one sampling group; missing: "
                + ", ".join(sorted(missing))
            )
        return tuple(groups)

    @staticmethod
    def _compile_tasks(
        dataset_name: str,
        meta: Mapping[str, Any],
        task_config: Any,
    ) -> tuple[tuple[_TaskPool, ...], tuple[float, ...]]:
        context = f"sampling task config for {dataset_name!r}"
        if not isinstance(task_config, Mapping):
            raise TypeError(f"{context} must be an object.")
        _require_keys(task_config, {"mode", "key"}, context)
        mode = task_config["mode"]
        key = task_config["key"]
        episodes = meta["_episodes"]
        if mode == "none":
            if key is not None:
                raise ValueError(f"{context}.key must be null when mode='none'.")
            return (_TaskPool(None, range(len(episodes))),), (1.0,)
        if mode != "uniform":
            raise ValueError(f"{context}.mode must be 'none' or 'uniform'.")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context}.key must be a non-empty string.")
        if not isinstance(episodes, list):
            raise ValueError(
                f"Task-balanced dataset {dataset_name!r} must use an inline datalist."
            )

        buckets: dict[str, list[int]] = {}
        for index, item in enumerate(episodes):
            task = _task_value(item, key, f"{dataset_name}[{index}]")
            buckets.setdefault(task, []).append(index)
        tasks = tuple(
            _TaskPool(name, tuple(indices))
            for name, indices in sorted(buckets.items())
        )
        return tasks, tuple(1.0 for _ in tasks)

    def sample(self, rng: random.Random | Any = random) -> SampleRequest:
        if self.strategy == "frame_uniform":
            name, index, valid = rng.choices(self.frame_leaves, weights=self.frame_weights, k=1)[0]
            return SampleRequest("frame_uniform", name, None, index, rng.choice(valid))
        if self.uses_epoch_sampling:
            raise RuntimeError(
                "hierarchical_epoch requires epoch_requests(), not sample()."
            )
        group = rng.choices(self.groups, weights=self.group_weights, k=1)[0]
        dataset = rng.choices(
            group.datasets,
            weights=[item.weight for item in group.datasets],
            k=1,
        )[0]
        task = rng.choices(dataset.tasks, weights=dataset.task_weights, k=1)[0]
        episode_index = rng.choice(task.episode_indices)
        return SampleRequest(
            group_name=group.name,
            dataset_name=dataset.name,
            task_name=task.name,
            episode_index=int(episode_index),
        )

    def _epoch_episodes(self, candidate_count) -> list[EpochEpisode]:
        group_total = sum(group.weight for group in self.groups)
        episodes: list[EpochEpisode] = []
        for group in self.groups:
            if group.samples_per_episode_per_epoch is None:
                raise RuntimeError("Epoch sampling group has no episode sample budget.")
            group_probability = group.weight / group_total
            dataset_total = sum(dataset.weight for dataset in group.datasets)
            for dataset in group.datasets:
                dataset_probability = dataset.weight / dataset_total
                task_total = sum(dataset.task_weights)
                for task, task_weight in zip(
                    dataset.tasks, dataset.task_weights, strict=True
                ):
                    episode_counts = []
                    for episode_index in task.episode_indices:
                        count = int(candidate_count(dataset.name, int(episode_index)))
                        if count > 0:
                            episode_counts.append((int(episode_index), count))
                    if not episode_counts:
                        raise ValueError(
                            f"Task {task.name!r} in dataset {dataset.name!r} has no "
                            "sampleable episodes."
                        )
                    task_probability = task_weight / task_total
                    episode_probability = 1.0 / len(episode_counts)
                    for episode_index, count in episode_counts:
                        episodes.append(
                            EpochEpisode(
                                group_name=group.name,
                                dataset_name=dataset.name,
                                task_name=task.name,
                                episode_index=episode_index,
                                probability=(
                                    group_probability
                                    * dataset_probability
                                    * task_probability
                                    * episode_probability
                                ),
                                capacity=min(
                                    group.samples_per_episode_per_epoch,
                                    count,
                                ),
                                candidate_count=count,
                            )
                        )
        return episodes

    def epoch_requests(
        self,
        candidate_count,
        *,
        seed: int,
        epoch: int,
    ) -> tuple[SampleRequest, ...]:
        """Build one deterministic, globally shardable no-replacement epoch."""
        if not self.uses_epoch_sampling:
            raise RuntimeError("epoch_requests() requires hierarchical_epoch.")
        return build_epoch_requests(
            self._epoch_episodes(candidate_count),
            seed=seed,
            epoch=epoch,
        )


__all__ = ["HierarchicalSampler", "SampleRequest"]
