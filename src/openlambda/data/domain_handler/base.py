"""Dataset adapter interface for canonical Lambda0 samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class DomainHandler(ABC):
    """Converts one source episode into canonical heterogeneous samples."""

    def __init__(self, meta: dict[str, Any], num_views: int) -> None:
        self.meta = meta
        self.num_views = num_views

    @abstractmethod
    def iter_episode(
        self,
        episode_index: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict[str, list[str]] | None,
    ) -> Iterable[dict[str, Any]]:
        """Yield samples with the fields consumed by HeterogeneousActionCollator."""
        raise NotImplementedError

    @abstractmethod
    def candidate_timestep_count(
        self,
        episode_index: int,
        *,
        num_actions: int,
    ) -> int:
        """Return the contiguous [0,count) training timestep range."""
        raise NotImplementedError

    @abstractmethod
    def sample_episode(
        self,
        episode_index: int,
        *,
        num_actions: int,
        image_aug,
        lang_aug_map: dict[str, list[str]] | None,
        sample_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Sample one timestep, honoring an explicit scheduled index if provided."""
        raise NotImplementedError

    def iter_validation_episode(
        self,
        episode_index: int,
        *,
        num_actions: int,
        samples_per_episode: int,
        image_aug,
        lang_aug_map: dict[str, list[str]] | None,
    ) -> Iterable[dict[str, Any]]:
        """Yield a deterministic, bounded validation view of one episode."""
        if samples_per_episode <= 0:
            raise ValueError("samples_per_episode must be positive.")
        for sample_index, sample in enumerate(
            self.iter_episode(
                episode_index,
                num_actions=num_actions,
                training=False,
                image_aug=image_aug,
                lang_aug_map=lang_aug_map,
            )
        ):
            if sample_index >= samples_per_episode:
                break
            yield sample


__all__ = ["DomainHandler"]
