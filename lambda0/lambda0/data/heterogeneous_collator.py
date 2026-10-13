"""Batch collation for samples with different action/proprio dimensions."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from lambda0.models.action_specs import MAX_ACTION_DIM, MAX_PROPRIO_DIM, get_action_space_spec


class HeterogeneousActionCollator:
    """Pads action/proprio tensors and emits explicit validity masks."""

    def __init__(
        self,
        max_action_dim: int = MAX_ACTION_DIM,
        max_proprio_dim: int = MAX_PROPRIO_DIM,
    ):
        self.max_action_dim = int(max_action_dim)
        self.max_proprio_dim = int(max_proprio_dim)
        if self.max_action_dim <= 0 or self.max_proprio_dim <= 0:
            raise ValueError("Padded action/proprio dimensions must be positive.")

    def _space_id(self, sample: Dict[str, Any]) -> int:
        if "action_space_id" not in sample:
            raise KeyError("Every sample must provide action_space_id.")
        value = sample["action_space_id"]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("action_space_id must be scalar per sample.")
            value = int(value.item())
        return int(value)

    @staticmethod
    def _scalar_id(sample: Dict[str, Any], key: str) -> int:
        if key not in sample:
            raise KeyError(f"Every sample must provide {key}.")
        value = sample[key]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{key} must be scalar per sample.")
            value = value.item()
        return int(value)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch.")
        action_space_ids = torch.tensor(
            [self._space_id(sample) for sample in samples],
            dtype=torch.long,
        )
        specs = [
            get_action_space_spec(int(value))
            for value in action_space_ids.tolist()
        ]

        actions: List[torch.Tensor] = []
        proprios: List[torch.Tensor] = []
        action_masks: List[torch.Tensor] = []
        proprio_masks: List[torch.Tensor] = []
        horizon: int | None = None
        for sample, spec in zip(samples, specs, strict=True):
            action = torch.as_tensor(sample["action"], dtype=torch.float32)
            proprio = torch.as_tensor(
                sample["proprio"], dtype=torch.float32
            ).reshape(-1)
            if action.ndim != 2:
                raise ValueError("Each action must have shape [T,D].")
            if horizon is None:
                horizon = action.shape[0]
            elif action.shape[0] != horizon:
                raise ValueError(
                    "All actions in a batch must use the same horizon."
                )
            if spec.action_dim > self.max_action_dim:
                raise ValueError(
                    f"Configured max_action_dim {self.max_action_dim} cannot hold "
                    f"{spec.name} width {spec.action_dim}."
                )
            if spec.proprio_dim > self.max_proprio_dim:
                raise ValueError(
                    f"Configured max_proprio_dim {self.max_proprio_dim} cannot hold "
                    f"{spec.name} width {spec.proprio_dim}."
                )
            if (
                not torch.isfinite(action).all()
                or not torch.isfinite(proprio).all()
            ):
                raise ValueError("Action and proprio tensors must contain only finite values.")
            if action.shape[-1] not in (spec.action_dim, self.max_action_dim):
                raise ValueError(
                    f"Action width {action.shape[-1]} does not match "
                    f"{spec.name} width {spec.action_dim}."
                )
            if proprio.shape[-1] not in (spec.proprio_dim, self.max_proprio_dim):
                raise ValueError(
                    f"Proprio width {proprio.shape[-1]} does not match "
                    f"{spec.name} width {spec.proprio_dim}."
                )
            if (
                action.shape[-1] == self.max_action_dim
                and spec.action_dim < self.max_action_dim
            ):
                if bool(action[:, spec.action_dim :].count_nonzero()):
                    raise ValueError(
                        f"{spec.name} contains non-zero values outside its "
                        f"{spec.action_dim} action channels."
                    )
            if (
                proprio.shape[-1] == self.max_proprio_dim
                and spec.proprio_dim < self.max_proprio_dim
            ):
                if bool(proprio[spec.proprio_dim :].count_nonzero()):
                    raise ValueError(
                        f"{spec.name} contains non-zero values outside its "
                        f"{spec.proprio_dim} proprio channels."
                    )
            action_pad = action.new_zeros((action.shape[0], self.max_action_dim))
            action_pad[:, : action.shape[-1]] = action
            proprio_pad = proprio.new_zeros((self.max_proprio_dim,))
            proprio_pad[: proprio.shape[-1]] = proprio
            action_mask = torch.zeros_like(action_pad, dtype=torch.bool)
            action_mask[:, : spec.action_dim] = True
            if "action_valid_mask" in sample:
                provided = torch.as_tensor(
                    sample["action_valid_mask"], dtype=torch.bool
                )
                if provided.shape != (action.shape[0], spec.action_dim):
                    raise ValueError(
                        f"action_valid_mask must be [{action.shape[0]},{spec.action_dim}]."
                    )
                action_mask[:, : spec.action_dim] = provided
            proprio_mask = torch.zeros_like(proprio_pad, dtype=torch.bool)
            proprio_mask[: spec.proprio_dim] = True
            if "proprio_valid_mask" in sample:
                provided = torch.as_tensor(
                    sample["proprio_valid_mask"], dtype=torch.bool
                ).reshape(-1)
                if provided.shape != (spec.proprio_dim,):
                    raise ValueError(
                        f"proprio_valid_mask must be [{spec.proprio_dim}]."
                    )
                proprio_mask[: spec.proprio_dim] = provided
            actions.append(action_pad)
            proprios.append(proprio_pad)
            action_masks.append(action_mask)
            proprio_masks.append(proprio_mask)

        batch: Dict[str, Any] = {
            "action_space_id": action_space_ids,
            "action": torch.stack(actions),
            "proprio": torch.stack(proprios),
            "action_valid_mask": torch.stack(action_masks),
            "proprio_valid_mask": torch.stack(proprio_masks),
        }
        batch["embodiment_id"] = torch.tensor(
            [self._scalar_id(sample, "embodiment_id") for sample in samples],
            dtype=torch.long,
        )
        if any("dataset_id" in sample for sample in samples):
            batch["dataset_id"] = torch.tensor(
                [self._scalar_id(sample, "dataset_id") for sample in samples],
                dtype=torch.long,
            )
        for key in ("image_input", "image_mask", "language_instruction"):
            present = [key in sample for sample in samples]
            if any(present) and not all(present):
                raise KeyError(
                    f"{key} must be present in either every sample or none."
                )
        for key in ("image_input", "image_mask"):
            if key in samples[0]:
                values = [sample[key] for sample in samples]
                batch[key] = values if key == "image_input" else torch.stack(values)
        if "language_instruction" in samples[0]:
            batch["language_instruction"] = [
                sample["language_instruction"] for sample in samples
            ]
        sampling_keys = (
            "sampling_group",
            "dataset_name",
            "task_name",
            "episode_index",
            "sample_index",
        )
        batch["dataset_name"] = [sample.get("dataset_name") for sample in samples]
        if any("sampling_group" in sample for sample in samples):
            for key in sampling_keys:
                if not all(key in sample for sample in samples):
                    raise KeyError(
                        f"Hierarchical samples must all provide {key!r}."
                    )
            batch["sampling_metadata"] = [
                {key: sample[key] for key in sampling_keys}
                for sample in samples
            ]
        return batch


__all__ = ["HeterogeneousActionCollator"]
