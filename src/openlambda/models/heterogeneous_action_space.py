"""Heterogeneous action-space encoders/decoders for mixed-embodiment batches."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn

from .action_specs import (
    ACTION_SPACE_SPECS,
    ActionSpaceSpec,
)


def timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    if timestep.ndim != 1:
        timestep = timestep.reshape(-1)
    half = dim // 2
    if half == 0:
        return timestep.new_zeros((timestep.shape[0], dim), dtype=torch.float32)
    exponent = -torch.log(torch.tensor(10000.0, device=timestep.device))
    exponent = (
        exponent
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    values = timestep.float()[:, None] * torch.exp(exponent)[None, :]
    embedding = torch.cat((values.cos(), values.sin()), dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class HeterogeneousActionSpace(nn.Module):
    """Encodes and decodes padded actions with per-space projections."""

    def __init__(
        self,
        hidden_size: int,
        dim_time: int,
        action_space_specs: Mapping[int, ActionSpaceSpec] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.dim_time = int(dim_time)
        self.specs: Dict[int, ActionSpaceSpec] = dict(
            action_space_specs or ACTION_SPACE_SPECS
        )
        if not self.specs:
            raise ValueError("At least one action-space spec is required.")
        self.max_action_dim = max(spec.action_dim for spec in self.specs.values())
        self.max_proprio_dim = max(spec.proprio_dim for spec in self.specs.values())
        self.action_encoders = nn.ModuleDict(
            {
                str(space_id): nn.Linear(
                    spec.action_dim + 2 * spec.proprio_dim + self.dim_time,
                    self.hidden_size,
                )
                for space_id, spec in sorted(self.specs.items())
            }
        )
        self.action_decoders = nn.ModuleDict(
            {
                str(space_id): nn.Linear(self.hidden_size, spec.action_dim)
                for space_id, spec in sorted(self.specs.items())
            }
        )

    def extend_specs(
        self,
        action_space_specs: Mapping[int, ActionSpaceSpec],
    ) -> tuple[int, ...]:
        """Append new contracts without changing any checkpointed contract."""
        requested = dict(action_space_specs)
        existing_names = {spec.name: space_id for space_id, spec in self.specs.items()}
        for space_id, current in self.specs.items():
            if space_id not in requested:
                raise ValueError(
                    f"Cannot remove checkpoint action-space id {space_id} "
                    f"({current.name})."
                )
            if requested[space_id] != current:
                raise ValueError(
                    f"Cannot change checkpoint action-space id {space_id}: "
                    f"{current} != {requested[space_id]}."
                )

        additions = []
        for space_id, spec in sorted(requested.items()):
            owner = existing_names.get(spec.name)
            if owner is not None and owner != space_id:
                raise ValueError(
                    f"Action-space name {spec.name!r} already belongs to id {owner}."
                )
            if space_id not in self.specs:
                additions.append((space_id, spec))

        if not additions:
            return ()
        reference = next(self.action_encoders.parameters())
        for space_id, spec in additions:
            encoder = nn.Linear(
                spec.action_dim + 2 * spec.proprio_dim + self.dim_time,
                self.hidden_size,
            ).to(device=reference.device, dtype=reference.dtype)
            decoder = nn.Linear(self.hidden_size, spec.action_dim).to(
                device=reference.device,
                dtype=reference.dtype,
            )
            self.action_encoders[str(space_id)] = encoder
            self.action_decoders[str(space_id)] = decoder
            self.specs[space_id] = spec
        self.max_action_dim = max(spec.action_dim for spec in self.specs.values())
        self.max_proprio_dim = max(spec.proprio_dim for spec in self.specs.values())
        return tuple(space_id for space_id, _ in additions)

    def _validate_ids(
        self,
        action_space_id: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        ids = action_space_id.reshape(-1).long()
        if ids.numel() != batch_size:
            raise ValueError(
                "action_space_id must contain one id per sample, got "
                f"{ids.numel()} for batch {batch_size}."
            )
        unknown = sorted(set(ids.detach().cpu().tolist()) - set(self.specs))
        if unknown:
            raise KeyError(f"Unknown action-space ids in batch: {unknown}.")
        return ids

    def _group_indices(self, ids: torch.Tensor) -> Iterable[Tuple[int, torch.Tensor]]:
        for space_id in sorted(set(ids.detach().cpu().tolist())):
            indices = torch.nonzero(ids == space_id, as_tuple=False).squeeze(-1)
            yield int(space_id), indices

    def encode(
        self,
        action: torch.Tensor,
        proprio: torch.Tensor,
        timestep: torch.Tensor,
        action_space_id: torch.Tensor,
        proprio_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action.ndim != 3 or proprio.ndim != 2:
            raise ValueError("action must be [B,T,D] and proprio must be [B,P].")
        batch_size, horizon, _ = action.shape
        if proprio.shape[0] != batch_size:
            raise ValueError("action and proprio batch sizes differ.")
        ids = self._validate_ids(action_space_id, batch_size).to(action.device)
        if action.shape[-1] != self.max_action_dim:
            raise ValueError(
                f"Padded action width must be {self.max_action_dim}, got {action.shape[-1]}."
            )
        if proprio.shape[-1] != self.max_proprio_dim:
            raise ValueError(
                f"Padded proprio width must be {self.max_proprio_dim}, got {proprio.shape[-1]}."
            )
        if proprio_valid_mask is None:
            proprio_valid_mask = torch.ones_like(proprio, dtype=torch.bool)
        if proprio_valid_mask.shape != proprio.shape:
            raise ValueError("proprio_valid_mask must have the same shape as proprio.")
        proprio_valid_mask = proprio_valid_mask.to(
            device=proprio.device,
            dtype=torch.bool,
        )
        timestep = timestep.to(action.device)
        if timestep.ndim == 1:
            if timestep.shape[0] != batch_size:
                raise ValueError("A 1D timestep must contain one value per sample.")
            time = timestep_embedding(timestep, self.dim_time).to(action.dtype)
            time = time.unsqueeze(1).expand(-1, horizon, -1)
        elif timestep.ndim == 2:
            if timestep.shape != (batch_size, horizon):
                raise ValueError("A 2D timestep must have shape [B,T].")
            time = timestep_embedding(
                timestep.reshape(-1),
                self.dim_time,
            ).reshape(batch_size, horizon, self.dim_time).to(action.dtype)
        else:
            raise ValueError("timestep must have shape [B] or [B,T].")
        encoded = action.new_zeros((batch_size, horizon, self.hidden_size))
        for space_id, indices in self._group_indices(ids):
            spec = self.specs[space_id]
            group_action = action.index_select(0, indices)[:, :, : spec.action_dim]
            group_proprio = proprio.index_select(0, indices)[:, : spec.proprio_dim]
            group_proprio_mask = proprio_valid_mask.index_select(
                0, indices
            )[:, : spec.proprio_dim]
            group_proprio = group_proprio * group_proprio_mask.to(group_proprio.dtype)
            group_time = time.index_select(0, indices)
            group_input = torch.cat(
                (
                    group_action,
                    group_proprio.unsqueeze(1).expand(-1, horizon, -1),
                    group_proprio_mask.to(group_proprio.dtype)
                    .unsqueeze(1)
                    .expand(-1, horizon, -1),
                    group_time,
                ),
                dim=-1,
            )
            group_encoded = self.action_encoders[str(space_id)](group_input)
            # Inputs are stored in float32 for the loss contract while the
            # model may run under bf16 autocast; keep the mixed-space buffer
            # and each grouped projection at one dtype for index_copy.
            encoded = encoded.index_copy(0, indices, group_encoded.to(encoded.dtype))
        return encoded

    def decode(self, hidden: torch.Tensor, action_space_id: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [B,T,H].")
        ids = self._validate_ids(action_space_id, hidden.shape[0]).to(hidden.device)
        decoded = hidden.new_zeros(
            (hidden.shape[0], hidden.shape[1], self.max_action_dim)
        )
        for space_id, indices in self._group_indices(ids):
            spec = self.specs[space_id]
            group_hidden = hidden.index_select(0, indices)
            group_decoded = self.action_decoders[str(space_id)](group_hidden)
            padded = hidden.new_zeros(
                (indices.numel(), hidden.shape[1], self.max_action_dim)
            )
            padded[:, :, : spec.action_dim] = group_decoded
            decoded = decoded.index_copy(0, indices, padded)
        return decoded

    def action_mask(
        self,
        action_space_id: torch.Tensor,
        horizon: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ids = action_space_id.reshape(-1).long()
        mask = torch.zeros(
            (ids.numel(), horizon, self.max_action_dim),
            dtype=torch.bool,
            device=device or ids.device,
        )
        ids = ids.to(mask.device)
        self._validate_ids(ids, ids.numel())
        for space_id, indices in self._group_indices(ids):
            mask[indices, :, : self.specs[space_id].action_dim] = True
        return mask

    def proprio_mask(
        self,
        action_space_id: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ids = action_space_id.reshape(-1).long()
        mask = torch.zeros(
            (ids.numel(), self.max_proprio_dim),
            dtype=torch.bool,
            device=device or ids.device,
        )
        ids = ids.to(mask.device)
        self._validate_ids(ids, ids.numel())
        for space_id, indices in self._group_indices(ids):
            mask[indices, : self.specs[space_id].proprio_dim] = True
        return mask

    def trim_single_space(
        self,
        action: torch.Tensor,
        action_space_id: torch.Tensor,
    ) -> torch.Tensor:
        ids = self._validate_ids(action_space_id, action.shape[0])
        unique = sorted(set(ids.detach().cpu().tolist()))
        if len(unique) != 1:
            return action
        return action[..., : self.specs[unique[0]].action_dim]


__all__ = ["HeterogeneousActionSpace", "timestep_embedding"]
