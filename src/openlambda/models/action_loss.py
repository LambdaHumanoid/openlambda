"""Losses for batches containing multiple action-space contracts."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .action_specs import ACTION_SPACE_SPECS, ActionSpaceSpec


def compute_heterogeneous_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_space_id: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    action_space_specs: Mapping[int, ActionSpaceSpec] | None = None,
) -> dict[str, torch.Tensor]:
    """Mean over valid scalar targets; blocks are diagnostics, not weights."""
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target must have the same shape [B,T,D].")
    # Accumulate counts and squared errors in FP32 even under BF16 autocast.
    predicted, target = predicted.float(), target.float()
    ids = action_space_id.reshape(-1).long().to(predicted.device)
    if ids.numel() != predicted.shape[0]:
        raise ValueError("action_space_id must have one entry per batch sample.")
    if valid_mask is None:
        valid_mask = torch.ones_like(predicted, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=predicted.device, dtype=torch.bool)
    if valid_mask.shape != predicted.shape:
        raise ValueError("valid_mask must have the same shape as predicted.")

    specs = dict(action_space_specs or ACTION_SPACE_SPECS)
    present_ids = set(ids.detach().cpu().tolist())
    unknown_ids = sorted(present_ids - set(specs))
    if unknown_ids:
        raise KeyError(f"Unknown action-space ids in loss: {unknown_ids}.")

    sample_loss_sum = predicted.new_zeros((predicted.shape[0],))
    sample_weight_sum = predicted.new_zeros((predicted.shape[0],))
    metrics: dict[str, torch.Tensor] = {}
    for space_id in sorted(present_ids):
        sample_indices = torch.nonzero(
            ids == space_id, as_tuple=False
        ).squeeze(-1)
        spec = specs[space_id]
        pred_group = predicted.index_select(0, sample_indices)
        target_group = target.index_select(0, sample_indices)
        mask_group = valid_mask.index_select(0, sample_indices)
        for block in spec.blocks:
            block_slice = slice(block.start, block.end)
            block_mask = mask_group[:, :, block_slice]
            squared_error = F.mse_loss(
                pred_group[:, :, block_slice],
                target_group[:, :, block_slice],
                reduction="none",
            )
            block_count = block_mask.sum(dim=(1, 2))
            valid_samples = block_count > 0
            if not bool(valid_samples.any()):
                continue
            block_sum = squared_error.masked_fill(~block_mask, 0.0).sum(dim=(1, 2))
            block_per_sample = (
                block_sum[valid_samples] / block_count[valid_samples]
            )
            original_indices = sample_indices[valid_samples]
            sample_loss_sum = sample_loss_sum.index_add(
                0, original_indices, block_sum[valid_samples]
            )
            sample_weight_sum = sample_weight_sum.index_add(
                0,
                original_indices,
                block_count[valid_samples].to(predicted.dtype),
            )
            metrics[f"loss/{spec.name}/{block.name}"] = (
                block_per_sample.mean().detach()
            )

    supervised = sample_weight_sum > 0
    if not bool(supervised.all()):
        missing = torch.nonzero(
            ~supervised, as_tuple=False
        ).reshape(-1).tolist()
        raise ValueError(f"Samples without valid action supervision: {missing}.")
    metrics["loss"] = sample_loss_sum.sum() / sample_weight_sum.sum()
    return metrics


__all__ = ["compute_heterogeneous_loss"]
