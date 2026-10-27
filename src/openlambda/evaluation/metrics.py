"""Open-loop errors in the original action scale; affine offsets cancel."""
import json
from functools import lru_cache
from pathlib import Path
import torch


@lru_cache(maxsize=256)
def _scale(path):
    return json.loads(Path(path).read_text())["action"]["scale"]


def accumulate_original_scale_errors(predicted, inputs, names, metas, specs, totals, counts):
    for row, name in enumerate(names):
        meta = metas[name]
        spec = specs[int(inputs["action_space_id"][row])]
        scale = torch.tensor(_scale(meta["normalization_path"]), device=predicted.device, dtype=torch.float32)
        mask = inputs["action_valid_mask"][row, :, :spec.action_dim]
        error = (predicted[row, :, :spec.action_dim].float() - inputs["action"][row, :, :spec.action_dim].float()) * scale
        for suffix, selection in [("", slice(None)), *[(f"/{spec.name}/{b.name}", slice(b.start, b.end)) for b in spec.blocks]]:
            valid = mask[:, selection]
            values = error[:, selection][valid].double()
            totals["original_scale_mae" + suffix] += values.abs().sum()
            totals["original_scale_mse" + suffix] += values.square().sum()
            counts["original_scale_mae" + suffix] += values.numel()
            counts["original_scale_mse" + suffix] += values.numel()
