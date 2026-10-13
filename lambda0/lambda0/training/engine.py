from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from packaging.version import Version

from lambda0.data import create_dataloader
from lambda0.models import ACTION_CHUNK_SIZE, Lambda0, Lambda0Processor
from lambda0.models.state_conditioning import (
    DEFAULT_STATE_DISCRETE_BINS,
    DISCRETE_VLM,
    STATE_CONDITIONING_MODES,
    validate_state_conditioning,
)


FAST_LINEAR_ATTENTION_MINIMUMS = {
    "triton": Version("3.2.0"),
    "flash-linear-attention": Version("0.4.2"),
    "fla-core": Version("0.4.2"),
    "causal-conv1d": Version("1.5.0.post8"),
}


def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if is_main:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(Path(output_dir) / "train.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_args_parser():
    parser = argparse.ArgumentParser("Lambda0 Training", add_help=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Official Qwen3.5 checkpoint or an existing Lambda0 checkpoint.",
    )
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--variant", choices=("0.8b", "2b", "4b"), default=None)
    parser.add_argument("--output_dir", type=str, default="runnings")
    parser.add_argument(
        "--report_to",
        nargs="+",
        choices=("tensorboard", "wandb"),
        default=("tensorboard",),
        help="Metric backends initialized by Accelerate.",
    )
    parser.add_argument("--wandb_project", type=str, default="Lambda0")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)

    parser.add_argument("--train_metas_path", type=str, required=True)
    parser.add_argument("--release_data_manifest", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--prediction_type", choices=("x0", "velocity"), default="velocity")
    parser.add_argument(
        "--validation_metas_path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable fixed validation split definition.",
    )
    parser.add_argument("--validation_interval", type=int, default=0)
    parser.add_argument("--validation_samples_per_episode", type=int, default=4)
    parser.add_argument("--validation_noise_samples", type=int, default=4)
    parser.add_argument("--validation_seed", type=int, default=20260813)
    parser.add_argument("--validation_max_batches", type=int, default=0)
    parser.add_argument(
        "--validation_action_chunk_mse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Generate complete action chunks and report mask-aware action MSE "
            "in addition to the flow-matching validation objective."
        ),
    )
    parser.add_argument(
        "--validation_action_chunk_samples",
        type=int,
        default=4,
        help="Stochastic action chunks averaged before action MSE is computed.",
    )
    parser.add_argument(
        "--validation_action_chunk_steps",
        type=int,
        default=10,
        help="Flow-integration steps used to generate each validation chunk.",
    )
    parser.add_argument(
        "--validation_at_start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run validation at step zero before the first optimizer update.",
    )
    parser.add_argument(
        "--validation_use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional metadata dataset_name allowlist.",
    )
    parser.add_argument(
        "--dataset_weights_json",
        type=str,
        default=None,
        help="Optional JSON object overriding per-dataset sampling weights.",
    )
    parser.add_argument(
        "--sampling_plan_json",
        type=str,
        default=None,
        help="Optional hierarchical IID sampling plan encoded as JSON.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Micro-batches per optimizer step; the logged step is an optimizer step.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_actions",
        type=int,
        choices=(ACTION_CHUNK_SIZE,),
        default=ACTION_CHUNK_SIZE,
        help="Fixed action chunk shared by every Lambda0 training stage.",
    )
    parser.add_argument("--num_domains", type=int, default=30)
    parser.add_argument(
        "--reset_action_space_ids", type=int, nargs="+", default=[],
        help="Reinitialize only these action-space projections after loading initial weights; full resume restores checkpoint weights afterwards.",
    )
    parser.add_argument(
        "--trainable_action_space_ids",
        type=int,
        nargs="+",
        default=None,
        help="Only optimize the listed heterogeneous action-space heads.",
    )
    parser.add_argument("--soft_prompt_length", type=int, default=32)
    parser.add_argument(
        "--use_proprio",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--state_conditioning_mode",
        choices=STATE_CONDITIONING_MODES,
        default="discrete_vlm",
        help=(
            "Inject normalized state as discrete tokens in the VLM prompt."
        ),
    )
    parser.add_argument(
        "--state_discrete_bins",
        type=int,
        default=DEFAULT_STATE_DISCRETE_BINS,
        help="Number of bins used by discrete VLM state conditioning.",
    )

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--vlm_lr_coef", type=float, default=0.1)
    parser.add_argument("--conditioning_lr_coef", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--fused_optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the fused CUDA AdamW implementation.",
    )

    parser.add_argument("--iters", type=int, default=100_000)
    parser.add_argument("--freeze_steps", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--new_module_warmup_steps", type=int, default=1000)
    parser.add_argument("--use_cosine_decay", action="store_true")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--save_interval", type=int, default=10_000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay in [0,1); set to 0 to disable EMA.",
    )
    parser.add_argument(
        "--ema_update_interval",
        type=int,
        default=1,
        help="Update EMA every N optimizer steps using decay**N.",
    )
    parser.add_argument(
        "--rtc_max_delay",
        type=int,
        default=0,
        help="Maximum ground-truth action-prefix length for training-time RTC.",
    )
    parser.add_argument(
        "--action_expert_noise_samples",
        type=int,
        default=4,
        help=(
            "Independent flow-matching noise/timestep samples per VLM encoding. "
            "The loss remains averaged over the expanded action batch."
        ),
    )
    parser.add_argument(
        "--action_expert_gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Checkpoint only Action Expert blocks without recomputing the VLM.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="auto",
        help="'auto' resumes output_dir/last, 'none' starts fresh, or use a checkpoint path.",
    )
    parser.add_argument(
        "--allow_world_size_change",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow a replicated DDP checkpoint to resume with a different world "
            "size while still restoring model and optimizer state."
        ),
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require_fast_linear_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of falling back when the Qwen3.5 FLA path is unavailable.",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override DDP unused-parameter detection. By default it is disabled "
            "for a static graph and enabled otherwise."
        ),
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ddp_bucket_cap_mb", type=int, default=25)
    return parser


def set_seed(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def tracker_configuration(args: argparse.Namespace) -> dict[str, object]:
    """Return scalar tracker metadata accepted by every enabled backend."""
    configuration: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)):
            configuration[key] = value
        elif value is None:
            configuration[key] = "null"
        else:
            configuration[key] = json.dumps(value, sort_keys=True)
    return configuration


def wandb_resume_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Resume an existing run without requesting W&B's private rewind API.

    Training checkpoints can trail the last uploaded metric by a few logging
    steps.  Public ``resume='allow'`` preserves the cloud history and ignores
    any duplicate steps until training advances beyond it.  ``resume_from``
    would instead request server-side history rewinding, which is not enabled
    for ordinary W&B accounts and makes tracker initialization fail.
    """
    os.environ.pop("WANDB_RESUME", None)
    run_id = os.environ.get("WANDB_RUN_ID")
    if not run_id:
        return {}
    return {"id": run_id, "resume": "allow"}


def tracker_init_kwargs(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    if "wandb" not in args.report_to:
        return {}
    wandb_kwargs = {
        key: value
        for key, value in {
            "dir": str(Path(args.output_dir)),
            "entity": args.wandb_entity,
            "name": args.wandb_run_name,
            "group": args.wandb_group,
            "tags": args.wandb_tags,
        }.items()
        if value is not None
    }
    wandb_kwargs.update(wandb_resume_kwargs(args))
    return {"wandb": wandb_kwargs}


def validate_fast_linear_attention_environment() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "Qwen3.5 fast linear attention requires Python 3.11 or newer; "
            f"found {sys.version_info.major}.{sys.version_info.minor}."
        )
    for package, minimum in FAST_LINEAR_ATTENTION_MINIMUMS.items():
        try:
            installed = Version(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"Qwen3.5 fast linear attention requires {package}>={minimum}."
            ) from error
        if installed < minimum:
            raise RuntimeError(
                f"Qwen3.5 fast linear attention requires {package}>={minimum}; "
                f"found {installed}."
            )


def load_model_and_processor(args):
    if args.require_fast_linear_attention:
        validate_fast_linear_attention_environment()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            is_fast_path_available,
        )

        if not is_fast_path_available:
            raise RuntimeError(
                "Qwen3.5 fast linear attention is required, but FLA or "
                "causal-conv1d is unavailable."
            )
    model = Lambda0.from_pretrained_or_vlm(
        args.models,
        variant=args.variant,
        num_domains=args.num_domains,
        soft_prompt_length=args.soft_prompt_length,
        num_actions=args.num_actions,
        use_proprio=args.use_proprio,
        state_conditioning_mode=args.state_conditioning_mode,
        state_discrete_bins=args.state_discrete_bins,
        dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
        strict_checkpoint_loading=True,
        revision=args.model_revision,
        prediction_type=args.prediction_type,
    )
    for space_id in getattr(args, "reset_action_space_ids", []):
        if space_id not in model.action_space.specs:
            raise ValueError(f"Unknown action-space ID to reset: {space_id}")
        model.action_space.action_encoders[str(space_id)].reset_parameters()
        model.action_space.action_decoders[str(space_id)].reset_parameters()
        print(f"Reinitialized action-space {space_id} encoder/decoder; VLM and action-expert backbone retained.", flush=True)
    processor = Lambda0Processor.from_pretrained(args.processor_path or args.models, revision=args.model_revision)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    elif args.action_expert_gradient_checkpointing:
        model.action_expert.gradient_checkpointing = True
    return model, processor


def _group_name(parameter_name: str) -> str:
    if parameter_name.startswith("vlm.") or ".vlm." in parameter_name:
        return "vlm"
    if (
        "soft_prompt_hub" in parameter_name
        or "embodiment_bias" in parameter_name
    ):
        return "conditioning"
    if (
        "action_space.action_encoders" in parameter_name
        or "action_space.action_decoders" in parameter_name
    ):
        return "action_heads"
    return "action_expert"


def configure_trainable_action_spaces(model, requested_ids) -> tuple[int, ...]:
    policy = model
    action_space = policy.action_space
    available = {int(space_id) for space_id in action_space.action_encoders}
    if requested_ids is None:
        selected = available
    else:
        selected = {int(space_id) for space_id in requested_ids}
        if not selected:
            raise ValueError("--trainable_action_space_ids must not be empty.")
        unknown = sorted(selected - available)
        if unknown:
            raise ValueError(
                f"Unknown trainable action-space ids {unknown}; "
                f"available ids are {sorted(available)}."
            )
    for space_id in available:
        trainable = space_id in selected
        for module_dict in (
            action_space.action_encoders,
            action_space.action_decoders,
        ):
            for parameter in module_dict[str(space_id)].parameters():
                parameter.requires_grad_(trainable)
    return tuple(sorted(selected))


def build_optimizer(model, args):
    grouped = {
        name: [] for name in ("vlm", "action_expert", "conditioning", "action_heads")
    }
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped[_group_name(parameter_name)].append(parameter)
    if not any(grouped.values()):
        raise ValueError("The model has no trainable parameters.")

    initial_lrs = {
        "vlm": 0.0,
        "action_expert": 0.0,
        "conditioning": 0.0,
        "action_heads": 0.0,
    }
    parameter_groups = [
        {
            "name": name,
            "params": parameters,
            "lr": initial_lrs[name],
            "weight_decay": args.weight_decay,
        }
        for name, parameters in grouped.items()
        if parameters
    ]
    return AdamW(
        parameter_groups,
        betas=tuple(args.betas),
        eps=1e-8,
        fused=args.fused_optimizer,
    )


def set_group_lr(optimizer, name: str, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        if group["name"] == name:
            group["lr"] = learning_rate


def scheduled_learning_rate(
    step,
    start,
    warmup,
    total,
    base_lr,
    min_ratio,
    use_cosine_decay,
):
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress + 1) / warmup
    if not use_cosine_decay:
        return base_lr
    remaining_steps = max(1, total - start - warmup - 1)
    progress = min(1.0, (progress - warmup) / remaining_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)


def update_group_lrs(optimizer, step, args) -> None:
    base_lrs = {
        "vlm": args.learning_rate * args.vlm_lr_coef,
        "action_expert": args.learning_rate,
        "conditioning": args.learning_rate * args.conditioning_lr_coef,
        "action_heads": args.learning_rate,
    }
    for name, base_lr in base_lrs.items():
        if name == "vlm":
            start = args.freeze_steps
            warmup = args.warmup_steps
        else:
            start = 0
            warmup = args.new_module_warmup_steps
        learning_rate = scheduled_learning_rate(
            step,
            start,
            warmup,
            args.iters,
            base_lr,
            args.min_lr_ratio,
            args.use_cosine_decay,
        )
        set_group_lr(optimizer, name, learning_rate)


def prepare_inputs(batch, processor, device, policy):
    batch = dict(batch)
    batch.pop("dataset_id", None)
    language_instruction = batch.pop("language_instruction")
    image_input = batch.pop("image_input")
    image_mask = batch.pop("image_mask")
    multimodal_inputs = processor.prepare_batch(
        image_input,
        image_mask,
        language_instruction,
        proprio=batch.get("proprio"),
        proprio_valid_mask=batch.get("proprio_valid_mask"),
        state_conditioning_mode=policy.config.state_conditioning_mode,
        state_discrete_bins=policy.config.state_discrete_bins,
    )
    inputs = {**batch, **multimodal_inputs}
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }


def parse_validation_metas(values: list[str]) -> dict[str, str]:
    result = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        name, path = name.strip(), path.strip()
        if not separator or not name or not path:
            raise ValueError(
                "--validation_metas_path must use NAME=PATH syntax."
            )
        if name in result:
            raise ValueError(f"Duplicate validation split name: {name}")
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Validation metadata path does not exist: {path}")
        result[name] = path
    return result


def required_validation_metrics(
    split_names,
    action_chunk_mse: bool,
) -> set[str]:
    suffix = "action_chunk_mse" if action_chunk_mse else "loss_total"
    return {f"validation/{name}/{suffix}" for name in split_names}


def accumulate_action_chunk_mse(predicted, target, action_space_id, valid_mask, specs, totals, counts):
    """Equal-weight normalized scalar MSE plus block diagnostics."""
    if predicted.shape[:2] != target.shape[:2] or predicted.shape[-1] > target.shape[-1]:
        raise ValueError("Generated/target action shapes are incompatible.")
    if predicted.shape[-1] < target.shape[-1]:
        padded = predicted.new_zeros(target.shape)
        padded[..., :predicted.shape[-1]] = predicted
        predicted = padded
    squared = (predicted.float() - target.float()).square()
    mask = valid_mask.bool()
    if mask.shape != target.shape or not mask.any(dim=(1, 2)).all():
        raise ValueError("Invalid or empty validation supervision.")
    totals["action_chunk_mse"] += squared[mask].double().sum()
    counts["action_chunk_mse"] += mask.sum()
    for space_id, spec in specs.items():
        selected = action_space_id.reshape(-1) == space_id
        for block in spec.blocks:
            key = f"action_chunk_mse/{spec.name}/{block.name}"
            valid = mask[selected, :, block.start:block.end]
            error = squared[selected, :, block.start:block.end]
            totals[key] += error[valid].double().sum()
            counts[key] += valid.sum()


def run_validation(
    accelerator,
    model,
    processor,
    dataloaders: dict[str, object],
    global_step: int,
    noise_samples: int,
    seed: int,
    max_batches: int,
    action_chunk_mse: bool = False,
    action_chunk_samples: int = 4,
    action_chunk_steps: int = 10,
    wandb_commit: bool = True,
    ema_policy=None,
    metric_prefix="validation",
) -> dict[str, float]:
    policy = accelerator.unwrap_model(model)
    was_training = policy.training
    policy.eval()
    logs = {}
    with torch.no_grad():
        for split_index, (split_name, dataloader) in enumerate(dataloaders.items()):
            totals = {
                "loss": torch.zeros(
                    (), device=accelerator.device, dtype=torch.float64
                )
            }
            counts = {"loss": torch.zeros_like(totals["loss"])}
            if action_chunk_mse:
                totals["action_chunk_mse"] = torch.zeros_like(totals["loss"])
                counts["action_chunk_mse"] = torch.zeros_like(totals["loss"])
            for spec in policy.action_space.specs.values():
                for suffix in ("", *(f"/{spec.name}/{b.name}" for b in spec.blocks)):
                    for metric in ("original_scale_mae", "original_scale_mse"):
                        totals[metric + suffix] = torch.zeros_like(totals["loss"])
                        counts[metric + suffix] = torch.zeros_like(totals["loss"])
                for block in spec.blocks:
                    name = f"loss/{spec.name}/{block.name}"
                    totals[name] = torch.zeros_like(totals["loss"])
                    counts[name] = torch.zeros_like(totals["loss"])
                    if action_chunk_mse:
                        action_name = (
                            f"action_chunk_mse/{spec.name}/{block.name}"
                        )
                        totals[action_name] = torch.zeros_like(totals["loss"])
                        counts[action_name] = torch.zeros_like(totals["loss"])
            for batch_index, batch in enumerate(dataloader):
                if max_batches and batch_index >= max_batches:
                    break
                inputs = prepare_inputs(
                    batch,
                    processor,
                    accelerator.device,
                    policy,
                )
                loss_dict = policy(
                    **inputs,
                    rtc_max_delay=0,
                    action_expert_noise_samples=noise_samples,
                    noise_seed=(
                        seed
                        + split_index * 1_000_000_000
                        + accelerator.process_index * 1_000_000
                        + batch_index
                    ),
                )
                if action_chunk_mse:
                    generation_keys = (
                        "input_ids",
                        "attention_mask",
                        "pixel_values",
                        "image_grid_thw",
                        "mm_token_type_ids",
                        "proprio",
                        "action_space_id",
                        "proprio_valid_mask",
                        "embodiment_id",
                        "aux_pixel_values",
                        "aux_image_grid_thw",
                        "aux_image_owner",
                    )
                    generation_inputs = {
                        name: inputs[name]
                        for name in generation_keys
                        if name in inputs
                    }
                    generator = torch.Generator(device=accelerator.device)
                    generator.manual_seed(
                        seed
                        + 500_000_000_000
                        + split_index * 1_000_000_000
                        + accelerator.process_index * 1_000_000
                        + batch_index
                    )
                    with accelerator.autocast():
                        predicted_action = policy.generate_actions(
                            **generation_inputs,
                            steps=action_chunk_steps,
                            num_samples=action_chunk_samples,
                            generator=generator,
                        )
                    accumulate_action_chunk_mse(
                        predicted_action,
                        inputs["action"],
                        inputs["action_space_id"],
                        inputs["action_valid_mask"],
                        policy.action_space.specs,
                        totals,
                        counts,
                    )
                    from lambda0.evaluation.metrics import accumulate_original_scale_errors
                    accumulate_original_scale_errors(predicted_action, inputs, batch["dataset_name"],
                                               dataloader.dataset.metas, policy.action_space.specs,
                                               totals, counts)
                batch_size = int(inputs["action"].shape[0])
                totals["loss"] += loss_dict["loss"].detach().to(
                    dtype=torch.float64
                ) * batch_size
                counts["loss"] += batch_size
                ids = inputs["action_space_id"].reshape(-1)
                for name, value in loss_dict.items():
                    if name == "loss":
                        continue
                    matches = [
                        (spec, block)
                        for spec in policy.action_space.specs.values()
                        for block in spec.blocks
                        if name == f"loss/{spec.name}/{block.name}"
                    ]
                    if len(matches) != 1:
                        raise KeyError(f"Unknown validation metric: {name}")
                    spec, block = matches[0]
                    selected = inputs["action_valid_mask"][ids == spec.id]
                    sample_count = selected[
                        :, :, block.start : block.end
                    ].any(dim=(1, 2)).sum()
                    totals[name] += value.detach().to(
                        dtype=torch.float64
                    ) * sample_count
                    counts[name] += sample_count
            packed_names = sorted(totals)
            packed = torch.stack(
                [
                    *(totals[name] for name in packed_names),
                    *(counts[name] for name in packed_names),
                ]
            )
            packed = accelerator.reduce(packed, reduction="sum")
            offset = len(packed_names)
            global_counts = packed[offset:]
            if global_counts[packed_names.index("loss")].item() == 0:
                raise RuntimeError(
                    f"Validation split {split_name!r} emitted no samples."
                )
            for index, name in enumerate(packed_names):
                if global_counts[index].item() == 0:
                    continue
                metric_name = "loss_total" if name == "loss" else name
                logs[f"validation/{split_name}/{metric_name}"] = (
                    packed[index] / global_counts[index]
                ).item()
                if name.startswith("original_scale_mse"):
                    logs[f"validation/{split_name}/{name.replace('original_scale_mse', 'original_scale_rmse', 1)}"] = math.sqrt(
                        max(0.0, (packed[index] / global_counts[index]).item()))
            logs[f"validation/{split_name}/samples"] = global_counts[
                packed_names.index("loss")
            ].item()
            if action_chunk_mse:
                logs[f"validation/{split_name}/action_chunk_samples"] = int(
                    action_chunk_samples
                )
                logs[f"validation/{split_name}/action_chunk_steps"] = int(
                    action_chunk_steps
                )
    policy.train(was_training)
    if metric_prefix != "validation":
        logs = {key.replace("validation/", metric_prefix + "/", 1): value for key, value in logs.items()}
    accelerator.log(
        logs,
        step=global_step,
        log_kwargs={"wandb": {"commit": wandb_commit and ema_policy is None}},
    )
    if ema_policy is not None:
        logs.update(run_validation(
            accelerator, ema_policy, processor, dataloaders, global_step,
            noise_samples, seed, max_batches, action_chunk_mse,
            action_chunk_samples, action_chunk_steps, wandb_commit,
            metric_prefix="validation_ema"))
    return logs


def create_ema_model(
    model,
    accelerator,
    decay: float,
    update_interval: int,
) -> AveragedModel | None:
    if decay == 0:
        return None
    ema_model = AveragedModel(
        accelerator.unwrap_model(model),
        device=accelerator.device,
        multi_avg_fn=get_ema_multi_avg_fn(decay**update_interval),
        use_buffers=False,
    )
    ema_model.requires_grad_(False)
    # AveragedModel otherwise copies the first update, skipping EMA at step 1.
    ema_model.n_averaged.fill_(1)
    ema_model.eval()
    accelerator.register_for_checkpointing(ema_model)
    return ema_model


def sync_ema_additions(
    ema_model: AveragedModel | None,
    model,
) -> None:
    """Copy newly introduced action heads before the first EMA update."""
    if ema_model is None:
        return
    online = model
    ema_parameters = dict(ema_model.module.named_parameters())
    ema_buffers = dict(ema_model.module.named_buffers())
    with torch.no_grad():
        for name, parameter in online.named_parameters():
            if name in ema_parameters:
                ema_parameters[name].copy_(parameter)
        for name, buffer in online.named_buffers():
            if name in ema_buffers and name != "n_averaged":
                ema_buffers[name].copy_(buffer)


def resolve_resume_checkpoint(
    output_dir: Path,
    resume_from_checkpoint: str,
) -> Path | None:
    value = resume_from_checkpoint.strip()
    if value.lower() == "none":
        return None
    checkpoint = output_dir / "last" if value.lower() == "auto" else Path(value)
    if value.lower() == "auto" and not checkpoint.exists():
        return None
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    checkpoint = checkpoint.resolve()
    required = (checkpoint / "state.json", checkpoint / "training_state")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Resume checkpoint is incomplete; missing " + ", ".join(missing)
        )
    return checkpoint


def load_training_checkpoint(
    accelerator: Accelerator,
    checkpoint: Path,
    args,
) -> int:
    with (checkpoint / "state.json").open(encoding="utf-8") as file:
        state = json.load(file)
    expected = {
        "training_contract": f"lambda0_{args.prediction_type}_v1",
        "num_actions": int(args.num_actions),
        "use_proprio": bool(args.use_proprio),
        "state_conditioning_mode": args.state_conditioning_mode,
        "state_discrete_bins": int(args.state_discrete_bins),
        "ema_decay": float(args.ema_decay),
        "ema_update_interval": int(args.ema_update_interval),
        "action_expert_noise_samples": int(
            args.action_expert_noise_samples
        ),
    }
    for key, expected_value in expected.items():
        if key not in state:
            raise KeyError(f"Resume checkpoint has no {key!r}: {checkpoint}")
        actual_value = state[key]
        if isinstance(expected_value, float):
            matches = math.isclose(
                float(actual_value),
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        elif isinstance(expected_value, bool):
            matches = bool(actual_value) is expected_value
        elif isinstance(expected_value, str):
            matches = str(actual_value) == expected_value
        else:
            matches = int(actual_value) == expected_value
        if not matches:
            raise ValueError(
                f"Resume checkpoint {key}={actual_value!r} does not match "
                f"current value {expected_value!r}."
            )
    if "world_size" not in state:
        raise KeyError(f"Resume checkpoint has no 'world_size': {checkpoint}")
    stored_world_size = int(state["world_size"])
    current_world_size = int(accelerator.num_processes)
    if (
        stored_world_size != current_world_size
        and not args.allow_world_size_change
    ):
        raise ValueError(
            f"Resume checkpoint world_size={stored_world_size} does not match "
            f"current value {current_world_size}; explicitly enable "
            "--allow_world_size_change only for replicated DDP state."
        )
    stored_accumulation = int(state.get("gradient_accumulation_steps", 1))
    # Include the per-rank micro batch when validating a topology change.
    stored_batch = None
    config_path = checkpoint / "training_config.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as file:
            stored_batch = json.load(file).get("batch_size")
    current_batch = getattr(args, "batch_size", None)
    if stored_batch is not None and current_batch is not None:
        preserved_batch = (
            stored_world_size * stored_accumulation * int(stored_batch)
            == current_world_size * args.gradient_accumulation_steps * int(current_batch)
        )
    else:
        # Older checkpoints may not carry their micro-batch configuration.
        preserved_batch = (
            stored_world_size * stored_accumulation
            == current_world_size * args.gradient_accumulation_steps
        )
    preserved_rank_accumulation = (
        args.allow_world_size_change
        and preserved_batch
    )
    if (
        stored_accumulation != args.gradient_accumulation_steps
        and not preserved_rank_accumulation
    ):
        raise ValueError(
            "Resume checkpoint gradient_accumulation_steps="
            f"{stored_accumulation} does not match current value "
            f"{args.gradient_accumulation_steps}."
        )
    global_step = int(state["global_step"])
    if global_step < 0:
        raise ValueError("Resume checkpoint global_step must be non-negative.")
    accelerator.load_state(str(checkpoint / "training_state"))
    return global_step


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _read_step_marker(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open(encoding="utf-8") as file:
        marker = json.load(file)
    step = int(marker["global_step"])
    if step < 0:
        raise ValueError(f"Invalid negative global_step in {path}: {step}")
    return step


def save_training_checkpoint(
    accelerator: Accelerator,
    model,
    ema_model: AveragedModel | None,
    processor,
    output_dir: Path,
    global_step: int,
    args,
    logger,
) -> Path:
    save_dir = output_dir / f"ckpt-{global_step}"
    temporary_dir = output_dir / f".ckpt-{global_step}.incomplete"
    if accelerator.is_main_process:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir(parents=True)
    accelerator.wait_for_everyone()

    accelerator.save_state(
        str(temporary_dir / "training_state"),
        safe_serialization=True,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        online_model = accelerator.unwrap_model(model)
        online_model.save_pretrained(temporary_dir, safe_serialization=True)
        processor.save_pretrained(temporary_dir)
        if args.release_data_manifest:
            bundle = Path(args.release_data_manifest).parent
            shutil.copytree(bundle, temporary_dir / "data_bundle")
        _write_json_atomic(temporary_dir / "training_config.json", vars(args))
        from lambda0.runtime.environment import environment
        _write_json_atomic(temporary_dir / "environment.json", environment())
        ema_updates = 0
        if ema_model is not None:
            ema_dir = temporary_dir / "ema"
            ema_model.module.save_pretrained(ema_dir, safe_serialization=True)
            processor.save_pretrained(ema_dir)
            ema_updates = int(ema_model.n_averaged.item()) - 1
        _write_json_atomic(
            temporary_dir / "state.json",
            {
                "global_step": global_step,
                "num_actions": int(args.num_actions),
                "use_proprio": bool(args.use_proprio),
                "state_conditioning_mode": args.state_conditioning_mode,
                "state_discrete_bins": int(args.state_discrete_bins),
                "world_size": int(accelerator.num_processes),
                "ema_decay": float(args.ema_decay),
                "ema_update_interval": int(args.ema_update_interval),
                "ema_updates": ema_updates,
                "training_contract": f"lambda0_{args.prediction_type}_v1",
                "prediction_type": args.prediction_type,
                "rtc_max_delay": int(args.rtc_max_delay),
                "action_expert_noise_samples": int(
                    args.action_expert_noise_samples
                ),
                "gradient_accumulation_steps": int(
                    args.gradient_accumulation_steps
                ),
            },
        )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        os.replace(temporary_dir, save_dir)
        last_path = output_dir / "last"
        if last_path.exists() and not last_path.is_symlink():
            raise RuntimeError(
                f"Refusing to replace non-symlink last checkpoint: {last_path}"
            )
        temporary_link = output_dir / ".last.tmp"
        if temporary_link.is_symlink() or temporary_link.exists():
            temporary_link.unlink()
        temporary_link.symlink_to(save_dir.name, target_is_directory=True)
        os.replace(temporary_link, last_path)
        logger.info("Saved checkpoint to %s and updated %s", save_dir, last_path)
    accelerator.wait_for_everyone()
    return save_dir


def trace_training_phase(
    accelerator,
    step: int,
    trace_steps: int,
    phase: str,
    started_at: float,
    batch=None,
) -> None:
    if step >= trace_steps:
        return
    if phase != "batch_ready" and accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    details = ""
    if batch is not None:
        metadata = {}
        tensor_shapes = {}
        for key in ("dataset_id", "embodiment_id", "action_space_id"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                metadata[key] = value.detach().cpu().tolist()
            elif value is not None:
                metadata[key] = value
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                tensor_shapes[key] = tuple(value.shape)
        details = f" metadata={metadata} tensor_shapes={tensor_shapes}"
    if accelerator.device.type == "cuda":
        gib = 1024**3
        details += (
            f" cuda_allocated_gib="
            f"{torch.cuda.memory_allocated(accelerator.device) / gib:.3f}"
            f" cuda_reserved_gib="
            f"{torch.cuda.memory_reserved(accelerator.device) / gib:.3f}"
            f" cuda_max_allocated_gib="
            f"{torch.cuda.max_memory_allocated(accelerator.device) / gib:.3f}"
            f" cuda_max_reserved_gib="
            f"{torch.cuda.max_memory_reserved(accelerator.device) / gib:.3f}"
        )
    elapsed = time.monotonic() - started_at
    print(
        f"[trace rank={accelerator.process_index} step={step} "
        f"phase={phase} elapsed={elapsed:.3f}s]{details}",
        flush=True,
    )


def run_training(args, model, processor, tracker_name="Lambda0") -> None:
    output_dir = Path(args.output_dir)
    find_unused_parameters = (
        not args.ddp_static_graph
        if args.ddp_find_unused_parameters is None
        else args.ddp_find_unused_parameters
    )
    ddp_kwargs = DistributedDataParallelKwargs(
        bucket_cap_mb=args.ddp_bucket_cap_mb,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        static_graph=args.ddp_static_graph,
    )
    report_to = tuple(dict.fromkeys(args.report_to))
    accelerator = Accelerator(
        log_with=report_to,
        project_dir=output_dir,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    project_name = args.wandb_project if "wandb" in report_to else tracker_name
    accelerator.init_trackers(
        project_name,
        config=tracker_configuration(args),
        init_kwargs=tracker_init_kwargs(args),
    )
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    set_seed(args.seed + accelerator.process_index, args.deterministic)
    logger.info("Args: %s", args)
    if accelerator.is_main_process:
        from lambda0.runtime.environment import environment
        _write_json_atomic(output_dir / "resolved_config.json", vars(args))
        _write_json_atomic(output_dir / "environment.json", environment())

    policy = model
    dataset_weights = None
    if args.dataset_weights_json is not None:
        dataset_weights = json.loads(args.dataset_weights_json)
        if not isinstance(dataset_weights, dict):
            raise TypeError("--dataset_weights_json must decode to a JSON object.")
    sampling_plan = None
    if args.sampling_plan_json is not None:
        sampling_plan = json.loads(args.sampling_plan_json)
        if not isinstance(sampling_plan, dict):
            raise TypeError("--sampling_plan_json must decode to a JSON object.")
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=policy.num_actions,
        training=True,
        num_workers=args.num_workers,
        dataset_names=args.dataset_names,
        sample_weights=dataset_weights,
        sampling_plan=sampling_plan,
        sampling_seed=args.seed,
        shard_index=accelerator.process_index,
        num_shards=accelerator.num_processes,
        data_root=args.data_root,
        image_max_pixels=getattr(args, "image_max_pixels", None),
    )
    validation_metas = parse_validation_metas(args.validation_metas_path)
    validation_dataloaders = {
        name: create_dataloader(
            batch_size=args.batch_size,
            metas_path=metas_path,
            num_actions=policy.num_actions,
            training=False,
            num_workers=args.num_workers,
            validation_samples_per_episode=(
                args.validation_samples_per_episode
            ),
            shard_index=accelerator.process_index,
            num_shards=accelerator.num_processes,
            data_root=args.data_root,
        image_max_pixels=getattr(args, "image_max_pixels", None),
        )
        for name, metas_path in validation_metas.items()
    }
    trainable_action_spaces = configure_trainable_action_spaces(
        model,
        args.trainable_action_space_ids,
    )
    logger.info("Trainable action-space heads: %s", trainable_action_spaces)
    optimizer = build_optimizer(model, args)
    # The iterable dataset owns rank and worker sharding. Preparing this
    # dataloader would shard it a second time and discard prefetched batches.
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()
    ema_model = create_ema_model(
        model,
        accelerator,
        args.ema_decay,
        args.ema_update_interval,
    )
    sync_ema_additions(ema_model, accelerator.unwrap_model(model))

    resume_checkpoint = resolve_resume_checkpoint(
        output_dir,
        args.resume_from_checkpoint,
    )
    global_step = 0
    if resume_checkpoint is not None:
        global_step = load_training_checkpoint(
            accelerator,
            resume_checkpoint,
            args,
        )
        logger.info(
            "Resumed complete training state from %s at global_step=%d",
            resume_checkpoint,
            global_step,
        )
        if (
            validation_dataloaders
            and global_step < args.iters
            and global_step > 0
            and global_step % args.validation_interval == 0
        ):
            validation_state_path = (
                output_dir / f"validation_state_{global_step}.json"
            )
            validation_logs = None
            try:
                if validation_state_path.is_file():
                    validation_state = json.loads(
                        validation_state_path.read_text(encoding="utf-8")
                    )
                    validation_step = int(
                        validation_state.get("global_step", -1)
                    )
                    candidate_logs = validation_state.get("metrics")
                    required_metrics = required_validation_metrics(
                        validation_dataloaders,
                        args.validation_action_chunk_mse,
                    )
                    if (
                        validation_step == global_step
                        and isinstance(candidate_logs, dict)
                        and required_metrics.issubset(candidate_logs)
                        and all(
                            math.isfinite(float(value))
                            for value in candidate_logs.values()
                        )
                    ):
                        validation_logs = candidate_logs
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                validation_logs = None
            if validation_logs is None:
                validation_model = (
                    ema_model.module
                    if args.validation_use_ema and ema_model is not None
                    else model
                )
                validation_logs = run_validation(
                    accelerator,
                    validation_model,
                    processor,
                    validation_dataloaders,
                    global_step,
                    args.validation_noise_samples,
                    args.validation_seed,
                    args.validation_max_batches,
                    args.validation_action_chunk_mse,
                    args.validation_action_chunk_samples,
                    args.validation_action_chunk_steps,
                    wandb_commit=False,
                    ema_policy=ema_model.module if ema_model is not None else None,
                )
                if accelerator.is_main_process:
                    _write_json_atomic(
                        output_dir / "validation_state.json",
                        {"global_step": global_step, "metrics": validation_logs},
                    )
                    _write_json_atomic(
                        validation_state_path,
                        {"global_step": global_step, "metrics": validation_logs},
                    )
                    logger.info(
                        "Recovered missing checkpoint validation at step %d: %s",
                        global_step,
                        json.dumps(validation_logs, sort_keys=True),
                    )
            else:
                accelerator.log(
                    validation_logs,
                    step=global_step,
                    log_kwargs={"wandb": {"commit": False}},
                )
                if accelerator.is_main_process:
                    logger.info(
                        "Re-emitted checkpoint validation at step %d: %s",
                        global_step,
                        json.dumps(validation_logs, sort_keys=True),
                    )
    last_log_time = time.time()
    logger.info(
        "Start training at step %d for %d total iterations | world_size=%d "
        "| micro_batch_size=%d | gradient_accumulation_steps=%d "
        "| effective_global_batch_size=%d | ema_decay=%.6f "
        "| action_expert_noise_samples=%d "
        "| effective_action_expert_batch_size=%d "
        "| ema_update_interval=%d | rtc_max_delay=%d",
        global_step,
        args.iters,
        accelerator.num_processes,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.batch_size
        * args.gradient_accumulation_steps
        * accelerator.num_processes,
        args.ema_decay,
        args.action_expert_noise_samples,
        args.batch_size
        * args.gradient_accumulation_steps
        * accelerator.num_processes
        * args.action_expert_noise_samples,
        args.ema_update_interval,
        args.rtc_max_delay,
    )
    if validation_dataloaders and args.validation_at_start and global_step == 0:
        validation_model = (
            ema_model.module
            if args.validation_use_ema and ema_model is not None
            else model
        )
        validation_logs = run_validation(
            accelerator,
            validation_model,
            processor,
            validation_dataloaders,
            global_step,
            args.validation_noise_samples,
            args.validation_seed,
            args.validation_max_batches,
            args.validation_action_chunk_mse,
            args.validation_action_chunk_samples,
            args.validation_action_chunk_steps,
            wandb_commit=False,
            ema_policy=ema_model.module if ema_model is not None else None,
        )
        if accelerator.is_main_process:
            _write_json_atomic(
                output_dir / "validation_state.json",
                {"global_step": global_step, "metrics": validation_logs},
            )
            _write_json_atomic(
                output_dir / "validation_state_0.json",
                {"global_step": global_step, "metrics": validation_logs},
            )
            logger.info(
                "Initial validation at step 0: %s",
                json.dumps(validation_logs, sort_keys=True),
            )
    if global_step >= args.iters:
        validation_step = _read_step_marker(output_dir / "validation_state.json")
        if validation_dataloaders:
            validation_logs = None
            if validation_step >= args.iters:
                try:
                    validation_state = json.loads(
                        (output_dir / "validation_state.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    candidate_logs = validation_state.get("metrics")
                    required_metrics = required_validation_metrics(
                        validation_dataloaders,
                        args.validation_action_chunk_mse,
                    )
                    if (
                        isinstance(candidate_logs, dict)
                        and required_metrics.issubset(candidate_logs)
                        and all(
                            math.isfinite(float(value))
                            for value in candidate_logs.values()
                        )
                    ):
                        validation_logs = candidate_logs
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    validation_logs = None
            if validation_logs is None:
                validation_model = (
                    ema_model.module
                    if args.validation_use_ema and ema_model is not None
                    else model
                )
                validation_logs = run_validation(
                    accelerator,
                    validation_model,
                    processor,
                    validation_dataloaders,
                    args.iters,
                    args.validation_noise_samples,
                    args.validation_seed,
                    args.validation_max_batches,
                    args.validation_action_chunk_mse,
                    args.validation_action_chunk_samples,
                    args.validation_action_chunk_steps,
                    ema_policy=ema_model.module if ema_model is not None else None,
                )
                if accelerator.is_main_process:
                    _write_json_atomic(
                        output_dir / "validation_state.json",
                        {"global_step": args.iters, "metrics": validation_logs},
                    )
                    _write_json_atomic(
                        output_dir
                        / f"validation_state_{args.iters}.json",
                        {"global_step": args.iters, "metrics": validation_logs},
                    )
                    logger.info(
                        "Recovered missing final validation at step %d: %s",
                        args.iters,
                        json.dumps(validation_logs, sort_keys=True),
                    )
            else:
                accelerator.log(
                    validation_logs,
                    step=args.iters,
                    log_kwargs={"wandb": {"commit": True}},
                )
                if accelerator.is_main_process:
                    logger.info(
                        "Re-emitted final validation at step %d: %s",
                        args.iters,
                        json.dumps(validation_logs, sort_keys=True),
                    )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _write_json_atomic(
                output_dir / "training_complete.json",
                {
                    "global_step": args.iters,
                    "validation_step": (
                        args.iters if validation_dataloaders else None
                    ),
                },
            )
        accelerator.end_training()
        logger.info(
            "Checkpoint and final validation reached iters=%d; training is complete.",
            args.iters,
        )
        return
    trace_steps = int(os.environ.get("LAMBDA0_TRACE_STEPS", "0"))
    if trace_steps < 0:
        raise ValueError("LAMBDA0_TRACE_STEPS must be non-negative.")
    data_iterator = iter(train_dataloader)
    training_metric_blocks = {
        f"loss/{spec.name}/{block.name}": (spec, block)
        for spec in policy.action_space.specs.values()
        for block in spec.blocks
    }
    training_metric_names = ("loss", *training_metric_blocks)
    training_metric_name_set = set(training_metric_names)
    while global_step < args.iters:
        step_started = time.monotonic()
        if global_step < trace_steps and accelerator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        update_group_lrs(optimizer, global_step, args)
        optimizer.zero_grad(set_to_none=True)
        accumulated_log_sums: dict[str, torch.Tensor] = {}
        accumulated_log_counts: dict[str, torch.Tensor] = {}
        accumulation_steps = args.gradient_accumulation_steps
        for micro_step in range(accumulation_steps):
            batch = next(data_iterator)
            phase_prefix = f"micro_{micro_step}"
            trace_training_phase(
                accelerator,
                global_step,
                trace_steps,
                f"{phase_prefix}_batch_ready",
                step_started,
                batch,
            )
            inputs = prepare_inputs(
                batch,
                processor,
                accelerator.device,
                policy,
            )
            trace_training_phase(
                accelerator,
                global_step,
                trace_steps,
                f"{phase_prefix}_inputs_ready",
                step_started,
                inputs,
            )
            sync_context = (
                nullcontext()
                if micro_step + 1 == accumulation_steps
                else accelerator.no_sync(model)
            )
            with sync_context:
                loss_dict = model(
                    **inputs,
                    rtc_max_delay=args.rtc_max_delay,
                    action_expert_noise_samples=(
                        args.action_expert_noise_samples
                    ),
                )
                loss = (
                    loss_dict["loss"]
                    if "loss" in loss_dict
                    else sum(loss_dict.values())
                )
                accelerator.backward(loss / accumulation_steps)
            for name, value in loss_dict.items():
                if name not in training_metric_name_set:
                    raise KeyError(f"Unknown training metric: {name}")
                if name == "loss":
                    sample_count = torch.tensor(
                        inputs["action"].shape[0]
                        * args.action_expert_noise_samples,
                        device=value.device,
                        dtype=torch.float32,
                    )
                else:
                    spec, block = training_metric_blocks[name]
                    ids = inputs["action_space_id"].reshape(-1)
                    selected = inputs["action_valid_mask"][ids == spec.id]
                    sample_count = selected[
                        :, :, block.start : block.end
                    ].any(dim=(1, 2)).sum().to(dtype=torch.float32)
                    sample_count *= args.action_expert_noise_samples
                contribution = value.detach().float() * sample_count
                accumulated_log_sums[name] = (
                    accumulated_log_sums.get(
                        name,
                        torch.zeros_like(contribution),
                    )
                    + contribution
                )
                accumulated_log_counts[name] = (
                    accumulated_log_counts.get(
                        name,
                        torch.zeros_like(sample_count),
                    )
                    + sample_count
                )
            trace_training_phase(
                accelerator,
                global_step,
                trace_steps,
                f"{phase_prefix}_backward_done",
                step_started,
            )
        trace_training_phase(
            accelerator,
            global_step,
            trace_steps,
            "accumulation_done",
            step_started,
        )
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if (
            ema_model is not None
            and (global_step + 1) % args.ema_update_interval == 0
        ):
            ema_model.update_parameters(accelerator.unwrap_model(model))
        optimizer.zero_grad(set_to_none=True)
        trace_training_phase(
            accelerator,
            global_step,
            trace_steps,
            "optimizer_done",
            step_started,
        )

        if global_step % args.log_interval == 0:
            zero = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            local_sums = torch.stack(
                [accumulated_log_sums.get(name, zero) for name in training_metric_names]
            )
            local_counts = torch.stack(
                [
                    accumulated_log_counts.get(name, zero)
                    for name in training_metric_names
                ]
            )
            reduced = accelerator.reduce(
                torch.cat((local_sums, local_counts)),
                reduction="sum",
            )
            metric_count = len(training_metric_names)
            global_sums = reduced[:metric_count]
            global_counts = reduced[metric_count:]
            logs = {
                name: (global_sums[index] / global_counts[index]).item()
                for index, name in enumerate(training_metric_names)
                if global_counts[index].item() > 0
            }
            logs["loss_total"] = (
                logs["loss"]
                if "loss" in logs
                else sum(logs.values())
            )
            logs.update(
                {f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups}
            )
            accelerator.log(
                logs,
                step=global_step,
                log_kwargs={"wandb": {"commit": True}},
            )
            if accelerator.is_main_process:
                elapsed = time.time() - last_log_time
                interval = 1 if global_step == 0 else args.log_interval
                last_log_time = time.time()
                logger.info(
                    "[%d/%d] loss=%.4f (%.2fs/it)",
                    global_step,
                    args.iters,
                    logs["loss_total"],
                    elapsed / interval,
                )

        global_step += 1
        should_save = global_step == args.iters or global_step % args.save_interval == 0
        if should_save:
            save_training_checkpoint(
                accelerator,
                model,
                ema_model,
                processor,
                output_dir,
                global_step,
                args,
                logger,
            )
        should_validate = bool(validation_dataloaders) and (
            global_step == args.iters
            or global_step % args.validation_interval == 0
        )
        if should_validate:
            validation_model = (
                ema_model.module
                if args.validation_use_ema and ema_model is not None
                else model
            )
            validation_logs = run_validation(
                accelerator,
                validation_model,
                processor,
                validation_dataloaders,
                global_step,
                args.validation_noise_samples,
                args.validation_seed,
                args.validation_max_batches,
                args.validation_action_chunk_mse,
                args.validation_action_chunk_samples,
                args.validation_action_chunk_steps,
                wandb_commit=(global_step == args.iters),
                ema_policy=ema_model.module if ema_model is not None else None,
            )
            if accelerator.is_main_process:
                _write_json_atomic(
                    output_dir / "validation_state.json",
                    {"global_step": global_step, "metrics": validation_logs},
                )
                _write_json_atomic(
                    output_dir / f"validation_state_{global_step}.json",
                    {"global_step": global_step, "metrics": validation_logs},
                )
                logger.info(
                    "Validation at step %d: %s",
                    global_step,
                    json.dumps(validation_logs, sort_keys=True),
                )
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if accelerator.is_main_process:
        _write_json_atomic(
            output_dir / "training_complete.json",
            {
                "global_step": global_step,
                "validation_step": (
                    _read_step_marker(output_dir / "validation_state.json")
                    if validation_dataloaders
                    else None
                ),
            },
        )


def main(args) -> None:
    if not args.release_data_manifest:
        raise ValueError("Use python -m lambda0 train with a prepared data bundle.")
    release_manifest = json.loads(Path(args.release_data_manifest).read_text())
    if release_manifest.get("contract") != "lambda0_velocity_v1":
        raise ValueError("Prepared data contract mismatch.")
    if args.ema_update_interval != 1:
        raise ValueError("Release EMA must update every optimizer step.")
    args.report_to = tuple(dict.fromkeys(args.report_to))
    if not args.report_to:
        raise ValueError("--report_to must contain at least one backend.")
    (
        args.state_conditioning_mode,
        args.state_discrete_bins,
    ) = validate_state_conditioning(
        args.state_conditioning_mode,
        args.state_discrete_bins,
    )
    if args.state_conditioning_mode == DISCRETE_VLM and not args.use_proprio:
        raise ValueError(
            "--state_conditioning_mode discrete_vlm requires --use_proprio."
        )
    if "wandb" not in args.report_to and any(
        value is not None
        for value in (
            args.wandb_entity,
            args.wandb_run_name,
            args.wandb_group,
            args.wandb_tags,
        )
    ):
        raise ValueError("W&B run options require '--report_to wandb'.")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("--ema_decay must be in [0,1).")
    if not 0 <= args.rtc_max_delay < args.num_actions:
        raise ValueError(
            "--rtc_max_delay must be in [0,num_actions) so every sample "
            "retains a supervised suffix."
        )
    if args.action_expert_noise_samples <= 0:
        raise ValueError("--action_expert_noise_samples must be positive.")
    if args.validation_metas_path and args.validation_interval <= 0:
        raise ValueError(
            "--validation_interval must be positive when validation is enabled."
        )
    if args.validation_interval < 0:
        raise ValueError("--validation_interval must be non-negative.")
    if args.validation_samples_per_episode <= 0:
        raise ValueError("--validation_samples_per_episode must be positive.")
    if args.validation_noise_samples <= 0:
        raise ValueError("--validation_noise_samples must be positive.")
    if args.validation_max_batches < 0:
        raise ValueError("--validation_max_batches must be non-negative.")
    if args.validation_action_chunk_samples <= 0:
        raise ValueError("--validation_action_chunk_samples must be positive.")
    if args.validation_action_chunk_steps <= 0:
        raise ValueError("--validation_action_chunk_steps must be positive.")
    if (
        args.validation_action_chunk_mse or args.validation_at_start
    ) and not args.validation_metas_path:
        raise ValueError(
            "Action-chunk or step-zero validation requires validation metadata."
        )
    if args.validation_use_ema and args.ema_decay == 0:
        raise ValueError("--validation_use_ema requires --ema_decay > 0.")
    if args.save_interval <= 0:
        raise ValueError("--save_interval must be positive.")
    if args.ema_update_interval <= 0:
        raise ValueError("--ema_update_interval must be positive.")
    if args.ddp_bucket_cap_mb <= 0:
        raise ValueError("--ddp_bucket_cap_mb must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive.")
    if args.ddp_static_graph and args.gradient_accumulation_steps > 1:
        raise ValueError(
            "--ddp_static_graph is incompatible with gradient accumulation via "
            "DDP no_sync; disable the static graph for accumulated training."
        )
    if args.ddp_static_graph and args.ddp_find_unused_parameters is True:
        raise ValueError(
            "--ddp_static_graph cannot enable --ddp_find_unused_parameters."
        )
    if args.ema_decay > 0:
        if args.save_interval % args.ema_update_interval:
            raise ValueError(
                "--save_interval must be divisible by --ema_update_interval."
            )
        if args.iters % args.ema_update_interval:
            raise ValueError("--iters must be divisible by --ema_update_interval.")
    # Initialize newly added action-space heads identically across scaling runs.
    set_seed(args.seed, args.deterministic)
    model, processor = load_model_and_processor(args)
    run_training(args, model, processor)
