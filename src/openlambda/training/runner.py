"""Resolve a portable experiment and launch training (or print a read-only preflight)."""
import argparse
import json
import math
import os
from importlib.resources import files
from pathlib import Path

from ..runtime.paths import model_reference, resolve_path


def batch_layout(global_batch, micro_batch, world_size):
    for name, value in (("global batch", global_batch), ("micro batch", micro_batch), ("world size", world_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    denominator = micro_batch * world_size
    if global_batch % denominator:
        raise ValueError(f"Global batch {global_batch} must be divisible by micro batch {micro_batch} × world size {world_size}.")
    return global_batch // denominator


def resolve_pretrain(cli, config, cfg_path):
    """Resolve Stage I separately from the action-expert training contract."""
    allowed = {"stage", "data", "data_root", "model", "model_revision", "output",
               "seed", "batch_size", "global_batch_size", "num_workers", "resume", "training"}
    if set(config) - allowed:
        raise ValueError(f"Unknown Stage-I experiment keys: {sorted(set(config) - allowed)}")
    base = cfg_path.parent if cfg_path else Path.cwd()
    def option(name, default=None):
        value = getattr(cli, name, None)
        return config.get(name, default) if value is None else value
    def path_option(name):
        value = option(name)
        if not value:
            raise ValueError(f"Missing --{name.replace('_', '-')} or config.{name}")
        return resolve_path(value, Path.cwd() if getattr(cli, name, None) is not None else base)
    profile = json.loads(files("lambda0").joinpath("recipes/pretrain.json").read_text())
    settings = profile["training"]
    overrides = config.get("training", {})
    allowed_overrides = {"iters", "learning_rate", "vlm_lr_coef", "weight_decay", "betas",
                         "warmup_steps", "min_lr_ratio", "max_grad_norm", "ema_decay",
                         "save_interval", "log_interval", "mixed_precision", "gradient_checkpointing",
                         "max_sequence_tokens", "image_max_pixels"}
    if not isinstance(overrides, dict) or set(overrides) - allowed_overrides:
        raise ValueError("Unknown Stage-I training override.")
    settings.update(overrides)
    for key in ("iters", "warmup_steps", "save_interval", "log_interval", "max_sequence_tokens"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "warmup_steps" else 1):
            raise ValueError(f"Stage-I training.{key} has an invalid value.")
    if settings["warmup_steps"] >= settings["iters"]:
        raise ValueError("Stage-I warmup_steps must be smaller than iters.")
    for key in ("learning_rate", "vlm_lr_coef", "max_grad_norm", "ema_decay"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Stage-I training.{key} must be positive and finite.")
    if settings["ema_decay"] >= 1:
        raise ValueError("Stage-I ema_decay must be below 1.")
    if settings["mixed_precision"] not in ("no", "fp16", "bf16"):
        raise ValueError("Unsupported Stage-I mixed_precision.")
    for key in ("weight_decay", "min_lr_ratio"):
        value = settings[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key == "min_lr_ratio" and value > 1)):
            raise ValueError(f"Stage-I training.{key} must be finite and nonnegative (ratio <= 1).")
    betas = settings["betas"]
    if not isinstance(betas, list) or len(betas) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or not 0 <= value < 1 for value in betas
    ):
        raise ValueError("Stage-I training.betas must contain two numbers in [0,1).")
    if not isinstance(settings["gradient_checkpointing"], bool):
        raise ValueError("Stage-I gradient_checkpointing must be boolean.")
    if settings.get("image_max_pixels") is not None and (isinstance(settings["image_max_pixels"], bool) or not isinstance(settings["image_max_pixels"], int) or settings["image_max_pixels"] < 1):
        raise ValueError("Stage-I image_max_pixels must be a positive integer.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batch = option("batch_size", settings["batch_size"])
    global_batch = option("global_batch_size", profile.get("global_batch_size", profile["world_size"] * profile["training"]["batch_size"]))
    settings["gradient_accumulation_steps"] = batch_layout(global_batch, micro_batch, world_size)
    settings["batch_size"] = micro_batch
    settings["seed"] = option("seed", 0)
    settings["num_workers"] = option("num_workers", settings["num_workers"])
    for key in ("seed", "num_workers"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Stage-I {key} must be a nonnegative integer.")
    data, data_root, output = path_option("data"), path_option("data_root"), path_option("output")
    package_root = Path(__file__).resolve().parents[1]
    if output.is_relative_to(package_root) or any(output == p or p.is_relative_to(output) for p in (data, data_root)):
        raise ValueError("Stage-I output must be separate from package, source data, and bundle.")
    model_value = option("model")
    if not model_value:
        raise ValueError("Stage I requires an official Qwen3.5 model.")
    model = model_reference(model_value, Path.cwd() if cli.model is not None else base)
    revision = option("model_revision")
    model_path = Path(model)
    if not model_path.is_dir() and not revision:
        raise ValueError("Remote Stage-I model IDs require --model-revision.")
    if model_path.is_dir():
        model_config = json.loads((model_path / "config.json").read_text())
        if model_config.get("model_type") != "qwen3_5":
            raise ValueError("Stage I requires a Qwen3.5 VLM checkpoint, not a Lambda0 action expert.")
    manifest = json.loads((data / "manifest.json").read_text())
    sampling = manifest.get("sampling")
    if not sampling or sampling.get("strategy") not in ("hierarchical_iid", "hierarchical_epoch"):
        raise ValueError("Stage I requires a prepared, explicit dataset sampling mixture.")
    from ..data.bundle import verify_bundle
    verify_bundle(data, data_root, sampling)
    for path in (data / "train").glob("*.json"):
        meta = json.loads(path.read_text())
        if meta.get("action_space") != "human_hand_geometry" or meta.get("handler") != "lambda0.data.domain_handler.hand_state_episode:HandStateEpisodeHandler":
            raise ValueError("Stage I requires 138D hand-state metadata, not action data.")
    resume = option("resume", "none")
    if resume not in ("auto", "none"):
        resume = str(resolve_path(resume, Path.cwd() if cli.resume is not None else base))
    if output.exists() and any(output.iterdir()) and resume == "none":
        raise FileExistsError("Stage-I output is not empty; select a new output or --resume auto.")
    settings.update(models=model, model_revision=revision, output_dir=str(output),
                    train_metas_path=str(data / "train"), data_root=str(data_root),
                    sampling_plan_json=json.dumps(sampling), release_data_manifest=str(data / "manifest.json"),
                    resume_from_checkpoint=resume)
    return {"stage": "pretrain", "world_size": world_size,
            "recommended_world_size": profile["world_size"],
            "global_batch_size": global_batch, "training": settings}


def validate_settings(settings):
    """Fail on malformed JSON before allocating the model."""
    for key in ("image_max_pixels", "iters", "save_interval", "log_interval", "validation_interval", "num_domains",
                "validation_samples_per_episode", "validation_action_chunk_steps"):
        if key in settings:
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"training.{key} must be a positive integer.")
    for key in ("seed", "num_workers", "warmup_steps", "new_module_warmup_steps", "validation_max_batches"):
        if key in settings:
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"training.{key} must be a nonnegative integer.")
    for key in ("learning_rate", "vlm_lr_coef", "conditioning_lr_coef", "weight_decay", "max_grad_norm", "min_lr_ratio"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be a finite nonnegative number.")
    if settings["min_lr_ratio"] > 1:
        raise ValueError("min_lr_ratio must be <= 1.")
    for key in ("gradient_checkpointing", "action_expert_gradient_checkpointing", "require_fast_linear_attention",
                "fused_optimizer", "ddp_find_unused_parameters", "allow_world_size_change", "deterministic"):
        if key in settings and not isinstance(settings[key], bool):
            raise ValueError(f"training.{key} must be a JSON boolean.")
    betas = settings["betas"]
    if not isinstance(betas, list) or len(betas) != 2 or any(isinstance(v, bool) or not isinstance(v, (float,int)) or not 0 <= v < 1 for v in betas):
        raise ValueError("training.betas must contain two numbers in [0,1).")
    if settings.get("mixed_precision", "bf16") not in ("no", "fp16", "bf16"):
        raise ValueError("Unsupported mixed_precision.")
    if settings.get("dtype", "auto") not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("Unsupported dtype.")


def resolve_experiment(cli):
    cfg_path = resolve_path(cli.config) if cli.config else None
    config = json.loads(cfg_path.read_text()) if cfg_path else {}
    allowed = {"stage", "data", "data_root", "model", "model_revision", "output", "rtc", "seed",
               "batch_size", "global_batch_size", "num_workers", "resume", "training",
               "prediction_type", "initialization", "midtrain_weight_source",
               "gradient_accumulation_steps"}
    if set(config) - allowed:
        raise ValueError(f"Unknown experiment keys: {sorted(set(config) - allowed)}")
    base = cfg_path.parent if cfg_path else Path.cwd()
    def option(name, default=None):
        value = getattr(cli, name, None)
        return config.get(name, default) if value is None else value
    def path_option(name):
        value = option(name)
        if not value:
            raise ValueError(f"Missing --{name.replace('_', '-')} or config.{name}")
        return resolve_path(value, Path.cwd() if getattr(cli, name, None) is not None else base)
    stage = option("stage")
    if stage == "pretrain":
        return resolve_pretrain(cli, config, cfg_path)
    if stage not in ("midtrain", "posttrain"):
        raise ValueError("Specify stage: pretrain, midtrain, or posttrain.")
    profile = json.loads(files("lambda0").joinpath(f"recipes/{stage}.json").read_text())
    settings = profile["training"]
    overrides = config.get("training", {})
    # Algorithmic contracts remain fixed; optimizer/system knobs are explicit.
    allowed_overrides = {"image_max_pixels", "learning_rate", "vlm_lr_coef", "conditioning_lr_coef", "weight_decay", "betas",
                         "max_grad_norm", "iters", "warmup_steps", "new_module_warmup_steps", "min_lr_ratio",
                         "validation_interval", "validation_samples_per_episode", "validation_max_batches",
                         "validation_action_chunk_steps", "save_interval", "log_interval", "num_workers",
                         "gradient_checkpointing", "action_expert_gradient_checkpointing", "mixed_precision",
                         "dtype", "require_fast_linear_attention", "fused_optimizer", "num_domains",
                         "state_conditioning_mode", "state_discrete_bins", "reset_action_space_ids",
                         "trainable_action_space_ids", "ddp_find_unused_parameters", "allow_world_size_change",
                         "deterministic", "report_to", "wandb_project", "wandb_entity", "wandb_run_name", "wandb_group", "wandb_tags"}
    if not isinstance(overrides, dict) or set(overrides) - allowed_overrides:
        raise ValueError("Unknown/contract-changing training override.")
    settings.update(overrides)
    prediction_type = option("prediction_type", "velocity")
    if prediction_type not in ("x0", "velocity"):
        raise ValueError("prediction_type must be x0 or velocity.")
    settings["prediction_type"] = prediction_type
    initialization = option("initialization", "midtrain" if stage == "posttrain" else "auto")
    if initialization not in ("auto", "vlm", "midtrain"):
        raise ValueError("initialization must be auto, vlm, or midtrain.")
    if stage == "posttrain" and initialization == "auto":
        raise ValueError("Posttrain initialization must explicitly be vlm or midtrain.")
    settings["initialization"] = initialization
    midtrain_weight_source = option("midtrain_weight_source", "ema")
    if midtrain_weight_source not in ("ema", "raw"):
        raise ValueError("midtrain_weight_source must be ema or raw.")
    if midtrain_weight_source != "ema" and initialization != "midtrain":
        raise ValueError("Raw midtrain weights require initialization=midtrain.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = option("global_batch_size", profile["world_size"] * profile["training"]["batch_size"])
    micro_batch = option("batch_size", settings["batch_size"])
    settings["batch_size"] = micro_batch
    settings["gradient_accumulation_steps"] = batch_layout(global_batch, micro_batch, world_size)
    required_accumulation = option("gradient_accumulation_steps")
    if required_accumulation is not None:
        if isinstance(required_accumulation, bool) or not isinstance(required_accumulation, int) or required_accumulation < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        if required_accumulation != settings["gradient_accumulation_steps"]:
            raise ValueError("GPU count/batch settings disagree with explicit gradient_accumulation_steps; no automatic accumulation change allowed.")
    settings["seed"] = option("seed", 0)
    settings["num_workers"] = option("num_workers", settings.get("num_workers", 4))
    rtc = option("rtc", True)
    if not isinstance(rtc, bool):
        raise ValueError("rtc must be a JSON boolean.")
    settings["rtc_max_delay"] = 8 if rtc else 0
    validate_settings(settings)
    data, data_root, output = path_option("data"), path_option("data_root"), path_option("output")
    package_root = Path(__file__).resolve().parents[1]
    if output.is_relative_to(package_root):
        raise ValueError("Output must not be inside the installed package.")
    for protected in (data, data_root):
        if output == protected or protected.is_relative_to(output):
            raise ValueError("Output must not contain the source data or bundle.")
    raw_model = option("model")
    if not raw_model:
        raise ValueError("Specify --model or config.model.")
    model = model_reference(raw_model, Path.cwd() if cli.model is not None else base)
    revision = option("model_revision")
    model_path = Path(model)
    if not model_path.is_dir() and not revision:
        raise ValueError("Remote model IDs require --model-revision; use an immutable version for reproducibility.")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model_config = json.loads((model_path / "config.json").read_text()) if model_path.is_dir() else {}
    if initialization == "vlm" and model_config and model_config.get("model_type") != "qwen3_5":
        raise ValueError("VLM initialization requires official Qwen3.5 weights, not Lambda0 weights.")
    if model_config.get("model_type") == "lambda0":
        if model_config.get("training_contract") != f"lambda0_{prediction_type}_v1" or model_config.get("prediction_type", "velocity") != prediction_type:
            raise ValueError("Checkpoint prediction_type/contract mismatch.")
    if stage == "midtrain":
        sampling = manifest.get("sampling")
        if not sampling:
            raise ValueError("Midtrain requires an explicitly imported sampling recipe.")
    else:
        if initialization == "midtrain":
            expected_contract = f"lambda0_{prediction_type}_v1"
            if model_config.get("training_contract") != expected_contract:
                raise ValueError("Midtrain initialization requires a matching release checkpoint.")
            if midtrain_weight_source == "ema":
                if model_path.name != "ema":
                    raise ValueError("EMA midtrain initialization requires the checkpoint's ema directory.")
                provenance_root = model_path.parent
            else:
                if model_path.name == "ema":
                    raise ValueError("Raw midtrain initialization requires the published checkpoint root.")
                provenance_root = model_path
            parent_config = json.loads((provenance_root / "training_config.json").read_text())
            if parent_config.get("release_stage") != "midtrain":
                raise ValueError("Initialization was not produced by midtraining.")
        sampling = manifest.get("sampling") or {"strategy": "frame_uniform"}
    from ..data.bundle import verify_bundle
    verify_bundle(data, data_root, sampling)
    validation = bool(list((data / "validation").glob("*.json")))
    resume = option("resume", "none")  # Never silently resume an unrelated output.
    if resume not in ("auto", "none"):
        resume = str(resolve_path(resume, Path.cwd() if cli.resume is not None else base))
    if output.exists() and any(output.iterdir()) and resume == "none":
        raise FileExistsError("Output is not empty; explicitly choose --resume auto or a new output.")
    settings.update(models=model, model_revision=revision, output_dir=str(output), release_stage=stage,
                    train_metas_path=str(data / "train"), data_root=str(data_root),
                    sampling_plan_json=json.dumps(sampling), release_data_manifest=str(manifest_path),
                    validation_metas_path=[f"holdout={data / 'validation'}"] if validation else [],
                    validation_action_chunk_mse=validation, resume_from_checkpoint=resume)
    return {"stage": stage, "world_size": world_size, "recommended_world_size": profile["world_size"],
            "global_batch_size": global_batch, "training": settings}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lambda0 train", description=__doc__)
    parser.add_argument("--config", type=Path, help="Experiment JSON; its relative paths are relative to this file.")
    parser.add_argument("--stage", choices=("pretrain", "midtrain", "posttrain"))
    for name in ("data", "data-root", "model", "model-revision", "output", "resume"):
        parser.add_argument("--" + name)
    for name in ("batch-size", "global-batch-size", "gradient-accumulation-steps", "seed", "num-workers"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prediction-type", choices=("x0", "velocity"))
    parser.add_argument("--initialization", choices=("vlm", "midtrain", "auto"))
    parser.add_argument("--check-only", action="store_true")
    cli = parser.parse_args(argv)
    resolved = resolve_experiment(cli)
    if cli.check_only:
        print(json.dumps(resolved, indent=2))
        return
    if resolved["stage"] == "pretrain":
        from .pretrain import run
        run(resolved["training"])
        return
    from . import engine as training
    settings = resolved["training"]
    args = training.get_args_parser().parse_args(["--models", settings["models"], "--train_metas_path", settings["train_metas_path"]])
    for key, value in settings.items():
        setattr(args, key, value)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    training.main(args)
