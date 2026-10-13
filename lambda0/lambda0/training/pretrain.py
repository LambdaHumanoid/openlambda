"""VLM-only future hand-state token training for Stage I.

Uses pi0.5-style scalar bins and a masked autoregressive language loss. The
continuous action expert is not constructed in this stage.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from uuid import uuid4
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from lambda0.data import create_dataloader
from lambda0.models.hand_state_tokens import HAND_STATE_HORIZON
from lambda0.training.pretrain_encoding import encode_batch
from lambda0.runtime.environment import environment


@torch.no_grad()
def _update_ema(ema, model, decay: float) -> None:
    for averaged, online in zip(ema.parameters(), model.parameters(), strict=True):
        averaged.lerp_(online.detach().to(device="cpu", dtype=torch.float32), 1 - decay)
    for averaged, online in zip(ema.buffers(), model.buffers(), strict=True):
        averaged.copy_(online.detach().to(device="cpu", dtype=averaged.dtype))


def _schedule(step: int, *, warmup: int, total: int, minimum: float) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def _save(accelerator, model, ema, processor, output: Path, step: int, settings: dict) -> None:
    checkpoint = output / f"ckpt-{step}"
    temporary = output / f".ckpt-{step}.incomplete"
    if accelerator.is_main_process:
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        if temporary.exists():
            os.replace(temporary, output / f".ckpt-{step}.abandoned-{uuid4().hex}")
        temporary.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(temporary / "trainer"))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(temporary / "online", safe_serialization=True)
        ema.save_pretrained(temporary / "ema", safe_serialization=True)
        processor.save_pretrained(temporary / "online")
        processor.save_pretrained(temporary / "ema")
        (temporary / "state.json").write_text(json.dumps({
            "stage": "pretrain", "step": step, "seed": settings["seed"],
            "data_manifest": settings["release_data_manifest"],
        }, indent=2))
        os.replace(temporary, checkpoint)
        marker = output / f".last-{uuid4().hex}.json"
        marker.write_text(json.dumps({"checkpoint": checkpoint.name}))
        os.replace(marker, output / "last.json")
    accelerator.wait_for_everyone()


def run(settings: dict) -> None:
    """Train an official Qwen3.5 VLM; EMA directories initialize Stage II."""
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        mixed_precision=settings.get("mixed_precision", "bf16"),
    )
    seed = settings["seed"] + accelerator.process_index
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resume = settings["resume_from_checkpoint"]
    if resume == "auto":
        resume = output / json.loads((output / "last.json").read_text())["checkpoint"]
    elif resume == "none":
        resume = None
    else:
        resume = Path(resume)
    if resume is not None and json.loads((resume / "state.json").read_text())["stage"] != "pretrain":
        raise ValueError("Stage I can resume only a Stage-I checkpoint.")
    model_source = settings["models"]
    load_options = {"revision": settings["model_revision"]} if settings["model_revision"] else {}
    processor = AutoProcessor.from_pretrained(model_source, **load_options)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_source, **load_options)
    ema = (Qwen3_5ForConditionalGeneration.from_pretrained(resume / "ema", dtype=torch.float32)
           if resume is not None and accelerator.is_main_process else
           copy.deepcopy(model).float().cpu() if accelerator.is_main_process else None)
    if ema is not None:
        ema.requires_grad_(False).eval()
    if settings.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    optimizer = AdamW(model.parameters(), lr=settings["learning_rate"] * settings["vlm_lr_coef"],
                      weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]))
    sampling = json.loads(settings["sampling_plan_json"])
    loader = create_dataloader(
        batch_size=settings["batch_size"], metas_path=settings["train_metas_path"],
        num_actions=HAND_STATE_HORIZON, training=True, num_workers=settings["num_workers"],
        sampling_plan=sampling, sampling_seed=settings["seed"],
        shard_index=accelerator.process_index, num_shards=accelerator.num_processes,
        data_root=settings["data_root"], image_max_pixels=settings.get("image_max_pixels"),
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(
        step, warmup=settings["warmup_steps"], total=settings["iters"],
        minimum=settings["min_lr_ratio"]
    ))
    accelerator.register_for_checkpointing(scheduler)
    step = 0
    if resume is not None:
        accelerator.load_state(str(resume / "trainer"))
        step = int(json.loads((resume / "state.json").read_text())["step"])
    if accelerator.is_main_process:
        (output / "resolved_config.json").write_text(json.dumps(settings, indent=2))
        (output / "environment.json").write_text(json.dumps(environment(), indent=2))
    accelerator.wait_for_everyone()
    iterator = iter(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < settings["iters"]:
        batch = next(iterator)
        inputs = {key: value.to(accelerator.device) for key, value in encode_batch(
            batch, processor, settings["max_sequence_tokens"]
        ).items()}
        with accelerator.accumulate(model):
            loss = model(**inputs).loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if not accelerator.sync_gradients:
            continue
        scheduler.step()
        step += 1
        if accelerator.is_main_process:
            _update_ema(ema, accelerator.unwrap_model(model), settings["ema_decay"])
            if step % settings["log_interval"] == 0:
                print(f"stage=pretrain step={step} loss={loss.detach().float().item():.5f}", flush=True)
        if step % settings["save_interval"] == 0 or step == settings["iters"]:
            _save(accelerator, model, ema, processor, output, step, settings)
    accelerator.end_training()


__all__ = ["run"]
