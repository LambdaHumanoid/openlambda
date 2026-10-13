from __future__ import annotations

import argparse
import random
import json
import logging
import os
import threading
import traceback
from base64 import b64decode, b64encode
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from PIL import Image

from lambda0.data.normalization import DatasetNormalizer
from lambda0.inference.rtc import RtcSessionStore
from lambda0.models import Lambda0, Lambda0Processor
from lambda0.models.state_conditioning import (
    DEFAULT_STATE_DISCRETE_BINS,
    DISCRETE_VLM,
    validate_state_conditioning,
)


SIMPLE_IMAGE_KEY = "rgb_head_stereo_left"
SIMPLE_STATE_KEY = "states"
SIMPLE_ACTION_SPACE = "robot_g1_simple_sim"
SIMPLE_COORDINATE_CONTRACT = "psi_simple_absolute_v1"
QUANTILE_NORMALIZATION_CONTRACT = "quantile_q01_q99_v1"
LOGGER = logging.getLogger("uvicorn.error")


def decode_numpy_envelope(value: Any) -> Any:
    """Decode the NumPy JSON envelope used by the SIMPLE HTTP client."""
    if not isinstance(value, dict) or "__numpy__" not in value:
        return value
    if set(value) != {"__numpy__", "dtype", "shape"}:
        raise ValueError("Malformed SIMPLE NumPy envelope.")
    dtype = descr_to_dtype(value["dtype"])
    if dtype.hasobject:
        raise ValueError("Object dtypes are not accepted.")
    shape = tuple(int(size) for size in value["shape"])
    if any(size < 0 for size in shape):
        raise ValueError("NumPy envelope shape must be non-negative.")
    raw = b64decode(value["__numpy__"], validate=True)
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError(
            f"NumPy envelope contains {len(raw)} bytes, expected {expected_bytes}."
        )
    array = np.frombuffer(raw, dtype=dtype)
    return array.reshape(shape) if shape else array[0]


def encode_numpy_envelope(value: np.ndarray) -> dict[str, Any]:
    """Encode an array exactly as SIMPLE's HttpActionClient expects."""
    array = np.ascontiguousarray(value)
    return {
        "__numpy__": b64encode(array.tobytes()).decode("ascii"),
        "dtype": dtype_to_descr(array.dtype),
        "shape": list(array.shape),
    }


def resolve_dtype(name: str, device: torch.device):
    if name == "auto":
        return torch.float32 if device.type == "cpu" else "auto"
    return getattr(torch, name)


def decode_image(value: Any) -> Image.Image:
    value = decode_numpy_envelope(value)
    if isinstance(value, str):
        value = json_numpy.loads(value)
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Unsupported image payload type: {type(value).__name__}.")
    if value.ndim == 1:
        decoded = cv2.imdecode(value.astype(np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("OpenCV could not decode the image bytes.")
        value = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if value.ndim not in (2, 3):
        raise ValueError(f"Image must have 2 or 3 dimensions, got {value.ndim}.")
    return Image.fromarray(value.astype(np.uint8)).convert("RGB")


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device):
    return {
        key: value.to(device=device, non_blocking=True) for key, value in inputs.items()
    }


def parse_request(
    payload: dict[str, Any],
    *,
    default_embodiment_id: int | None,
    default_action_space_id: int | None,
) -> tuple[list[Image.Image], str, Any, int, int, bool]:
    is_simple = "image" in payload or "state" in payload
    if is_simple:
        image_values = payload.get("image")
        state_values = payload.get("state")
        if not isinstance(image_values, dict) or not isinstance(state_values, dict):
            raise ValueError("SIMPLE requests require image and state objects.")
        if SIMPLE_IMAGE_KEY not in image_values:
            raise ValueError(f"SIMPLE image is missing {SIMPLE_IMAGE_KEY!r}.")
        if SIMPLE_STATE_KEY not in state_values:
            raise ValueError(f"SIMPLE state is missing {SIMPLE_STATE_KEY!r}.")
        instruction = payload.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("SIMPLE instruction must be a non-empty string.")
        images = [decode_image(image_values[SIMPLE_IMAGE_KEY])]
        proprio_value = decode_numpy_envelope(state_values[SIMPLE_STATE_KEY])
    else:
        images = [
            decode_image(payload[key])
            for key in ("image0", "image1", "image2")
            if key in payload
        ]
        instruction = payload.get("language_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("language_instruction must be a non-empty string.")
        if "proprio" not in payload:
            raise ValueError("proprio is required.")
        proprio_value = payload["proprio"]
        if isinstance(proprio_value, str):
            proprio_value = json_numpy.loads(proprio_value)
    if not images:
        raise ValueError("No valid images found.")
    embodiment_value = payload.get("embodiment_id", default_embodiment_id)
    space_value = payload.get("action_space_id", default_action_space_id)
    if embodiment_value is None or space_value is None:
        raise ValueError(
            "embodiment_id and action_space_id are required in the request or "
            "as server defaults."
        )
    return (
        images,
        instruction,
        proprio_value,
        int(embodiment_value),
        int(space_value),
        is_simple,
    )


def validate_default_ids(
    policy,
    default_embodiment_id: int | None,
    default_action_space_id: int | None,
) -> None:
    if default_embodiment_id is not None and not (
        0 <= default_embodiment_id < policy.config.num_domains
    ):
        raise ValueError(
            f"Default embodiment ID must be in [0,{policy.config.num_domains})."
        )
    if (
        default_action_space_id is not None
        and default_action_space_id not in policy.action_space.specs
    ):
        raise ValueError(f"Unknown default action space ID {default_action_space_id}.")


def derive_inference_seed(
    base_seed: int | None,
    history: Any,
) -> int | None:
    """Derive a stable per-query seed from the evaluation session history."""
    if base_seed is None:
        return None
    if not isinstance(history, dict):
        raise ValueError(
            "Deterministic inference requires a history object with session_id "
            "and step_index."
        )
    session_id = history.get("session_id")
    step_index = history.get("step_index")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError(
            "Deterministic inference requires a non-empty history.session_id."
        )
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError(
            "Deterministic inference requires an integer history.step_index."
        )
    if step_index < 0:
        raise ValueError("history.step_index must be non-negative.")
    return random.Random(f"{base_seed}\0{session_id}\0{step_index}").randrange(1 << 63)


def build_app(
    model,
    processor,
    device: torch.device,
    *,
    default_embodiment_id: int | None = None,
    default_action_space_id: int | None = None,
    normalizer: DatasetNormalizer | None = None,
    rtc_training_max_delay: int | None = None,
    rtc_allow_extrapolation: bool = False,
    inference_seed: int | None = None,
    inference_steps: int = 10,
    model_path: str | None = None,
) -> FastAPI:
    if inference_seed is not None and inference_seed < 0:
        raise ValueError("inference_seed must be non-negative.")
    if inference_steps <= 0:
        raise ValueError("inference_steps must be positive.")
    app = FastAPI()
    policy = model
    validate_default_ids(policy, default_embodiment_id, default_action_space_id)
    state_conditioning_mode, state_discrete_bins = validate_state_conditioning(
        getattr(
            policy.config,
            "state_conditioning_mode",
            DISCRETE_VLM,
        ),
        getattr(
            policy.config,
            "state_discrete_bins",
            DEFAULT_STATE_DISCRETE_BINS,
        ),
    )
    vlm_dtype = next(policy.vlm.parameters()).dtype
    inference_lock = threading.Lock()
    rtc_store = (
        None
        if rtc_training_max_delay is None
        else RtcSessionStore(
            num_actions=policy.config.num_actions,
            training_max_delay=rtc_training_max_delay,
            allow_extrapolation=rtc_allow_extrapolation,
        )
    )
    normalization_space_id = None
    if normalizer is not None:
        matching_spaces = [
            space_id
            for space_id, spec in policy.action_space.specs.items()
            if spec.name == normalizer.action_space
        ]
        if len(matching_spaces) != 1:
            raise ValueError(
                f"Normalizer action space {normalizer.action_space!r} does not "
                "match exactly one model action space."
            )
        normalization_space_id = matching_spaces[0]
        spec = policy.action_space.specs[normalization_space_id]
        if (
            normalizer.action_dim != spec.action_dim
            or normalizer.proprio_dim != spec.proprio_dim
        ):
            raise ValueError(
                "Normalizer dimensions do not match the model action-space contract."
            )
        if (
            normalizer.action_space == SIMPLE_ACTION_SPACE
            and normalizer.coordinate_contract != SIMPLE_COORDINATE_CONTRACT
        ):
            raise ValueError(
                "SIMPLE normalizer does not use the required coordinate contract."
            )
    if state_conditioning_mode == DISCRETE_VLM:
        if normalizer is None:
            raise ValueError(
                "Discrete VLM state conditioning requires a q01/q99 normalizer."
            )
        if normalizer.normalization_contract != QUANTILE_NORMALIZATION_CONTRACT:
            raise ValueError(
                "Discrete VLM state conditioning requires the "
                f"{QUANTILE_NORMALIZATION_CONTRACT!r} normalization contract."
            )
        if (
            default_action_space_id is not None
            and default_action_space_id != normalization_space_id
        ):
            raise ValueError(
                "The default action space must match the discrete-state normalizer."
            )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "default_embodiment_id": default_embodiment_id,
            "default_action_space_id": default_action_space_id,
            "normalization_dataset": (
                None if normalizer is None else normalizer.dataset_name
            ),
            "normalization_action_space_id": normalization_space_id,
            "state_conditioning_mode": state_conditioning_mode,
            "state_discrete_bins": state_discrete_bins,
            "rtc_enabled": rtc_store is not None,
            "rtc_training_max_delay": rtc_training_max_delay,
            "rtc_allow_extrapolation": rtc_allow_extrapolation,
            "rtc_active_sessions": (
                None if rtc_store is None else rtc_store.active_sessions
            ),
            "inference_seed": inference_seed,
            "inference_steps": inference_steps,
            "model_path": model_path,
        }

    @app.post("/act")
    def act(payload: dict[str, Any]):
        try:
            (
                images,
                instruction,
                proprio_value,
                embodiment_value,
                space_value,
                is_simple,
            ) = parse_request(
                payload,
                default_embodiment_id=default_embodiment_id,
                default_action_space_id=default_action_space_id,
            )
            proprio = torch.as_tensor(
                np.asarray(proprio_value), dtype=torch.float32
            ).reshape(1, -1)
            if not 0 <= embodiment_value < policy.config.num_domains:
                return JSONResponse(
                    {
                        "error": (
                            f"embodiment_id must be in "
                            f"[0,{policy.config.num_domains})."
                        )
                    },
                    status_code=400,
                )
            if space_value not in policy.action_space.specs:
                return JSONResponse(
                    {"error": f"Unknown action_space_id {space_value}."},
                    status_code=400,
                )
            embodiment_id = torch.tensor(
                [embodiment_value],
                dtype=torch.long,
            )
            action_space_id = torch.tensor(
                [space_value],
                dtype=torch.long,
            )
            spec = policy.action_space.specs[space_value]
            if not torch.isfinite(proprio).all():
                return JSONResponse(
                    {"error": "proprio must contain only finite values."},
                    status_code=400,
                )
            if proprio.shape[-1] != spec.proprio_dim:
                return JSONResponse(
                    {
                        "error": (
                            f"proprio width {proprio.shape[-1]} does not match "
                            f"action space {spec.name} width {spec.proprio_dim}."
                        )
                    },
                    status_code=400,
                )
            if is_simple and normalization_space_id != space_value:
                return JSONResponse(
                    {
                        "error": (
                            "SIMPLE requests require a matching q01/q99 "
                            "normalization file."
                        )
                    },
                    status_code=400,
                )
            if (
                state_conditioning_mode == DISCRETE_VLM
                and normalization_space_id != space_value
            ):
                return JSONResponse(
                    {
                        "error": (
                            "Discrete VLM state conditioning requires the "
                            "request action space to match the server normalizer."
                        )
                    },
                    status_code=400,
                )
            if normalization_space_id == space_value:
                proprio_array = normalizer.normalize_proprio(
                    proprio.numpy(),
                )
                proprio = torch.from_numpy(proprio_array)
            proprio_valid_mask = torch.ones_like(proprio, dtype=torch.bool)
            inputs = processor(
                images=images,
                language_instruction=instruction,
                proprio=proprio,
                proprio_valid_mask=proprio_valid_mask,
                state_conditioning_mode=state_conditioning_mode,
                state_discrete_bins=state_discrete_bins,
            )
            padded_proprio = torch.zeros(
                (1, policy.max_proprio_dim), dtype=proprio.dtype
            )
            padded_proprio[:, : spec.proprio_dim] = proprio
            inputs = move_inputs(inputs, device)
            inputs["proprio"] = padded_proprio.to(device=device)
            inputs["embodiment_id"] = embodiment_id.to(device=device)
            inputs["action_space_id"] = action_space_id.to(device=device)
            use_autocast = device.type == "cuda" and vlm_dtype in (
                torch.float16,
                torch.bfloat16,
            )
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=vlm_dtype)
                if use_autocast
                else nullcontext()
            )
            requested_steps = payload.get("steps")
            if requested_steps is not None and (
                isinstance(requested_steps, bool)
                or not isinstance(requested_steps, int)
                or requested_steps != inference_steps
            ):
                return JSONResponse(
                    {
                        "error": (
                            f"This server uses exactly {inference_steps} inference "
                            "steps."
                        )
                    },
                    status_code=400,
                )
            request_seed = derive_inference_seed(
                inference_seed,
                payload.get("history"),
            )
            with inference_lock, torch.inference_mode(), autocast_context:
                rtc_context = (
                    None
                    if rtc_store is None
                    else rtc_store.prepare(payload.get("history"))
                )
                if rtc_context is not None:
                    LOGGER.info(
                        "RTC session=%s delay=%d prefix=%s extrapolating=%s",
                        rtc_context.session_id,
                        rtc_context.inference_delay,
                        rtc_context.action_prefix is not None,
                        rtc_context.extrapolating,
                    )
                generation_kwargs: dict[str, Any] = {}
                if (
                    rtc_context is not None
                    and rtc_context.action_prefix is not None
                    and rtc_context.inference_delay > 0
                ):
                    generation_kwargs["action_prefix"] = torch.from_numpy(
                        rtc_context.action_prefix
                    ).unsqueeze(0).to(device=device)
                    generation_kwargs["prefix_lengths"] = torch.tensor(
                        [rtc_context.inference_delay],
                        dtype=torch.long,
                        device=device,
                    )
                if request_seed is not None:
                    generation_kwargs["generator"] = torch.Generator(
                        device=device
                    ).manual_seed(request_seed)
                action = policy.generate_actions(
                    **inputs,
                    steps=inference_steps,
                    **generation_kwargs,
                )
                normalized_action_array = action[0].float().cpu().numpy()
                if rtc_store is not None:
                    rtc_store.commit(rtc_context, normalized_action_array)
            action_array = normalized_action_array
            if normalization_space_id == space_value:
                action_array = normalizer.denormalize_action(action_array)
            if is_simple:
                return JSONResponse(
                    {
                        "action": encode_numpy_envelope(action_array),
                        "err": 0.0,
                        "traj_image": encode_numpy_envelope(
                            np.zeros((1, 1, 3), dtype=np.uint8)
                        ),
                    }
                )
            return JSONResponse(
                {
                    "action": action_array.tolist(),
                    "action_space_id": int(action_space_id.item()),
                }
            )
        except Exception:
            logging.error("Action request failed:\n%s", traceback.format_exc())
            return JSONResponse({"error": "Request failed."}, status_code=400)
    return app


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="lambda0 serve", description="Launch Lambda0 FastAPI server")
    parser.add_argument("--model", dest="model_path", type=str, required=True)
    parser.add_argument("--processor", dest="processor_path", type=str, default=None)
    parser.add_argument("--output", dest="output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--port", default=8010, type=int)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--default-embodiment-id", type=int)
    parser.add_argument("--default-action-space-id", type=int)
    parser.add_argument("--normalization", dest="normalization_path", type=Path)
    parser.add_argument(
        "--rtc-training-max-delay",
        type=int,
        default=None,
        help="Enable RTC and record the maximum delay used during training.",
    )
    parser.add_argument(
        "--rtc-allow-extrapolation",
        action="store_true",
        help="Allow inference delays beyond rtc_training_max_delay.",
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help=(
            "Base seed for deterministic per-session, per-step diffusion noise."
        ),
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=10,
        help="Fixed number of action diffusion steps accepted by this server.",
    )
    args = parser.parse_args(argv)
    if args.rtc_allow_extrapolation and args.rtc_training_max_delay is None:
        parser.error(
            "--rtc_allow_extrapolation requires --rtc_training_max_delay."
        )
    if args.inference_seed is not None and args.inference_seed < 0:
        parser.error("--inference_seed must be non-negative.")
    if args.inference_steps <= 0:
        parser.error("--inference_steps must be positive.")
    if args.normalization_path is None:
        parser.error("Release inference requires checkpoint-bound --normalization_path.")
    checkpoint = Path(args.model_path)
    bundle = (checkpoint.parent if checkpoint.name == "ema" else checkpoint) / "data_bundle"
    manifest = json.loads((bundle / "manifest.json").read_text())
    statistics = json.loads(args.normalization_path.read_text())
    identity = statistics.get("dataset_name")
    if identity not in manifest["datasets"]:
        parser.error("Normalization dataset is not listed in this checkpoint.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = Lambda0Processor.from_pretrained(
        args.processor_path or args.model_path
    )
    model = Lambda0.from_pretrained(
        args.model_path,
        dtype=resolve_dtype(args.dtype, device),
        low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()
    policy = model
    state_conditioning_mode, state_discrete_bins = validate_state_conditioning(
        getattr(
            policy.config,
            "state_conditioning_mode",
            DISCRETE_VLM,
        ),
        getattr(
            policy.config,
            "state_discrete_bins",
            DEFAULT_STATE_DISCRETE_BINS,
        ),
    )
    normalizer = (
        None
        if args.normalization_path is None
        else DatasetNormalizer(args.normalization_path)
    )
    host = args.host
    info_path = output_dir / "info.json"
    if info_path.exists():
        raise FileExistsError(
            f"{info_path} already exists; stop the existing server or choose "
            "another output directory."
        )
    with info_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "host": host,
                "port": args.port,
                "model_path": args.model_path,
                "default_embodiment_id": args.default_embodiment_id,
                "default_action_space_id": args.default_action_space_id,
                "normalization_path": (
                    None
                    if args.normalization_path is None
                    else str(args.normalization_path.resolve())
                ),
                "rtc_enabled": args.rtc_training_max_delay is not None,
                "rtc_training_max_delay": args.rtc_training_max_delay,
                "rtc_allow_extrapolation": args.rtc_allow_extrapolation,
                "inference_seed": args.inference_seed,
                "inference_steps": args.inference_steps,
                "state_conditioning_mode": state_conditioning_mode,
                "state_discrete_bins": state_discrete_bins,
            },
            file,
            indent=2,
        )
    app = build_app(
        model,
        processor,
        device,
        default_embodiment_id=args.default_embodiment_id,
        default_action_space_id=args.default_action_space_id,
        normalizer=normalizer,
        rtc_training_max_delay=args.rtc_training_max_delay,
        rtc_allow_extrapolation=args.rtc_allow_extrapolation,
        inference_seed=args.inference_seed,
        inference_steps=args.inference_steps,
        model_path=str(Path(args.model_path).resolve()),
    )
    uvicorn.run(app, host=host, port=args.port)


if __name__ == "__main__":
    main()
