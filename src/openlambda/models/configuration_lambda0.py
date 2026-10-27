from __future__ import annotations

from typing import Any

from transformers import AutoConfig, Qwen3_5Config
from transformers.configuration_utils import PretrainedConfig

from .action_specs import (
    action_space_config,
    action_space_specs_from_config,
)
from .state_conditioning import (
    DISCRETE_VLM,
    DEFAULT_STATE_DISCRETE_BINS,
    validate_state_conditioning,
)


ACTION_CHUNK_SIZE = 50


LAMBDA0_VARIANTS: dict[str, dict[str, Any]] = {
    "0.8b": {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "vlm_hidden_size": 1024,
        "num_hidden_layers": 24,
        "expert_hidden_size": 512,
        "expert_num_heads": 8,
    },
    "2b": {
        "model_id": "Qwen/Qwen3.5-2B",
        "vlm_hidden_size": 2048,
        "num_hidden_layers": 24,
        "expert_hidden_size": 768,
        "expert_num_heads": 12,
    },
    "4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "vlm_hidden_size": 2560,
        "num_hidden_layers": 32,
        "expert_hidden_size": 1024,
        "expert_num_heads": 16,
    },
}


def normalize_variant(variant: str) -> str:
    key = variant.lower().replace("qwen3.5-", "").replace("qwen35-", "")
    if key not in LAMBDA0_VARIANTS:
        supported = ", ".join(LAMBDA0_VARIANTS)
        raise ValueError(
            f"Unsupported Lambda0 variant '{variant}'. Supported: {supported}."
        )
    return key


def infer_variant(qwen_config: Qwen3_5Config) -> str:
    signature = (
        qwen_config.text_config.hidden_size,
        qwen_config.text_config.num_hidden_layers,
    )
    for name, preset in LAMBDA0_VARIANTS.items():
        if signature == (preset["vlm_hidden_size"], preset["num_hidden_layers"]):
            return name
    raise ValueError(
        "The Qwen3.5 checkpoint does not match a supported VLA size: "
        f"hidden_size={signature[0]}, num_hidden_layers={signature[1]}."
    )


class Lambda0Config(PretrainedConfig):
    model_type = "lambda0"
    is_composition = True
    has_no_defaults_at_init = True

    def __init__(
        self,
        qwen_config: dict[str, Any] | Qwen3_5Config | None = None,
        variant: str | None = None,
        base_model_name_or_path: str | None = None,
        expert_hidden_size: int | None = None,
        expert_num_heads: int | None = None,
        expert_mlp_ratio: float = 4.0,
        expert_dropout: float = 0.0,
        num_domains: int = 30,
        soft_prompt_length: int = 32,
        dim_time: int = 32,
        num_actions: int = ACTION_CHUNK_SIZE,
        use_proprio: bool = True,
        state_conditioning_mode: str = DISCRETE_VLM,
        state_discrete_bins: int = DEFAULT_STATE_DISCRETE_BINS,
        prediction_type: str = "velocity",
        training_contract: str | None = None,
        action_space_specs: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if qwen_config is None:
            raise ValueError(
                "qwen_config is required; initialize from an official Qwen3.5 checkpoint."
            )
        if isinstance(qwen_config, dict):
            qwen_config = Qwen3_5Config(**qwen_config)
        if not isinstance(qwen_config, Qwen3_5Config):
            raise TypeError(
                f"qwen_config must be Qwen3_5Config, got {type(qwen_config).__name__}."
            )

        inferred_variant = infer_variant(qwen_config)
        if variant is not None and normalize_variant(variant) != inferred_variant:
            raise ValueError(
                f"Requested variant '{variant}' does not match the checkpoint architecture "
                f"('{inferred_variant}')."
            )
        preset = LAMBDA0_VARIANTS[inferred_variant]

        self.qwen_config = qwen_config
        self.variant = inferred_variant
        self.base_model_name_or_path = base_model_name_or_path or preset["model_id"]
        self.vlm_hidden_size = int(qwen_config.text_config.hidden_size)
        self.expert_num_layers = int(qwen_config.text_config.num_hidden_layers)
        self.expert_hidden_size = int(
            expert_hidden_size or preset["expert_hidden_size"]
        )
        self.expert_num_heads = int(expert_num_heads or preset["expert_num_heads"])
        self.expert_mlp_ratio = float(expert_mlp_ratio)
        self.expert_dropout = float(expert_dropout)
        self.num_domains = int(num_domains)
        self.soft_prompt_length = int(soft_prompt_length)
        self.dim_time = int(dim_time)
        self.num_actions = int(num_actions)
        if self.num_actions != ACTION_CHUNK_SIZE:
            raise ValueError("The release contract requires a 50-frame horizon.")
        if prediction_type not in ("x0", "velocity"):
            raise ValueError("prediction_type must be x0 or velocity.")
        expected_contract = f"lambda0_{prediction_type}_v1"
        if training_contract is not None and training_contract != expected_contract:
            raise ValueError("Unsupported prediction/checkpoint contract.")
        self.prediction_type = prediction_type
        self.training_contract = expected_contract
        self.use_proprio = bool(use_proprio)
        (
            self.state_conditioning_mode,
            self.state_discrete_bins,
        ) = validate_state_conditioning(
            state_conditioning_mode,
            state_discrete_bins,
        )
        if not self.use_proprio:
            raise ValueError(
                "Discrete VLM state conditioning requires use_proprio=True."
            )
        runtime_specs = action_space_specs_from_config(action_space_specs)
        self.action_space_specs = list(action_space_config(runtime_specs))

        if self.expert_hidden_size % self.expert_num_heads != 0:
            raise ValueError(
                "expert_hidden_size must be divisible by expert_num_heads."
            )
        if self.num_domains <= 0 or self.num_actions <= 0:
            raise ValueError("num_domains and num_actions must be positive.")
        if self.soft_prompt_length < 0:
            raise ValueError("soft_prompt_length cannot be negative.")

        kwargs.setdefault("architectures", ["Lambda0"])
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["qwen_config"] = self.qwen_config.to_dict()
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if config_dict.get("training_contract") not in ("lambda0_velocity_v1", "lambda0_x0_v1"):
            raise ValueError("Checkpoint lacks a prediction contract; use explicit legacy initialization for training.")
        return super().from_dict(config_dict, **kwargs)


AutoConfig.register(Lambda0Config.model_type, Lambda0Config, exist_ok=True)


__all__ = [
    "ACTION_CHUNK_SIZE",
    "LAMBDA0_VARIANTS",
    "Lambda0Config",
    "infer_variant",
    "normalize_variant",
]
