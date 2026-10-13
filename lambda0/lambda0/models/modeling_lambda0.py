from __future__ import annotations

from typing import Any

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5Model
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .action_expert import LayerwiseActionExpert
from .action_loss import compute_heterogeneous_loss
from .action_specs import action_space_config, action_space_specs_from_config
from .configuration_lambda0 import ACTION_CHUNK_SIZE, Lambda0Config
from .heterogeneous_action_space import HeterogeneousActionSpace
from .state_conditioning import (
    CONTINUOUS_ACTION_EXPERT,
    DEFAULT_STATE_DISCRETE_BINS,
    validate_state_conditioning,
)


_CONFIG_LOAD_KEYS = {
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "subfolder",
    "token",
}


class Lambda0(PreTrainedModel):
    """Qwen3.5 VLM with a robot-conditioned, layerwise action expert."""

    config_class = Lambda0Config
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
        "LayerwiseExpertBlock",
    ]

    def __init__(
        self, config: Lambda0Config, vlm: Qwen3_5Model | None = None
    ) -> None:
        super().__init__(config)
        self.num_actions = config.num_actions
        self.use_proprio = config.use_proprio
        self.state_conditioning_mode = config.state_conditioning_mode
        self.state_discrete_bins = config.state_discrete_bins

        self.vlm = vlm if vlm is not None else Qwen3_5Model(config.qwen_config)
        runtime_action_specs = action_space_specs_from_config(config.action_space_specs)
        self.action_space = HeterogeneousActionSpace(
            hidden_size=config.expert_hidden_size,
            dim_time=config.dim_time,
            action_space_specs=runtime_action_specs,
        )
        self.max_action_dim = self.action_space.max_action_dim
        self.max_proprio_dim = self.action_space.max_proprio_dim
        self.action_expert = LayerwiseActionExpert(
            vlm_hidden_size=config.vlm_hidden_size,
            hidden_size=config.expert_hidden_size,
            num_layers=config.expert_num_layers,
            num_heads=config.expert_num_heads,
            mlp_ratio=config.expert_mlp_ratio,
            dropout=config.expert_dropout,
            num_domains=config.num_domains,
            soft_prompt_length=config.soft_prompt_length,
            num_actions=config.num_actions,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        self.action_expert.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.vlm.gradient_checkpointing_disable()
        self.action_expert.gradient_checkpointing = False

    @classmethod
    def from_vlm_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        variant: str | None = None,
        num_domains: int = 30,
        soft_prompt_length: int = 32,
        num_actions: int = ACTION_CHUNK_SIZE,
        use_proprio: bool = True,
        state_conditioning_mode: str = "discrete_vlm",
        state_discrete_bins: int = DEFAULT_STATE_DISCRETE_BINS,
        prediction_type: str = "velocity",
        action_space_specs: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "Lambda0":
        config_kwargs = {key: kwargs[key] for key in _CONFIG_LOAD_KEYS if key in kwargs}
        qwen_config = Qwen3_5Config.from_pretrained(
            pretrained_model_name_or_path,
            **config_kwargs,
        )
        config = Lambda0Config(
            qwen_config=qwen_config,
            variant=variant,
            base_model_name_or_path=pretrained_model_name_or_path,
            num_domains=num_domains,
            soft_prompt_length=soft_prompt_length,
            num_actions=num_actions,
            use_proprio=use_proprio,
            state_conditioning_mode=state_conditioning_mode,
            state_discrete_bins=state_discrete_bins,
            prediction_type=prediction_type,
            action_space_specs=action_space_specs,
        )
        qwen_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            config=qwen_config,
            **kwargs,
        )
        return cls(config, vlm=qwen_model.model)

    @classmethod
    def from_pretrained_or_vlm(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "Lambda0":
        strict_checkpoint_loading = bool(
            kwargs.pop("strict_checkpoint_loading", True)
        )
        config_kwargs = {key: kwargs[key] for key in _CONFIG_LOAD_KEYS if key in kwargs}
        config_dict, _ = PretrainedConfig.get_config_dict(
            pretrained_model_name_or_path,
            **config_kwargs,
        )
        if config_dict.get("model_type") == Lambda0Config.model_type:
            config = Lambda0Config.from_dict(config_dict)
            if kwargs.get("prediction_type", config.prediction_type) != config.prediction_type:
                raise ValueError("Requested prediction_type differs from checkpoint; cannot silently reinterpret weights.")
            vla_only_keys = {
                "variant",
                "num_domains",
                "soft_prompt_length",
                "num_actions",
                "use_proprio",
                "state_conditioning_mode",
                "state_discrete_bins",
                "action_space_specs",
                "prediction_type",
            }
            load_kwargs = {
                key: value for key, value in kwargs.items() if key not in vla_only_keys
            }
            model, loading_info = cls.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
                output_loading_info=True,
                **load_kwargs,
            )
            if strict_checkpoint_loading:
                incompatible = {
                    key: loading_info.get(key, [])
                    for key in (
                        "missing_keys",
                        "unexpected_keys",
                        "mismatched_keys",
                        "error_msgs",
                    )
                    if loading_info.get(key)
                }
                if incompatible:
                    raise RuntimeError(
                        "Lambda0 checkpoint did not load exactly: "
                        f"{incompatible}"
                    )
            requested_variant = kwargs.get("variant")
            if (
                requested_variant is not None
                and requested_variant != model.config.variant
            ):
                raise ValueError(
                    f"Checkpoint variant {model.config.variant!r} does not match "
                    f"requested variant {requested_variant!r}."
                )
            requested_actions = kwargs.get("num_actions")
            if (
                requested_actions is not None
                and int(requested_actions) != model.num_actions
            ):
                raise ValueError(
                    f"Cannot change num_actions from {model.num_actions} to "
                    f"{requested_actions} while continuing a checkpoint."
                )
            requested_prompt_length = kwargs.get("soft_prompt_length")
            if (
                requested_prompt_length is not None
                and int(requested_prompt_length)
                != model.action_expert.soft_prompt_length
            ):
                raise ValueError(
                    "Cannot change soft_prompt_length while continuing a checkpoint."
                )

            requested_specs = action_space_specs_from_config(
                kwargs.get("action_space_specs")
                or list(action_space_config())
            )
            model.action_space.extend_specs(requested_specs)
            model.max_action_dim = model.action_space.max_action_dim
            model.max_proprio_dim = model.action_space.max_proprio_dim
            model.config.action_space_specs = list(
                action_space_config(model.action_space.specs)
            )

            requested_domains = kwargs.get("num_domains")
            if requested_domains is not None:
                model.action_expert.resize_num_domains(int(requested_domains))
                model.config.num_domains = model.action_expert.num_domains
            if "use_proprio" in kwargs:
                model.use_proprio = bool(kwargs["use_proprio"])
                model.config.use_proprio = model.use_proprio
            requested_mode = kwargs.get(
                "state_conditioning_mode",
                model.config.state_conditioning_mode,
            )
            requested_bins = kwargs.get(
                "state_discrete_bins",
                model.config.state_discrete_bins,
            )
            mode, bins = validate_state_conditioning(
                requested_mode,
                requested_bins,
            )
            if not model.use_proprio:
                raise ValueError(
                    "Discrete VLM state conditioning requires use_proprio=True."
                )
            model.state_conditioning_mode = mode
            model.state_discrete_bins = bins
            model.config.state_conditioning_mode = mode
            model.config.state_discrete_bins = bins
            return model
        if config_dict.get("model_type") != "qwen3_5":
            raise ValueError(
                f"Expected a Qwen3.5 base or Lambda0 checkpoint, got "
                f"model_type={config_dict.get('model_type')!r}."
            )
        return cls.from_vlm_pretrained(pretrained_model_name_or_path, **kwargs)

    def _pack_auxiliary_features(
        self,
        aux_pixel_values: torch.Tensor | None,
        aux_image_grid_thw: torch.LongTensor | None,
        aux_image_owner: torch.LongTensor | None,
        batch_size: int,
    ) -> tuple[torch.Tensor | None, torch.BoolTensor | None]:
        if aux_pixel_values is None:
            return None, None
        if aux_image_grid_thw is None or aux_image_owner is None:
            raise ValueError(
                "aux_image_grid_thw and aux_image_owner are required with aux_pixel_values."
            )
        vision_output = self.vlm.get_image_features(
            aux_pixel_values,
            aux_image_grid_thw,
            return_dict=True,
        )
        image_features = vision_output.pooler_output
        if len(image_features) != aux_image_owner.numel():
            raise ValueError(
                "Auxiliary image ownership does not match encoded image count."
            )
        owner_values = aux_image_owner.tolist()
        token_counts = [0] * batch_size
        for owner, features in zip(owner_values, image_features, strict=True):
            if owner < 0 or owner >= batch_size:
                raise ValueError(f"Invalid auxiliary image owner index {owner}.")
            token_counts[owner] += features.shape[0]
        max_tokens = max(token_counts, default=0)
        hidden_size = self.config.vlm_hidden_size
        packed = aux_pixel_values.new_zeros((batch_size, max_tokens, hidden_size))
        mask = torch.zeros(
            batch_size,
            max_tokens,
            dtype=torch.bool,
            device=aux_pixel_values.device,
        )
        offsets = [0] * batch_size
        for owner, features in zip(owner_values, image_features, strict=True):
            start = offsets[owner]
            end = start + features.shape[0]
            packed[owner, start:end] = features.to(dtype=packed.dtype)
            mask[owner, start:end] = True
            offsets[owner] = end
        return packed, mask

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        aux_pixel_values: torch.Tensor | None = None,
        aux_image_grid_thw: torch.LongTensor | None = None,
        aux_image_owner: torch.LongTensor | None = None,
    ) -> dict[str, Any]:
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Qwen3.5 did not return hidden states.")
        num_layers = self.config.expert_num_layers
        if len(outputs.hidden_states) < num_layers:
            raise RuntimeError(
                f"Qwen3.5 returned {len(outputs.hidden_states)} hidden states "
                f"for {num_layers} layers."
            )
        layer_states = tuple(outputs.hidden_states[-num_layers:])
        auxiliary_features, auxiliary_mask = self._pack_auxiliary_features(
            aux_pixel_values,
            aux_image_grid_thw,
            aux_image_owner,
            input_ids.shape[0],
        )
        return {
            "vlm_hidden_states": layer_states,
            "vlm_attention_mask": attention_mask,
            "auxiliary_features": auxiliary_features,
            "auxiliary_mask": auxiliary_mask,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        action_space_id: torch.LongTensor | None = None,
        embodiment_id: torch.LongTensor | None = None,
        action_valid_mask: torch.BoolTensor | None = None,
        proprio_valid_mask: torch.BoolTensor | None = None,
        aux_pixel_values: torch.Tensor | None = None,
        aux_image_grid_thw: torch.LongTensor | None = None,
        aux_image_owner: torch.LongTensor | None = None,
        rtc_max_delay: int = 0,
        rtc_no_prefix_probability: float = 0.5,
        action_expert_noise_samples: int = 1,
        noise_seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.forward_vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            aux_pixel_values=aux_pixel_values,
            aux_image_grid_thw=aux_image_grid_thw,
            aux_image_owner=aux_image_owner,
        )
        if embodiment_id is None:
            raise ValueError("embodiment_id is required.")
        if action_space_id is None:
            raise ValueError("action_space_id is required for heterogeneous training.")
        if action.shape[-1] != self.max_action_dim:
            raise ValueError(
                f"action must be padded to {self.max_action_dim} dimensions, "
                f"got {action.shape[-1]}."
            )
        if proprio.shape[-1] != self.max_proprio_dim:
            raise ValueError(
                f"proprio must be padded to {self.max_proprio_dim} dimensions, "
                f"got {proprio.shape[-1]}."
            )
        if action.shape[1] != self.num_actions:
            raise ValueError(
                f"action horizon must be {self.num_actions}, got {action.shape[1]}."
            )
        if not 0 <= int(rtc_max_delay) < self.num_actions:
            raise ValueError("rtc_max_delay must be in [0,num_actions).")
        if int(action_expert_noise_samples) != action_expert_noise_samples:
            raise ValueError("action_expert_noise_samples must be an integer.")
        action_expert_noise_samples = int(action_expert_noise_samples)
        if action_expert_noise_samples < 1:
            raise ValueError("action_expert_noise_samples must be positive.")
        batch_size = input_ids.shape[0]
        if action_valid_mask is None:
            action_valid_mask = self.action_space.action_mask(
                action_space_id, self.num_actions, action.device
            )
        if action_valid_mask.shape != action.shape:
            raise ValueError("action_valid_mask must have the same shape as action.")
        action_valid_mask = action_valid_mask.to(
            device=action.device, dtype=torch.bool
        )
        if proprio_valid_mask is None:
            proprio_valid_mask = self.action_space.proprio_mask(
                action_space_id, proprio.device
            )
        if proprio_valid_mask.shape != proprio.shape:
            raise ValueError("proprio_valid_mask must have the same shape as proprio.")
        proprio_valid_mask = proprio_valid_mask.to(
            device=proprio.device, dtype=torch.bool
        )
        if self.use_proprio and self.state_conditioning_mode == CONTINUOUS_ACTION_EXPERT:
            encoded_proprio = proprio
            encoded_proprio_mask = proprio_valid_mask
        else:
            encoded_proprio = torch.zeros_like(proprio)
            encoded_proprio_mask = torch.zeros_like(proprio_valid_mask)
        generator = None
        if noise_seed is not None:
            generator = torch.Generator(device=action.device)
            generator.manual_seed(int(noise_seed))
        if action_expert_noise_samples > 1:
            repeat_shape = (action_expert_noise_samples,)

            def repeat_batch(tensor: torch.Tensor | None) -> torch.Tensor | None:
                if tensor is None:
                    return None
                return tensor.repeat(repeat_shape + (1,) * (tensor.ndim - 1))

            action = repeat_batch(action)
            encoded_proprio = repeat_batch(encoded_proprio)
            encoded_proprio_mask = repeat_batch(encoded_proprio_mask)
            action_space_id = repeat_batch(action_space_id)
            embodiment_id = repeat_batch(embodiment_id)
            action_valid_mask = repeat_batch(action_valid_mask)
            encoded = {
                "vlm_hidden_states": tuple(
                    repeat_batch(hidden_state)
                    for hidden_state in encoded["vlm_hidden_states"]
                ),
                "vlm_attention_mask": repeat_batch(
                    encoded["vlm_attention_mask"]
                ),
                "auxiliary_features": repeat_batch(
                    encoded["auxiliary_features"]
                ),
                "auxiliary_mask": repeat_batch(encoded["auxiliary_mask"]),
            }
            batch_size *= action_expert_noise_samples

        # Draw after expansion: every noise replica has independent time/prefix.
        from .flow_matching import sample_flow_conditions
        sample_timestep, prefix_mask = sample_flow_conditions(
            action_valid_mask, int(rtc_max_delay), rtc_no_prefix_probability, generator
        )
        sample_timestep = sample_timestep.to(action.dtype)

        timestep = sample_timestep.unsqueeze(1).expand(-1, self.num_actions).clone()
        timestep = timestep.masked_fill(prefix_mask, 0.0)
        noise = torch.randn(
            action.shape,
            device=action.device,
            dtype=action.dtype,
            generator=generator,
        )
        noisy_action = noise * timestep.unsqueeze(-1) + action * (
            1 - timestep
        ).unsqueeze(-1)
        noisy_action = noisy_action.masked_fill(~action_valid_mask, 0.0)
        action_tokens = self.action_space.encode(
            noisy_action,
            encoded_proprio,
            timestep,
            action_space_id,
            encoded_proprio_mask,
        )
        action_hidden = self.action_expert(
            embodiment_id=embodiment_id,
            action_tokens=action_tokens,
            action_token_mask=action_valid_mask.any(dim=-1),
            **encoded,
        )
        predicted_action = self.action_space.decode(action_hidden, action_space_id)
        return compute_heterogeneous_loss(
            predicted_action,
            noise - action if self.config.prediction_type == "velocity" else action,
            action_space_id,
            action_valid_mask & ~prefix_mask.unsqueeze(-1),
            self.action_space.specs,
        )

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        proprio: torch.Tensor,
        action_space_id: torch.LongTensor,
        proprio_valid_mask: torch.BoolTensor | None = None,
        embodiment_id: torch.LongTensor | None = None,
        aux_pixel_values: torch.Tensor | None = None,
        aux_image_grid_thw: torch.LongTensor | None = None,
        aux_image_owner: torch.LongTensor | None = None,
        steps: int = 10,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        action_prefix: torch.Tensor | None = None,
        prefix_lengths: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        if int(num_samples) != num_samples or int(num_samples) < 1:
            raise ValueError("num_samples must be a positive integer.")
        num_samples = int(num_samples)
        encoded = self.forward_vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            aux_pixel_values=aux_pixel_values,
            aux_image_grid_thw=aux_image_grid_thw,
            aux_image_owner=aux_image_owner,
        )
        batch_size = input_ids.shape[0]
        observation_batch_size = batch_size
        if embodiment_id is None:
            raise ValueError("embodiment_id is required.")
        if proprio.shape[-1] != self.max_proprio_dim:
            raise ValueError(
                f"proprio must be padded to {self.max_proprio_dim} dimensions, "
                f"got {proprio.shape[-1]}."
            )
        if proprio_valid_mask is None:
            proprio_valid_mask = self.action_space.proprio_mask(
                action_space_id, proprio.device
            )
        if proprio_valid_mask.shape != proprio.shape:
            raise ValueError("proprio_valid_mask must have the same shape as proprio.")
        proprio_valid_mask = proprio_valid_mask.to(
            device=proprio.device, dtype=torch.bool
        )
        if self.use_proprio and self.state_conditioning_mode == CONTINUOUS_ACTION_EXPERT:
            encoded_proprio = proprio
            encoded_proprio_mask = proprio_valid_mask
        else:
            encoded_proprio = torch.zeros_like(proprio)
            encoded_proprio_mask = torch.zeros_like(proprio_valid_mask)
        action = torch.zeros(
            batch_size,
            self.num_actions,
            self.max_action_dim,
            device=proprio.device,
            dtype=proprio.dtype,
        )
        action_mask = self.action_space.action_mask(
            action_space_id,
            self.num_actions,
            proprio.device,
        )
        padded_prefix = torch.zeros_like(action)
        prefix_mask = torch.zeros(
            batch_size,
            self.num_actions,
            dtype=torch.bool,
            device=proprio.device,
        )
        if action_prefix is None:
            if prefix_lengths is not None:
                raise ValueError("prefix_lengths requires action_prefix.")
        else:
            if action_prefix.ndim != 3 or action_prefix.shape[0] != batch_size:
                raise ValueError("action_prefix must have shape [B,T,D].")
            prefix_horizon = action_prefix.shape[1]
            prefix_width = action_prefix.shape[2]
            if prefix_horizon > self.num_actions:
                raise ValueError("action_prefix horizon exceeds num_actions.")
            if prefix_width > self.max_action_dim:
                raise ValueError("action_prefix width exceeds max_action_dim.")
            if prefix_lengths is None:
                prefix_lengths = torch.full(
                    (batch_size,),
                    prefix_horizon,
                    dtype=torch.long,
                    device=proprio.device,
                )
            else:
                prefix_lengths = prefix_lengths.to(
                    device=proprio.device,
                    dtype=torch.long,
                ).reshape(-1)
                if prefix_lengths.numel() != batch_size:
                    raise ValueError("prefix_lengths must contain one value per sample.")
            if bool((prefix_lengths < 0).any()) or bool(
                (prefix_lengths > prefix_horizon).any()
            ):
                raise ValueError("prefix_lengths must be in [0,action_prefix.shape[1]].")
            padded_prefix[:, :prefix_horizon, :prefix_width] = action_prefix.to(
                device=proprio.device,
                dtype=proprio.dtype,
            )
            prefix_mask = (
                torch.arange(self.num_actions, device=proprio.device).unsqueeze(0)
                < prefix_lengths.unsqueeze(1)
            )
            prefix_mask &= action_mask.any(dim=-1)
            padded_prefix = padded_prefix.masked_fill(~action_mask, 0.0)
            action = torch.where(
                prefix_mask.unsqueeze(-1),
                padded_prefix,
                action,
            )
        original_action_space_id = action_space_id
        if num_samples > 1:
            repeat_shape = (num_samples,)

            def repeat_batch(tensor: torch.Tensor | None) -> torch.Tensor | None:
                if tensor is None:
                    return None
                return tensor.repeat(repeat_shape + (1,) * (tensor.ndim - 1))

            encoded_proprio = repeat_batch(encoded_proprio)
            encoded_proprio_mask = repeat_batch(encoded_proprio_mask)
            action_space_id = repeat_batch(action_space_id)
            embodiment_id = repeat_batch(embodiment_id)
            action = repeat_batch(action)
            action_mask = repeat_batch(action_mask)
            padded_prefix = repeat_batch(padded_prefix)
            prefix_mask = repeat_batch(prefix_mask)
            encoded = {
                "vlm_hidden_states": tuple(
                    repeat_batch(hidden_state)
                    for hidden_state in encoded["vlm_hidden_states"]
                ),
                "vlm_attention_mask": repeat_batch(
                    encoded["vlm_attention_mask"]
                ),
                "auxiliary_features": repeat_batch(
                    encoded["auxiliary_features"]
                ),
                "auxiliary_mask": repeat_batch(encoded["auxiliary_mask"]),
            }
            batch_size *= num_samples
        initial_noise = torch.randn(
            action.shape,
            dtype=action.dtype,
            device=action.device,
            generator=generator,
        )
        action = torch.where(prefix_mask.unsqueeze(-1), padded_prefix, initial_noise)
        steps = max(1, int(steps))
        for step in range(steps, 0, -1):
            timestep = torch.full(
                (batch_size, self.num_actions),
                step / steps,
                device=proprio.device,
                dtype=proprio.dtype,
            )
            timestep = timestep.masked_fill(prefix_mask, 0.0)
            if self.config.prediction_type == "x0":
                # Legacy x0 sampler: re-noise the latest clean prediction with fixed noise.
                noisy_action = initial_noise * timestep.unsqueeze(-1) + action * (1 - timestep.unsqueeze(-1))
            else:
                noisy_action = action
            noisy_action = torch.where(
                prefix_mask.unsqueeze(-1),
                padded_prefix,
                noisy_action,
            )
            noisy_action = noisy_action.masked_fill(~action_mask, 0.0)
            action_tokens = self.action_space.encode(
                noisy_action,
                encoded_proprio,
                timestep,
                action_space_id,
                encoded_proprio_mask,
            )
            action_hidden = self.action_expert(
                embodiment_id=embodiment_id,
                action_tokens=action_tokens,
                action_token_mask=action_mask.any(dim=-1),
                **encoded,
            )
            prediction = self.action_space.decode(action_hidden, action_space_id)
            action = action - prediction / steps if self.config.prediction_type == "velocity" else prediction
            action = torch.where(
                prefix_mask.unsqueeze(-1),
                padded_prefix,
                action,
            )
            action = action.masked_fill(~action_mask, 0.0)
        if num_samples > 1:
            action = action.float().reshape(
                num_samples,
                observation_batch_size,
                self.num_actions,
                self.max_action_dim,
            ).mean(dim=0)
        return self.action_space.trim_single_space(
            action,
            original_action_space_id,
        )


__all__ = ["Lambda0"]
