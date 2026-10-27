from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MaskedSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = float(dropout)

    def forward(self, tokens: torch.Tensor, allowed: torch.BoolTensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = tokens.shape
        qkv = self.qkv(tokens).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed.unsqueeze(1),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        return self.output(attended)


class LayerwiseExpertBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = MaskedSelfAttention(hidden_size, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, allowed: torch.BoolTensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens), allowed)
        return tokens + self.mlp(self.mlp_norm(tokens))


class LayerwiseActionExpert(nn.Module):
    """Consumes every Qwen layer state while evolving prompt and action tokens."""

    def __init__(
        self,
        vlm_hidden_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_domains: int,
        soft_prompt_length: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_domains = num_domains
        self.num_actions = num_actions
        self.soft_prompt_length = soft_prompt_length
        self.gradient_checkpointing = False

        self.context_norms = nn.ModuleList(
            nn.LayerNorm(vlm_hidden_size) for _ in range(num_layers)
        )
        self.context_projection = nn.Linear(vlm_hidden_size, hidden_size)
        self.context_scales = nn.Parameter(torch.ones(num_layers))
        self.auxiliary_norm = nn.LayerNorm(vlm_hidden_size)
        self.auxiliary_projection = nn.Linear(vlm_hidden_size, hidden_size)

        self.embodiment_bias = nn.Embedding(num_domains, hidden_size)
        nn.init.zeros_(self.embodiment_bias.weight)

        if soft_prompt_length:
            self.soft_prompt_hub: nn.Embedding | None = nn.Embedding(
                num_domains,
                soft_prompt_length * hidden_size,
            )
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)
            self.prompt_position = nn.Parameter(
                torch.zeros(1, soft_prompt_length, hidden_size)
            )
            nn.init.normal_(self.prompt_position, std=0.02)
        else:
            self.soft_prompt_hub = None
            self.register_parameter("prompt_position", None)

        self.action_position = nn.Parameter(torch.zeros(1, num_actions, hidden_size))
        nn.init.normal_(self.action_position, std=0.02)
        self.token_type_embedding = nn.Embedding(4, hidden_size)
        nn.init.normal_(self.token_type_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList(
            LayerwiseExpertBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def resize_num_domains(self, num_domains: int) -> None:
        """Append embodiment rows while preserving every learned row."""
        num_domains = int(num_domains)
        if num_domains < self.num_domains:
            raise ValueError(
                f"Cannot shrink num_domains from {self.num_domains} to {num_domains}."
            )
        if num_domains == self.num_domains:
            return

        old_bias = self.embodiment_bias
        new_bias = nn.Embedding(num_domains, self.hidden_size).to(
            device=old_bias.weight.device,
            dtype=old_bias.weight.dtype,
        )
        nn.init.zeros_(new_bias.weight)
        with torch.no_grad():
            new_bias.weight[: self.num_domains].copy_(old_bias.weight)
        self.embodiment_bias = new_bias

        if self.soft_prompt_hub is not None:
            old_prompt = self.soft_prompt_hub
            new_prompt = nn.Embedding(
                num_domains,
                self.soft_prompt_length * self.hidden_size,
            ).to(
                device=old_prompt.weight.device,
                dtype=old_prompt.weight.dtype,
            )
            nn.init.normal_(new_prompt.weight, std=0.02)
            with torch.no_grad():
                new_prompt.weight[: self.num_domains].copy_(old_prompt.weight)
            self.soft_prompt_hub = new_prompt
        self.num_domains = num_domains

    def _build_attention_mask(
        self,
        context_mask: torch.BoolTensor,
        auxiliary_mask: torch.BoolTensor,
        action_mask: torch.BoolTensor,
    ) -> torch.BoolTensor:
        batch_size, context_len = context_mask.shape
        auxiliary_len = auxiliary_mask.shape[1]
        prompt_start = context_len + auxiliary_len
        action_start = prompt_start + self.soft_prompt_length
        seq_len = action_start + self.num_actions

        structural = torch.zeros(
            seq_len, seq_len, dtype=torch.bool, device=context_mask.device
        )
        structural[:context_len, :context_len] = True
        structural[context_len:prompt_start, :prompt_start] = True
        structural[prompt_start:action_start, :action_start] = True
        structural[action_start:, :] = True

        prompt_valid = torch.ones(
            batch_size,
            self.soft_prompt_length,
            dtype=torch.bool,
            device=context_mask.device,
        )
        key_valid = torch.cat((context_mask, auxiliary_mask, prompt_valid, action_mask), dim=1)
        return structural.unsqueeze(0) & key_valid.unsqueeze(1)

    def _type_embedding(self, token_type: int, reference: torch.Tensor) -> torch.Tensor:
        index = torch.tensor(token_type, device=reference.device)
        return self.token_type_embedding(index).to(dtype=reference.dtype)

    def forward(
        self,
        embodiment_id: torch.LongTensor,
        vlm_hidden_states: Sequence[torch.Tensor],
        vlm_attention_mask: torch.Tensor,
        action_tokens: torch.Tensor,
        action_token_mask: torch.Tensor | None = None,
        auxiliary_features: torch.Tensor | None = None,
        auxiliary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(vlm_hidden_states) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} Qwen hidden states, got {len(vlm_hidden_states)}."
            )
        batch_size, num_actions, hidden_size = action_tokens.shape
        if num_actions != self.num_actions:
            raise ValueError(f"Expected {self.num_actions} actions, got {num_actions}.")
        if hidden_size != self.hidden_size:
            raise ValueError(f"Expected action token width {self.hidden_size}, got {hidden_size}.")
        embodiment_id = embodiment_id.reshape(-1).long()
        if embodiment_id.shape[0] != batch_size:
            raise ValueError(
                "embodiment_id batch size does not match the action batch."
            )
        if bool((embodiment_id < 0).any()) or bool(
            (embodiment_id >= self.num_domains).any()
        ):
            raise IndexError(
                f"embodiment_id must be in [0,{self.num_domains}), got "
                f"{sorted(set(embodiment_id.detach().cpu().tolist()))}."
            )
        if action_token_mask is None:
            action_token_mask = torch.ones(
                batch_size,
                num_actions,
                dtype=torch.bool,
                device=action_tokens.device,
            )
        if action_token_mask.shape != (batch_size, num_actions):
            raise ValueError("action_token_mask must have shape [B,T].")
        action_token_mask = action_token_mask.to(
            device=action_tokens.device, dtype=torch.bool
        )

        action_tokens = action_tokens + self.action_position.to(
            dtype=action_tokens.dtype
        )
        embodiment_bias = (
            self.embodiment_bias(embodiment_id)
            .unsqueeze(1)
            .to(dtype=action_tokens.dtype)
        )
        action_tokens = (
            action_tokens
            + embodiment_bias
            + self._type_embedding(3, action_tokens)
        )

        if self.soft_prompt_hub is not None:
            prompt_tokens = self.soft_prompt_hub(embodiment_id).view(
                batch_size,
                self.soft_prompt_length,
                self.hidden_size,
            )
            prompt_tokens = prompt_tokens.to(dtype=action_tokens.dtype)
            prompt_tokens = (
                prompt_tokens
                + self.prompt_position.to(dtype=action_tokens.dtype)
                + embodiment_bias
                + self._type_embedding(2, action_tokens)
            )
        else:
            prompt_tokens = action_tokens.new_empty(batch_size, 0, self.hidden_size)

        if auxiliary_features is None:
            auxiliary_features = vlm_hidden_states[0].new_empty(
                batch_size,
                0,
                vlm_hidden_states[0].shape[-1],
            )
        auxiliary_tokens = self.auxiliary_projection(
            self.auxiliary_norm(auxiliary_features)
        )
        auxiliary_tokens = auxiliary_tokens.to(dtype=action_tokens.dtype)
        auxiliary_tokens = auxiliary_tokens + self._type_embedding(1, action_tokens)
        if auxiliary_mask is None:
            auxiliary_mask = torch.ones(
                auxiliary_tokens.shape[:2],
                dtype=torch.bool,
                device=action_tokens.device,
            )

        context_mask = vlm_attention_mask.bool()
        auxiliary_mask = auxiliary_mask.bool()
        allowed = self._build_attention_mask(
            context_mask, auxiliary_mask, action_token_mask
        )

        for layer_index, (block, hidden_state) in enumerate(
            zip(self.blocks, vlm_hidden_states, strict=True)
        ):
            context_tokens = self.context_projection(
                self.context_norms[layer_index](hidden_state)
            )
            context_tokens = context_tokens * self.context_scales[layer_index]
            context_tokens = context_tokens.to(dtype=action_tokens.dtype)
            context_tokens = context_tokens + self._type_embedding(0, action_tokens)
            tokens = torch.cat(
                (context_tokens, auxiliary_tokens, prompt_tokens, action_tokens),
                dim=1,
            )
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, allowed, use_reentrant=False)
            else:
                tokens = block(tokens, allowed)
            prompt_start = context_tokens.shape[1] + auxiliary_tokens.shape[1]
            action_start = prompt_start + self.soft_prompt_length
            prompt_tokens = tokens[:, prompt_start:action_start]
            action_tokens = tokens[:, action_start:]

        return self.output_norm(action_tokens)


__all__ = ["LayerwiseActionExpert"]
