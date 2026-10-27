from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from .state_conditioning import (
    CONTINUOUS_ACTION_EXPERT,
    DEFAULT_STATE_DISCRETE_BINS,
    DISCRETE_VLM,
    inject_discrete_state_into_instructions,
    validate_state_conditioning,
)


class Lambda0Processor:
    """Wraps the official Qwen3.5 processor for main and auxiliary camera views."""

    text_padding_multiple = 64
    required_main_keys = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    }

    def __init__(self, processor) -> None:
        self.processor = processor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any):
        return cls(
            AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        )

    def save_pretrained(self, save_directory: str, **kwargs: Any):
        return self.processor.save_pretrained(save_directory, **kwargs)

    @staticmethod
    def _to_pil(image: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if image.ndim != 3:
                raise ValueError(
                    f"Expected a rank-3 image tensor, got shape {tuple(image.shape)}."
                )
            if image.shape[0] in (1, 3, 4):
                image = image.permute(1, 2, 0)
            array = image.numpy()
        else:
            array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"Expected an HWC image, got shape {array.shape}.")
        if np.issubdtype(array.dtype, np.floating):
            if array.size and float(np.nanmax(array)) > 1.0:
                array = np.clip(array, 0.0, 255.0).round().astype(np.uint8)
            else:
                array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array[..., :3]).convert("RGB")

    def _encode_batch(
        self,
        image_batch: Sequence[Sequence[Image.Image | np.ndarray | torch.Tensor]],
        language_instruction: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if len(image_batch) != len(language_instruction):
            raise ValueError("Image and language batch sizes must match.")
        if not image_batch:
            raise ValueError("At least one sample is required.")

        conversations = []
        auxiliary_images = []
        auxiliary_owners = []
        for batch_index, (views, instruction) in enumerate(
            zip(image_batch, language_instruction, strict=True)
        ):
            if not views:
                raise ValueError(
                    f"Sample {batch_index} has no valid main camera image."
                )
            main_image = self._to_pil(views[0])
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": main_image},
                            {"type": "text", "text": str(instruction)},
                        ],
                    }
                ]
            )
            for auxiliary_image in views[1:]:
                auxiliary_images.append(self._to_pil(auxiliary_image))
                auxiliary_owners.append(batch_index)

        main_inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "text_kwargs": {
                    "padding": True,
                    "pad_to_multiple_of": self.text_padding_multiple,
                }
            },
        )
        outputs = {
            key: value
            for key, value in main_inputs.items()
            if isinstance(value, torch.Tensor)
        }
        missing = self.required_main_keys.difference(outputs)
        if missing:
            raise RuntimeError(
                "The official Qwen3.5 processor did not return required fields: "
                + ", ".join(sorted(missing))
            )

        if auxiliary_images:
            auxiliary_inputs = self.processor.image_processor(
                images=auxiliary_images,
                return_tensors="pt",
            )
            if (
                "pixel_values" not in auxiliary_inputs
                or "image_grid_thw" not in auxiliary_inputs
            ):
                raise RuntimeError(
                    "Qwen3.5 image processor returned incomplete auxiliary inputs."
                )
            outputs.update(
                {
                    "aux_pixel_values": auxiliary_inputs["pixel_values"],
                    "aux_image_grid_thw": auxiliary_inputs["image_grid_thw"],
                    "aux_image_owner": torch.tensor(auxiliary_owners, dtype=torch.long),
                }
            )
        return outputs

    @staticmethod
    def _condition_instructions(
        language_instruction: Sequence[str],
        proprio: np.ndarray | torch.Tensor | None,
        proprio_valid_mask: np.ndarray | torch.Tensor | None,
        state_conditioning_mode: str,
        state_discrete_bins: int,
    ) -> list[str]:
        mode, bins = validate_state_conditioning(
            state_conditioning_mode,
            state_discrete_bins,
        )
        instructions = list(language_instruction)
        if mode == CONTINUOUS_ACTION_EXPERT:
            return instructions
        if mode != DISCRETE_VLM:
            raise AssertionError(f"Unhandled state conditioning mode: {mode}")
        if proprio is None or proprio_valid_mask is None:
            raise ValueError(
                "Discrete VLM state conditioning requires proprio and "
                "proprio_valid_mask."
            )
        return inject_discrete_state_into_instructions(
            instructions,
            proprio,
            proprio_valid_mask,
            num_bins=bins,
        )

    def prepare_batch(
        self,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        language_instruction: Sequence[str],
        proprio: np.ndarray | torch.Tensor | None = None,
        proprio_valid_mask: np.ndarray | torch.Tensor | None = None,
        state_conditioning_mode: str = "discrete_vlm",
        state_discrete_bins: int = DEFAULT_STATE_DISCRETE_BINS,
    ) -> dict[str, torch.Tensor]:
        if isinstance(image_input, torch.Tensor) and image_input.ndim != 5:
            raise ValueError(
                f"image_input must have shape [B,V,C,H,W], got {tuple(image_input.shape)}."
            )
        if image_mask.ndim != 2 or len(image_input) != image_mask.shape[0]:
            raise ValueError("image_mask must have shape [B,V].")
        image_batch = []
        for images, mask in zip(image_input, image_mask.bool(), strict=True):
            image_batch.append(
                [image for image, valid in zip(images, mask, strict=True) if valid]
            )
        instructions = self._condition_instructions(
            language_instruction,
            proprio,
            proprio_valid_mask,
            state_conditioning_mode,
            state_discrete_bins,
        )
        return self._encode_batch(image_batch, instructions)

    def __call__(
        self,
        images: Sequence[Any] | Sequence[Sequence[Any]],
        language_instruction: str | Sequence[str],
        proprio: np.ndarray | torch.Tensor | None = None,
        proprio_valid_mask: np.ndarray | torch.Tensor | None = None,
        state_conditioning_mode: str = "discrete_vlm",
        state_discrete_bins: int = DEFAULT_STATE_DISCRETE_BINS,
    ) -> dict[str, torch.Tensor]:
        if isinstance(language_instruction, str):
            instructions = [language_instruction]
        else:
            instructions = list(language_instruction)
        if not images:
            raise ValueError("images cannot be empty.")
        first = images[0]
        if isinstance(first, (Image.Image, np.ndarray, torch.Tensor)):
            image_batch = [list(images)]
        else:
            image_batch = [list(sample) for sample in images]
        conditioned_instructions = self._condition_instructions(
            instructions,
            proprio,
            proprio_valid_mask,
            state_conditioning_mode,
            state_discrete_bins,
        )
        return self._encode_batch(image_batch, conditioned_instructions)


__all__ = ["Lambda0Processor"]
