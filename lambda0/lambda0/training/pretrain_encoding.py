"""Stage-I prompt construction and assistant-only language labels."""

from __future__ import annotations

import torch

from lambda0.models.hand_state_tokens import HAND_STATE_DIM, HAND_STATE_HORIZON, hand_state_text
from lambda0.models.processing_lambda0 import Lambda0Processor


def _messages(batch: dict) -> tuple[list[list[dict]], list[list[dict]]]:
    prefix, full = [], []
    ids = batch["action_space_id"].tolist()
    if any(value != 0 for value in ids):
        raise ValueError("Stage I accepts only the human_hand_geometry action-space ID.")
    for index, (images, image_mask, instruction) in enumerate(zip(
        batch["image_input"], batch["image_mask"], batch["language_instruction"], strict=True
    )):
        state, target, _ = hand_state_text(
            batch["proprio"][index, :HAND_STATE_DIM].numpy(),
            batch["proprio_valid_mask"][index, :HAND_STATE_DIM].numpy(),
            batch["action"][index, :HAND_STATE_HORIZON, :HAND_STATE_DIM].numpy(),
            batch["action_valid_mask"][index, :HAND_STATE_HORIZON, :HAND_STATE_DIM].numpy(),
        )
        views = [Lambda0Processor._to_pil(image) for image, valid in zip(images, image_mask, strict=True) if valid]
        if not views:
            raise ValueError("Stage I requires an egocentric image for every sample.")
        task = str(instruction).strip().replace("\n", " ")
        if not task:
            raise ValueError("Stage I requires a nonempty language instruction.")
        prompt = (
            f"Task: {task}, State: {state};\n"
            "Predict the next 50 hand states as ordered 138-value discrete frames; use _ for missing coordinates."
        )
        user = {"role": "user", "content": [
            *({"type": "image", "image": view} for view in views),
            {"type": "text", "text": prompt},
        ]}
        prefix.append([user])
        full.append([user, {"role": "assistant", "content": target}])
    return prefix, full


def encode_batch(batch: dict, processor, max_sequence_tokens: int) -> dict[str, torch.Tensor]:
    """Encode a batch, supervising only assistant future-state tokens."""
    prefix_messages, full_messages = _messages(batch)
    options = {"tokenize": True, "return_dict": True, "return_tensors": "pt",
               "processor_kwargs": {"text_kwargs": {"padding": True, "truncation": False}}}
    prefix = processor.apply_chat_template(prefix_messages, add_generation_prompt=True, **options)
    full = processor.apply_chat_template(full_messages, add_generation_prompt=False, **options)
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if required.difference(full):
        raise RuntimeError("Qwen3.5 processor omitted required multimodal inputs: "
                           + ", ".join(sorted(required.difference(full))))
    labels = full["input_ids"].clone()
    for index in range(labels.shape[0]):
        positions = torch.nonzero(full["attention_mask"][index], as_tuple=True)[0]
        prefix_ids = prefix["input_ids"][index][prefix["attention_mask"][index].bool()]
        actual_ids = full["input_ids"][index][positions]
        if len(actual_ids) > max_sequence_tokens:
            raise ValueError(
                f"Stage-I sample has {len(actual_ids)} tokens; increase max_sequence_tokens "
                "and verify model context/memory. Targets are never truncated."
            )
        if len(actual_ids) <= len(prefix_ids) or not torch.equal(actual_ids[:len(prefix_ids)], prefix_ids):
            raise RuntimeError("Qwen3.5 assistant target does not follow the encoded prompt exactly.")
        labels[index].fill_(-100)
        labels[index, positions[len(prefix_ids):]] = actual_ids[len(prefix_ids):]
    return {**{key: value for key, value in full.items() if isinstance(value, torch.Tensor)}, "labels": labels}


__all__ = ["encode_batch"]
