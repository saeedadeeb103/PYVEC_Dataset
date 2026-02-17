"""Tokenization and batching for SFT and GRPO training."""

from __future__ import annotations

from functools import partial
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer


SYSTEM_PROMPT = (
    "You are an expert at generating Python Matplotlib code that recreates "
    "scientific figures. Output only valid, self-contained, executable Python code."
)

USER_TEMPLATE_MULTIMODAL = (
    "Convert this scientific figure into Python Matplotlib code that recreates it. "
    "Description: {caption}\n"
    "Output only executable Python code starting with imports."
)

USER_TEMPLATE_TEXT = (
    "Generate Python Matplotlib code that creates the following scientific figure:\n"
    "{caption}\n"
    "Output a complete, self-contained, executable Python script."
)


def build_tokenizer(model_path: str) -> PreTrainedTokenizer:
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _format_sft_messages(caption: str, code: str, multimodal: bool = False) -> list[dict]:
    template = USER_TEMPLATE_MULTIMODAL if multimodal else USER_TEMPLATE_TEXT
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": template.format(caption=caption)},
        {"role": "assistant", "content": code},
    ]


def _format_grpo_prompt(caption: str, multimodal: bool = False) -> list[dict]:
    template = USER_TEMPLATE_MULTIMODAL if multimodal else USER_TEMPLATE_TEXT
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": template.format(caption=caption)},
    ]


def make_sft_preprocessor(
    model_path: str,
    max_seq_len: int = 2048,
    multimodal: bool = False,
):
    tokenizer = build_tokenizer(model_path)

    def process(batch: dict[str, list]) -> dict[str, list]:
        input_ids_out, attention_mask_out, labels_out = [], [], []

        for caption, code in zip(batch["text"], batch["python_code"]):
            caption, code = caption.strip(), code.strip()
            if not code:
                continue

            messages = _format_sft_messages(caption, code, multimodal)

            try:
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                continue

            tokens = tokenizer(full_text, add_special_tokens=False)
            ids = tokens["input_ids"]

            if len(ids) >= max_seq_len:
                continue

            prompt_msgs = messages[:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=False
            )
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

            labels = [-100] * prompt_len + ids[prompt_len:]
            assert len(labels) == len(ids)

            input_ids_out.append(ids)
            attention_mask_out.append([1] * len(ids))
            labels_out.append(labels)

        return {"input_ids": input_ids_out, "attention_mask": attention_mask_out, "labels": labels_out}

    return process


def make_grpo_preprocessor(
    model_path: str,
    max_prompt_len: int = 512,
    max_completion_len: int = 2048,
    multimodal: bool = False,
):
    tokenizer = build_tokenizer(model_path)

    def process(batch: dict[str, list]) -> dict[str, list]:
        prompts, ground_truths, reward_signals = [], [], []

        for caption, code, image in zip(batch["text"], batch["python_code"], batch["image"]):
            caption, code = caption.strip(), code.strip()
            if not code:
                continue

            messages = _format_grpo_prompt(caption, multimodal)

            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                continue

            prompt_tokens = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"].shape[-1]
            code_tokens = tokenizer(code, truncation=False, return_tensors="pt")["input_ids"].shape[-1]

            if prompt_tokens > max_prompt_len or code_tokens > max_completion_len:
                continue

            prompts.append(text)
            ground_truths.append(code)
            reward_signals.append(image)

        return {"prompt": prompts, "ground_truth": ground_truths, "reward_signal": reward_signals}

    return process


def tokenize_detikzify(batch: dict[str, list], processor: Any, **kwargs) -> dict[str, list]:
    from detikzify.model import load as load_model

    input_ids_out, attention_mask_out, labels_out = [], [], []

    for image, caption, code in zip(batch["image"], batch["text"], batch["python_code"]):
        caption, code = caption.strip(), code.strip()
        if not code:
            continue

        prompt = USER_TEMPLATE_MULTIMODAL.format(caption=caption)
        full_text = f"{prompt}\n{code}"

        try:
            encoded = processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                truncation=kwargs.get("truncation", True),
                padding=kwargs.get("padding", "max_length"),
            )
        except Exception:
            continue

        ids = encoded["input_ids"].squeeze(0).tolist()
        mask = encoded["attention_mask"].squeeze(0).tolist()

        prompt_encoded = processor(text=prompt, images=image, return_tensors="pt", truncation=True)
        prompt_len = prompt_encoded["input_ids"].shape[-1]

        labels = [-100] * prompt_len + ids[prompt_len:]
        if len(labels) != len(ids):
            labels = [-100] * len(ids)

        input_ids_out.append(ids)
        attention_mask_out.append(mask)
        labels_out.append(labels)

    return {"input_ids": input_ids_out, "attention_mask": attention_mask_out, "labels": labels_out}
