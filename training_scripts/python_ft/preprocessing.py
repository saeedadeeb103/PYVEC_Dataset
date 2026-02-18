"""Tokenization, prompt formatting, and batching for SFT and GRPO training.

Supports three training modes:
  - text_only:         caption → code          (standard SFT with text-only LLM)
  - inverse_graphics:  image → code            (VLM, self-supervised, DeTikZify-style)
  - combined:          image + caption → code   (VLM, TikZero+ style)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Prompt templates — text-only
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert scientific visualization programmer. "
    "Given a description of a figure, you write complete, self-contained, "
    "executable Python code using Matplotlib (and optionally NumPy, SciPy, "
    "or Seaborn) that accurately recreates the described figure. "
    "Output only valid Python code with no explanations or markdown."
)

USER_TEMPLATE = (
    "Generate Python code that creates the following scientific figure:\n\n"
    "{caption}\n\n"
    "Requirements:\n"
    "- Use matplotlib.pyplot (import as plt)\n"
    "- The code must be complete and self-contained\n"
    "- Include all necessary imports\n"
    "- Do not call plt.show()"
)


# ---------------------------------------------------------------------------
# Prompt templates — multimodal (inverse graphics / combined)
# ---------------------------------------------------------------------------
INVERSE_GRAPHICS_SYSTEM_PROMPT = (
    "You are an expert at analyzing scientific figures and writing "
    "Python/Matplotlib code to reproduce them. Given an image of a figure, "
    "you produce complete, self-contained, executable Python code that "
    "accurately recreates the figure. Output only valid Python code "
    "with no explanations or markdown."
)

COMBINED_SYSTEM_PROMPT = (
    "You are an expert at analyzing scientific figures and writing "
    "Python/Matplotlib code to reproduce them. Given an image of a figure "
    "and its description, you produce complete, self-contained, executable "
    "Python code that accurately recreates the figure. Output only valid "
    "Python code with no explanations or markdown."
)

INVERSE_GRAPHICS_USER_TEXT = (
    "Analyze this scientific figure and generate the complete "
    "Python/Matplotlib code to recreate it.\n\n"
    "Requirements:\n"
    "- Use matplotlib.pyplot (import as plt)\n"
    "- The code must be complete and self-contained\n"
    "- Include all necessary imports\n"
    "- Do not call plt.show()"
)

COMBINED_USER_TEMPLATE = (
    "Analyze this scientific figure and generate the complete "
    "Python/Matplotlib code to recreate it.\n\n"
    "Figure description: {caption}\n\n"
    "Requirements:\n"
    "- Use matplotlib.pyplot (import as plt)\n"
    "- The code must be complete and self-contained\n"
    "- Include all necessary imports\n"
    "- Do not call plt.show()"
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def build_tokenizer(model_path: str) -> PreTrainedTokenizer:
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Message formatting — text-only
# ---------------------------------------------------------------------------
def format_sft_messages(caption: str, code: str) -> list[dict]:
    """Full conversation for text-only SFT: system + user + assistant."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(caption=caption)},
        {"role": "assistant", "content": code},
    ]


def format_grpo_prompt(caption: str) -> list[dict]:
    """Prompt-only for GRPO: system + user (model generates the rest)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(caption=caption)},
    ]


# ---------------------------------------------------------------------------
# Message formatting — multimodal (VLM)
# ---------------------------------------------------------------------------
def format_inverse_graphics_messages(image_path: str, code: str) -> list[dict]:
    """Multimodal SFT: image → code (inverse graphics, self-supervised).

    Following TikZero's approach: the model learns to synthesize graphics
    programs from their compiled visual representations. No captions needed.
    """
    return [
        {"role": "system", "content": [
            {"type": "text", "text": INVERSE_GRAPHICS_SYSTEM_PROMPT},
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": INVERSE_GRAPHICS_USER_TEXT},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": code},
        ]},
    ]


def format_combined_messages(image_path: str, caption: str, code: str) -> list[dict]:
    """Multimodal SFT: image + caption → code (TikZero+ style).

    Combines the inverse graphics model with caption conditioning.
    This is the approach from TikZero+ Sec. 5.2 (iii): fine-tune
    with both image and tokenized caption for best performance.
    """
    return [
        {"role": "system", "content": [
            {"type": "text", "text": COMBINED_SYSTEM_PROMPT},
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": COMBINED_USER_TEMPLATE.format(caption=caption)},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": code},
        ]},
    ]


# ---------------------------------------------------------------------------
# SFT preprocessor — text-only
# ---------------------------------------------------------------------------
def make_sft_preprocessor(model_path: str, max_seq_len: int = 2048):
    """Build a batched preprocessor for text-only SFT training.

    Masks the prompt tokens with -100 so the model only learns on completions.
    Filters out samples that exceed max_seq_len.
    """
    tokenizer = build_tokenizer(model_path)

    def process(batch: dict[str, list]) -> dict[str, list]:
        input_ids_out, attention_mask_out, labels_out = [], [], []

        for caption, code in zip(batch["text"], batch["python_code"]):
            caption, code = caption.strip(), code.strip()
            if not code:
                continue

            messages = format_sft_messages(caption, code)

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
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

            labels = [-100] * prompt_len + ids[prompt_len:]
            assert len(labels) == len(ids)

            input_ids_out.append(ids)
            attention_mask_out.append([1] * len(ids))
            labels_out.append(labels)

        return {
            "input_ids": input_ids_out,
            "attention_mask": attention_mask_out,
            "labels": labels_out,
        }

    return process


# ---------------------------------------------------------------------------
# SFT preprocessor — multimodal (builds messages dataset for VLM)
# ---------------------------------------------------------------------------
def build_multimodal_sft_dataset(
    hf_dataset,
    render_dir: str,
    training_mode: str = "inverse_graphics",
):
    """Convert a HF dataset to a messages-format dataset for VLM SFT.

    Constructs image paths from render_dir + sample_id and builds
    the appropriate message format based on training_mode.

    Args:
        hf_dataset: HuggingFace dataset with 'sample_id', 'text', 'python_code'
        render_dir: directory containing rendered PNGs ({sample_id}.png)
        training_mode: 'inverse_graphics' or 'combined'

    Returns:
        HF dataset with 'messages' column ready for SFTTrainer + VLM processor
    """
    render_path = Path(render_dir)

    def _build_messages(batch):
        all_messages = []
        for i in range(len(batch["sample_id"])):
            sid = batch["sample_id"][i]
            code = batch["python_code"][i].strip()
            img_path = str(render_path / f"{sid}.png")

            if not code:
                continue
            if not Path(img_path).exists():
                continue

            if training_mode == "inverse_graphics":
                msgs = format_inverse_graphics_messages(img_path, code)
            else:  # combined
                caption = batch["text"][i].strip()
                if not caption:
                    continue
                msgs = format_combined_messages(img_path, caption, code)

            all_messages.append(msgs)

        return {"messages": all_messages}

    return hf_dataset.map(
        _build_messages,
        batched=True,
        batch_size=256,
        remove_columns=hf_dataset.column_names,
        desc=f"Building {training_mode} messages",
    )


# ---------------------------------------------------------------------------
# GRPO preprocessor
# ---------------------------------------------------------------------------
def make_grpo_preprocessor(
    model_path: str,
    max_prompt_len: int = 512,
    max_completion_len: int = 2048,
):
    """Build a batched preprocessor for GRPO training.

    Returns prompt text, ground-truth code, and reward signal (rendered image)
    for each valid sample.
    """
    tokenizer = build_tokenizer(model_path)

    def process(batch: dict[str, list]) -> dict[str, list]:
        prompts, ground_truths, reward_signals = [], [], []

        for caption, code, image in zip(
            batch["text"], batch["python_code"], batch["rendered_image"]
        ):
            caption, code = caption.strip(), code.strip()
            if not code:
                continue

            messages = format_grpo_prompt(caption)

            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                continue

            prompt_tokens = len(tokenizer(text, truncation=False)["input_ids"])
            code_tokens = len(tokenizer(code, truncation=False)["input_ids"])

            if prompt_tokens > max_prompt_len or code_tokens > max_completion_len:
                continue

            prompts.append(text)
            ground_truths.append(code)
            reward_signals.append(image)

        return {
            "prompt": prompts,
            "ground_truth": ground_truths,
            "reward_signal": reward_signals,
        }

    return process
