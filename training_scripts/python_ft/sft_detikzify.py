"""SFT for DeTikZify: inverse graphics (image → Python/Matplotlib code).

Fine-tunes the DeTikZify VLM to produce Python/Matplotlib code
instead of TikZ code, given a rendered figure image.

Usage:
    python sft_detikzify.py
    python sft_detikzify.py lora.enabled=true lora.rank=64
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from functools import partial
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import hydra
import torch
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments

log = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_detikzify(model_path: str):
    """Load DeTikZify model + processor, registering custom architecture."""
    from detikzify.model import load, register
    register()

    model, processor = load(
        model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Force slow (PIL-based) image processor to avoid torchvision lanczos bug
    from transformers import AutoProcessor
    slow_processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    processor.image_processor = slow_processor.image_processor
    log.info("Replaced fast image processor with slow (PIL-based) version")

    return model, processor


def apply_lora(model, cfg: DictConfig):
    """Apply LoRA adapters to the DeTikZify model."""
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def tokenize_batch(batch, processor):
    """Tokenize a batch for DeTikZify: image tokens masked in labels."""
    image_token = processor.image_token
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)

    inputs = processor(
        text=batch["text"],
        images=batch["image"],
        max_length=processor.tokenizer.model_max_length,
        pad_to_multiple_of=8,
        add_eos_token=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    inputs["labels"] = copy.deepcopy(inputs["input_ids"])

    for label_ids in inputs["labels"]:
        for idx, label_id in enumerate(label_ids):
            if label_id in {image_token_id, processor.tokenizer.pad_token_id}:
                label_ids[idx] = IGNORE_INDEX

    return inputs


class ImageCodeDataset(TorchDataset):
    """Wraps HuggingFace dataset with lazy image loading for DeTikZify."""

    def __init__(self, hf_dataset, processor, render_dir: str, max_seq_len: int):
        self.processor = processor
        self.render_dir = render_dir
        self.max_seq_len = max_seq_len
        self.has_embedded = "rendered_image" in hf_dataset.column_names

        # Build an index of valid samples without loading images
        self.indices = []
        skipped = 0
        for i in range(len(hf_dataset)):
            if self.has_embedded:
                self.indices.append(i)
            else:
                sid = hf_dataset[i]["sample_id"]
                img_path = os.path.join(render_dir, f"{sid}.png")
                if os.path.isfile(img_path):
                    self.indices.append(i)
                else:
                    skipped += 1

        self.hf_dataset = hf_dataset
        if skipped:
            log.info(f"Skipped {skipped} samples without rendered images")
        log.info(f"ImageCodeDataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.hf_dataset[self.indices[idx]]

        # Load image lazily
        if self.has_embedded and row.get("rendered_image") is not None:
            image = row["rendered_image"].convert("RGB")
        else:
            img_path = os.path.join(
                self.render_dir, f"{row['sample_id']}.png"
            )
            image = Image.open(img_path).convert("RGB")

        code = row["python_code"].strip()

        inputs = tokenize_batch(
            {"text": [code], "image": [image]},
            self.processor,
        )

        result = {k: v.squeeze(0) for k, v in inputs.items()}

        if result["input_ids"].shape[0] > self.max_seq_len:
            result["input_ids"] = result["input_ids"][:self.max_seq_len]
            result["attention_mask"] = result["attention_mask"][:self.max_seq_len]
            result["labels"] = result["labels"][:self.max_seq_len]

        return result


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    padded = {"input_ids": [], "attention_mask": [], "labels": []}
    if "pixel_values" in batch[0]:
        padded["pixel_values"] = []

    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        padded["input_ids"].append(
            torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=0)
        )
        padded["attention_mask"].append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        padded["labels"].append(
            torch.nn.functional.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX)
        )
        if "pixel_values" in item:
            padded["pixel_values"].append(item["pixel_values"])

    result = {
        "input_ids": torch.stack(padded["input_ids"]),
        "attention_mask": torch.stack(padded["attention_mask"]),
        "labels": torch.stack(padded["labels"]),
    }
    if "pixel_values" in padded and padded["pixel_values"]:
        result["pixel_values"] = torch.stack(padded["pixel_values"])

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra.main(config_path="configs", config_name="sft_detikzify", version_base="1.3")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Force CUDA initialization early
    if torch.cuda.is_available():
        log.info(f"CUDA available: {torch.cuda.device_count()} GPU(s) — {torch.cuda.get_device_name(0)}")
    else:
        log.warning("CUDA not available — training will be slow / bf16 may fail")

    # --- Resolve model path ---
    local_path = os.path.join(cfg.paths.work_dir, "models", cfg.model.id)
    if os.path.isdir(local_path):
        model_path = local_path
        log.info(f"Using local model: {model_path}")
    else:
        model_path = cfg.model.get("hf_id", cfg.model.id)
        log.info(f"Using HuggingFace model: {model_path}")

    # --- Load model + processor ---
    model, processor = load_detikzify(model_path)
    log.info(f"Model loaded: {type(model).__name__}")

    if cfg.lora.enabled:
        model = apply_lora(model, cfg)

    # --- Load data ---
    dd = load_from_disk(cfg.paths.data_dir)
    train_ds, val_ds = dd["train"], dd["validation"]
    log.info(f"Raw — Train: {len(train_ds)}, Validation: {len(val_ds)}")

    # --- Build dataset (eval disabled — 8B model too large for eval on single A40) ---
    render_dir = cfg.paths.render_dir
    train_dataset = ImageCodeDataset(
        train_ds, processor, render_dir, cfg.model.max_seq_len
    )

    # --- Compute schedule ---
    steps_per_epoch = max(1, len(train_dataset) // (
        cfg.training.batch_size * cfg.training.gradient_accumulation
    ))
    save_steps = max(50, steps_per_epoch // 4)

    # --- Run ID & output path ---
    lora_tag = f"_lora{cfg.lora.rank}" if cfg.lora.enabled else ""
    run_id = (
        f"pyvec_sft_{cfg.model.id}_inverse"
        f"_lr{cfg.training.lr}_ep{cfg.training.epochs}"
        f"_bs{cfg.training.batch_size}x{cfg.training.gradient_accumulation}"
        f"{lora_tag}"
    )
    output_path = os.path.join(
        cfg.paths.work_dir, "trained_models_sft_python", run_id
    )
    log.info(f"Output: {output_path}")
    log.info(f"Save every {save_steps} steps, ~{steps_per_epoch} steps/epoch (eval disabled)")

    # --- Training config ---

    training_args = TrainingArguments(
        output_dir=output_path,
        seed=cfg.training.seed,
        optim="adamw_torch",
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.scheduler,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,
        logging_dir=f"tf_logs_python_sft/{run_id}",
        logging_steps=10,
        report_to=cfg.logging.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    # --- Build trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # --- Train ---
    trainer.train()

    # --- Save ---
    final_path = os.path.join(output_path, "final")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    log.info(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
