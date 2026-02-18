"""SFT training with three modes: text-only, inverse graphics, and combined.

Modes (inspired by TikZero+, adapted for Python/Matplotlib):
  text_only         — caption → code          (text-only LLM, AutomaTikZ-style)
  inverse_graphics  — image → code            (VLM, self-supervised, DeTikZify-style)
  combined          — image + caption → code   (VLM, TikZero+ style)

The TikZero+ pipeline:
  1. Train inverse_graphics first  (model learns image → code on ALL samples)
  2. Fine-tune in combined mode    (adds caption understanding, TikZero+ Sec. 5.2)

Usage:
    # Text-only (default)
    accelerate launch sft.py

    # Inverse graphics (VLM, self-supervised)
    accelerate launch sft.py training_mode=inverse_graphics model=qwen25_vl_3b

    # Combined (VLM + captions, TikZero+ style)
    accelerate launch sft.py training_mode=combined model=qwen25_vl_3b

    # Two-stage TikZero+ pipeline:
    accelerate launch sft.py training_mode=inverse_graphics model=qwen25_vl_3b
    accelerate launch sft.py training_mode=combined model=qwen25_vl_3b \\
        resume_from=<inverse_graphics_checkpoint>
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import hydra
import torch
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

log = logging.getLogger(__name__)

VALID_MODES = {"text_only", "inverse_graphics", "combined"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _register_detikzify():
    """Register DeTikZify custom architecture with transformers Auto classes."""
    try:
        from detikzify.model import register
        register()
        log.info("Registered DeTikZify architecture")
    except Exception as e:
        log.warning(f"Could not register DeTikZify: {e}")


def load_model(cfg: DictConfig, model_path: str):
    """Load model: text-only LLM, Qwen VLM, or DeTikZify VLM."""
    is_vlm = cfg.model.get("is_vlm", False)
    is_detikzify = cfg.model.get("is_detikzify", False)

    if is_detikzify:
        _register_detikzify()
        from transformers import AutoModelForImageTextToText
        log.info(f"Loading DeTikZify VLM from {model_path}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        )
    elif is_vlm:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = getattr(hf_config, "architectures", [""])[0]

        if "Qwen2_5" in arch or "Qwen2.5" in cfg.model.id:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass
        else:
            from transformers import Qwen2VLForConditionalGeneration as VLMClass

        log.info(f"Loading VLM ({VLMClass.__name__}) from {model_path}")
        model = VLMClass.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        )
    else:
        from transformers import AutoModelForCausalLM
        log.info(f"Loading LLM from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )

    return model


def load_processor(cfg: DictConfig, model_path: str):
    """Load processor for VLM (handles images + text) or tokenizer for LLM."""
    is_vlm = cfg.model.get("is_vlm", False)
    is_detikzify = cfg.model.get("is_detikzify", False)

    if is_detikzify:
        _register_detikzify()
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        return processor
    elif is_vlm:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        return processor
    else:
        return None


def apply_lora(model, cfg: DictConfig):
    """Apply LoRA adapters to the model."""
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
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Data preparation per mode
# ---------------------------------------------------------------------------
def prepare_text_only(cfg: DictConfig, train_ds, val_ds, model_path: str):
    """Text-only: tokenize with prompt masking (caption → code)."""
    from preprocessing import make_sft_preprocessor

    preprocessor = make_sft_preprocessor(model_path, cfg.model.max_seq_len)

    processed_train = train_ds.map(
        preprocessor, batched=True, batch_size=256,
        num_proc=min(16, os.cpu_count() or 1),
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    processed_val = val_ds.map(
        preprocessor, batched=True, batch_size=64, num_proc=1,
        remove_columns=val_ds.column_names,
        desc="Tokenizing validation",
    )
    log.info(f"After tokenization — Train: {len(processed_train)}, Val: {len(processed_val)}")
    return processed_train, processed_val


def prepare_multimodal(cfg: DictConfig, train_ds, val_ds):
    """Multimodal: build messages dataset with image paths for VLM."""
    from preprocessing import build_multimodal_sft_dataset

    render_dir = cfg.paths.render_dir
    mode = cfg.training_mode

    log.info(f"Building {mode} messages (render_dir={render_dir})")
    processed_train = build_multimodal_sft_dataset(train_ds, render_dir, mode)
    processed_val = build_multimodal_sft_dataset(val_ds, render_dir, mode)
    log.info(f"Multimodal — Train: {len(processed_train)}, Val: {len(processed_val)}")
    return processed_train, processed_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra.main(config_path="configs", config_name="sft", version_base="1.3")
def main(cfg: DictConfig):
    mode = cfg.training_mode
    if mode not in VALID_MODES:
        raise ValueError(f"training_mode must be one of {VALID_MODES}, got '{mode}'")

    is_vlm = cfg.model.get("is_vlm", False)
    if mode in ("inverse_graphics", "combined") and not is_vlm:
        raise ValueError(
            f"training_mode='{mode}' requires a VLM model. "
            f"Use model=qwen25_vl_3b or model=qwen25_vl_7b"
        )

    log.info(f"Training mode: {mode}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Resolve model path: local dir first, then HuggingFace hub ---
    local_path = os.path.join(cfg.paths.work_dir, "models", cfg.model.id)

    # For combined mode with resume: may point to an inverse_graphics checkpoint
    if cfg.resume_from and os.path.isdir(cfg.resume_from):
        model_path = cfg.resume_from
        log.info(f"Resuming from checkpoint: {model_path}")
    elif os.path.isdir(local_path):
        model_path = local_path
        log.info(f"Using local model: {model_path}")
    else:
        model_path = cfg.model.get("hf_id", cfg.model.id)
        log.info(f"Local model not found at {local_path}, using HuggingFace: {model_path}")

    # --- Load model ---
    model = load_model(cfg, model_path)

    if cfg.lora.enabled:
        model = apply_lora(model, cfg)

    # --- Load processor (VLM) or None (text-only) ---
    processor = load_processor(cfg, model_path)

    # --- Load data ---
    dd = load_from_disk(cfg.paths.data_dir)
    train_ds, val_ds = dd["train"], dd["validation"]
    log.info(f"Raw — Train: {len(train_ds)}, Validation: {len(val_ds)}")

    # --- Prepare data based on mode ---
    if mode == "text_only":
        processed_train, processed_val = prepare_text_only(
            cfg, train_ds, val_ds, model_path
        )
    else:
        processed_train, processed_val = prepare_multimodal(
            cfg, train_ds, val_ds
        )

    # --- Compute eval schedule ---
    steps_per_epoch = max(1, len(processed_train) // (
        cfg.training.batch_size * cfg.training.gradient_accumulation
    ))
    eval_steps = max(50, steps_per_epoch // 4)

    # --- Run ID & output path ---
    lora_tag = f"_lora{cfg.lora.rank}" if cfg.lora.enabled else ""
    mode_tag = "" if mode == "text_only" else f"_{mode}"
    run_id = (
        f"pyvec_sft_{cfg.model.id}{mode_tag}"
        f"_lr{cfg.training.lr}_ep{cfg.training.epochs}"
        f"_bs{cfg.training.batch_size}x{cfg.training.gradient_accumulation}"
        f"{lora_tag}"
    )
    output_path = os.path.join(cfg.paths.work_dir, "trained_models_sft_python", run_id)
    log.info(f"Output: {output_path}")
    log.info(f"Eval every {eval_steps} steps, ~{steps_per_epoch} steps/epoch")

    # --- Training config ---
    packing = cfg.training.packing if mode == "text_only" else False

    sft_kwargs = dict(
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
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        max_length=cfg.model.max_seq_len,

        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=1,

        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_accumulation_steps=1,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_dir=f"tf_logs_python_sft/{run_id}",
        logging_steps=10,
        report_to=cfg.logging.report_to,
    )

    if mode == "text_only":
        sft_kwargs["label_names"] = ["labels"]
        sft_kwargs["packing"] = packing
    else:
        sft_kwargs["dataset_text_field"] = "messages"
        sft_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    config = SFTConfig(**sft_kwargs)

    # --- Build trainer ---
    trainer_kwargs = dict(
        model=model,
        args=config,
        train_dataset=processed_train,
        eval_dataset=processed_val,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    if processor is not None:
        trainer_kwargs["processing_class"] = processor

    trainer = SFTTrainer(**trainer_kwargs)

    # --- Train ---
    if cfg.resume_from and not os.path.isdir(cfg.resume_from):
        trainer.train(cfg.resume_from)
    else:
        trainer.train()

    # --- Save ---
    final_path = os.path.join(output_path, "final")
    trainer.save_model(final_path)
    if processor is not None:
        processor.save_pretrained(final_path)
    log.info(f"Best model saved to {output_path}")
    log.info(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
