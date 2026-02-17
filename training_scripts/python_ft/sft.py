"""SFT training for DeTikZifyv2 and text-only models to output Python/Matplotlib."""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from preprocessing import make_sft_preprocessor, tokenize_detikzify


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT for TikZ→Python inverse graphics model.")
    p.add_argument("--mode", type=str, choices=["text", "multimodal"], default="text")
    p.add_argument("--model-id", type=str, default="Qwen2.5-3B")
    p.add_argument("--detikzify-model", type=str, default="detikzify-v2-8b")
    p.add_argument("--data-dir", type=str, required=True, help="Path to prepared HF dataset")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--scheduler", type=str, default="cosine")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=256)
    p.add_argument("--lora-alpha", type=int, default=512)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", type=str, required=True)
    return p.parse_args()


def train_text_model(args: argparse.Namespace):
    model_path = os.path.join(args.work_dir, "models", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

    if args.use_lora:
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, lora_cfg)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    dd = load_from_disk(args.data_dir)
    train_ds, val_ds = dd["train"], dd["validation"]

    preprocessor = make_sft_preprocessor(model_path, args.max_seq_len, multimodal=False)

    processed_train = train_ds.map(
        preprocessor, batched=True, batch_size=256, num_proc=min(32, os.cpu_count()),
        remove_columns=train_ds.column_names, desc="Tokenizing train",
    )
    processed_val = val_ds.map(
        preprocessor, batched=True, batch_size=64, num_proc=1,
        remove_columns=val_ds.column_names, desc="Tokenizing validation",
    )

    run_id = f"pythonft_text_{args.model_id}_lr{args.lr}_ep{args.epochs}"
    output_path = os.path.join(args.work_dir, "trained_models_sft_python", run_id)

    config = SFTConfig(
        output_dir=output_path,
        label_names=["labels"],
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_seq_len,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        logging_dir=f"tf_logs_python_sft/{run_id}",
        report_to="tensorboard",
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=processed_train,
        eval_dataset=processed_val,
    )
    trainer.train()


def train_multimodal(args: argparse.Namespace):
    from detikzify.model import load as load_model

    model_path = os.path.join(args.work_dir, "models", args.detikzify_model)
    model, processor = load_model(model_name_or_path=model_path, torch_dtype=torch.bfloat16)

    dd = load_from_disk(args.data_dir)
    train_ds, val_ds = dd["train"], dd["validation"]

    tok_fn = partial(tokenize_detikzify, processor=processor, truncation=True, padding="max_length")

    processed_train = train_ds.map(
        tok_fn, batched=True, batch_size=256, num_proc=min(32, os.cpu_count()),
        remove_columns=["image", "text"], desc="Tokenizing train",
    )
    processed_val = val_ds.map(
        tok_fn, batched=True, batch_size=64, num_proc=1,
        remove_columns=["image", "text"], desc="Tokenizing validation",
    )

    run_id = f"pythonft_multimodal_{args.detikzify_model}_lr{args.lr}_ep{args.epochs}"
    output_path = os.path.join(args.work_dir, "trained_models_sft_python", run_id)

    config = TrainingArguments(
        output_dir=output_path,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        logging_dir=f"tf_logs_python_sft/{run_id}",
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=processed_train,
        eval_dataset=processed_val,
    )
    trainer.train()


def main():
    args = parse_args()
    if args.mode == "text":
        train_text_model(args)
    else:
        train_multimodal(args)


if __name__ == "__main__":
    main()
