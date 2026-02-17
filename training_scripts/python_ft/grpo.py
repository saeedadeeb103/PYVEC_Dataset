"""GRPO post-training with Python execution rewards for inverse graphics models."""

from __future__ import annotations

import argparse
import os

import torch
from datasets import concatenate_datasets, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import GRPOConfig, GRPOTrainer

from preprocessing import make_grpo_preprocessor
from rewards import (
    CLIPScorer,
    DreamSimScorer,
    VisionScorer,
    make_python_reward,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO post-training with Python execution rewards.")
    p.add_argument("--model-id", type=str, required=True, help="SFT model directory name")
    p.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to load")
    p.add_argument("--base-model", action="store_true", help="Use base model instead of SFT checkpoint")
    p.add_argument("--data-dir", type=str, required=True, help="Path to prepared HF dataset")
    p.add_argument("--reward-backend", type=str, choices=["detikzify", "clip", "dreamsim"], default="clip")
    p.add_argument("--reward-model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=9)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon-high", type=float, default=0.28)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--scheduler", type=str, default="constant")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", type=str, required=True)
    p.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint path")
    return p.parse_args()


def build_scorer(args: argparse.Namespace):
    if args.reward_backend == "detikzify":
        return VisionScorer(args.reward_model)
    elif args.reward_backend == "clip":
        return CLIPScorer(args.reward_model)
    elif args.reward_backend == "dreamsim":
        return DreamSimScorer(args.reward_model)
    raise ValueError(f"Unknown backend: {args.reward_backend}")


def main():
    args = parse_args()

    if args.base_model:
        model_path = os.path.join(args.work_dir, "models", args.model_id)
    elif args.checkpoint:
        model_path = os.path.join(args.work_dir, "trained_models_sft_python", args.model_id, args.checkpoint)
    else:
        model_path = os.path.join(args.work_dir, "trained_models_sft_python", args.model_id)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dd = load_from_disk(args.data_dir)
    train_ds = dd["train"]

    preprocessor = make_grpo_preprocessor(model_path, max_prompt_len=args.max_seq_len // 4, max_completion_len=args.max_seq_len)
    processed = train_ds.shuffle(seed=args.seed).map(
        preprocessor, batched=True, batch_size=256, num_proc=min(32, os.cpu_count()),
        remove_columns=train_ds.column_names, desc="Tokenizing",
    )

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_id = tokenizer.pad_token_id or eos_id

    if eos_id is not None:
        gen_config = GenerationConfig(
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            eos_token_id=eos_id, pad_token_id=pad_id,
        )
    else:
        gen_config = GenerationConfig(
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_seq_len,
        )

    scorer = build_scorer(args)
    reward_fn = make_python_reward(scorer)

    run_id = f"pythonft_grpo_{args.model_id}_{args.reward_backend}_lr{args.lr}_gen{args.num_generations}"
    output_path = os.path.join(args.work_dir, "trained_models_grpo_python", run_id)

    config = GRPOConfig(
        output_dir=output_path,
        loss_type="dr_grpo",
        scale_rewards=False,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=False,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_prompt_length=args.max_seq_len // 4,
        max_completion_length=args.max_seq_len,
        num_generations=args.num_generations,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        weight_decay=args.weight_decay,
        mask_truncated_completions=True,
        logging_dir=f"tf_logs_python_grpo/{run_id}",
        report_to="tensorboard",
        save_steps=50,
        generation_kwargs=gen_config.to_dict(),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=processed,
        processing_class=tokenizer,
    )

    if args.resume_from:
        trainer.train(args.resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
