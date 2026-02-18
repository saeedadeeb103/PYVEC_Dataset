"""GRPO post-training with Python execution rewards.

Uses Hydra for configuration. Compatible with HuggingFace Accelerate.

Usage:
    accelerate launch grpo.py model.sft_run_id=<SFT_RUN>
    accelerate launch grpo.py model.sft_run_id=<SFT_RUN> reward=clip
    accelerate launch grpo.py model.sft_run_id=<SFT_RUN> training.lr=1e-5
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from preprocessing import make_grpo_preprocessor
from rewards import CLIPScorer, DINOv2Scorer, DreamSimScorer, make_python_reward

log = logging.getLogger(__name__)


def _is_lora_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def load_model_and_tokenizer(model_path: str):
    """Load model + tokenizer, handling both full checkpoints and LoRA adapters."""
    if _is_lora_adapter(model_path):
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(model_path)
        base_id = peft_cfg.base_model_name_or_path
        log.info(f"LoRA adapter detected — loading base model: {base_id}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        log.info("LoRA weights merged into base model")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        log.info("Loading full model checkpoint")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_scorer(cfg: DictConfig):
    backend = cfg.reward.backend
    model_name = cfg.reward.model_name
    log.info(f"Building scorer: {backend} ({model_name})")
    if backend == "dinov2":
        return DINOv2Scorer(model_name)
    elif backend == "clip":
        return CLIPScorer(model_name)
    elif backend == "dreamsim":
        return DreamSimScorer(model_name)
    elif backend == "dual":
        from rewards import DualScorer
        return DualScorer(model_name)
    raise ValueError(f"Unknown reward backend: {backend}")


def resolve_model_path(cfg: DictConfig) -> str:
    if cfg.model.use_base:
        local = os.path.join(cfg.paths.work_dir, "models", cfg.model.id)
        if os.path.isdir(local):
            return local
        return cfg.model.get("hf_id", cfg.model.id)
    elif cfg.model.checkpoint:
        return os.path.join(
            cfg.paths.work_dir, "trained_models_sft_python",
            cfg.model.sft_run_id, cfg.model.checkpoint,
        )
    else:
        candidate = os.path.join(
            cfg.paths.work_dir, "trained_models_sft_python",
            cfg.model.sft_run_id, "final",
        )
        if os.path.isdir(candidate):
            return candidate
        return os.path.join(
            cfg.paths.work_dir, "trained_models_sft_python", cfg.model.sft_run_id,
        )


@hydra.main(config_path="configs", config_name="grpo", version_base="1.3")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if not cfg.model.use_base and not cfg.model.sft_run_id:
        raise ValueError("Set model.sft_run_id to the SFT run directory, "
                         "or model.use_base=true to skip SFT.")

    # --- Load model ---
    model_path = resolve_model_path(cfg)
    log.info(f"Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)

    # --- Load & preprocess data ---
    dd = load_from_disk(cfg.paths.data_dir)
    train_ds = dd["train"]
    log.info(f"Training samples: {len(train_ds)}")

    preprocessor = make_grpo_preprocessor(
        model_path,
        max_prompt_len=cfg.model.max_seq_len // 4,
        max_completion_len=cfg.model.max_seq_len,
    )
    processed = train_ds.shuffle(seed=cfg.training.seed).map(
        preprocessor, batched=True, batch_size=256,
        num_proc=min(16, os.cpu_count() or 1),
        remove_columns=train_ds.column_names,
        desc="Tokenizing for GRPO",
    )
    log.info(f"After tokenization: {len(processed)} samples")

    # --- EOS / PAD ids for generation ---
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_id = tokenizer.pad_token_id or eos_id

    gen_kwargs = {}
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
        gen_kwargs["pad_token_id"] = pad_id

    # --- Reward function ---
    scorer = build_scorer(cfg)
    reward_fn = make_python_reward(scorer, execution_bonus=cfg.reward.execution_bonus)

    # --- Run ID & output ---
    sft_tag = cfg.model.sft_run_id or cfg.model.id
    run_id = (
        f"pyvec_grpo_{sft_tag}"
        f"_{cfg.reward.backend}"
        f"_lr{cfg.training.lr}"
        f"_gen{cfg.training.num_generations}"
        f"_bs{cfg.training.batch_size}x{cfg.training.gradient_accumulation}"
    )
    output_path = os.path.join(cfg.paths.work_dir, "trained_models_grpo_python", run_id)
    log.info(f"Output: {output_path}")

    # --- GRPO config ---
    config = GRPOConfig(
        output_dir=output_path,
        seed=cfg.training.seed,

        loss_type=cfg.grpo.loss_type,
        scale_rewards=False,

        optim="adamw_torch",
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.scheduler,

        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=cfg.training.gradient_accumulation,

        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        max_completion_length=cfg.model.max_seq_len,
        num_generations=cfg.training.num_generations,
        epsilon=cfg.grpo.epsilon,
        epsilon_high=cfg.grpo.epsilon_high,
        mask_truncated_completions=True,

        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,

        save_steps=50,
        save_total_limit=5,
        logging_dir=f"tf_logs_python_grpo/{run_id}",
        logging_steps=5,
        report_to=cfg.logging.report_to,

        generation_kwargs=gen_kwargs if gen_kwargs else None,
    )

    # --- Train ---
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=processed,
        processing_class=tokenizer,
    )

    if cfg.resume_from:
        trainer.train(cfg.resume_from)
    else:
        trainer.train()

    final_path = os.path.join(output_path, "final")
    trainer.save_model(final_path)
    log.info(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
