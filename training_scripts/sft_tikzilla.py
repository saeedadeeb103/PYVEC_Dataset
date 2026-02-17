import os
import sys
from pathlib import Path

# Add script directory to sys.path to allow imports from local Datasets/batching
sys.path.append(str(Path(__file__).resolve().parent))
import torch
import argparse

from accelerate import Accelerator
from trl import SFTConfig, SFTTrainer
from batching import make_qwen_base_preprocessor
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from Datasets.create_tikz_datasets_tikzilla import get_huggingface_dataset, get_huggingface_dataset_val

def arg_parser():
    parser = argparse.ArgumentParser(description="Finetune LLMs on TikZ code.")
    parser.add_argument('--model_id', type=str, default="Qwen2.5-3B", help="ID of the LLM to be finetuned.")
    parser.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length of input + output.")
    parser.add_argument('--input_variant', type=str, default="new_caption", help="Input types of the LLM.")
    parser.add_argument('--code_length', type=tuple, default=(100, 4000), help="Min and max TikZ code lengths.")
    parser.add_argument('--source_filter', type=str, default="arxiv_github_tex_synthetic_curated", help="Data types for finetuning.")
    parser.add_argument('--number_samples', type=int, default=100000, help="Number of samples.")
    parser.add_argument('--data_percentage', type=float, default=1.0, help="Percentage of all samples used.")
    parser.add_argument('--relative', type=bool, default=True, help="Switch between data_percentage and number_samples.")
    parser.add_argument('--compiled', type=bool, default=True, help="Use only code that can be compiled.")
    parser.add_argument('--debugged', type=bool, default=True, help="Use additional LLM debugged data.")
    parser.add_argument('--use_lora', type=bool, default=False, help="Employ Low Rank Adaption (LORA).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--lora_alpha', type=int, default=512, help="Scaling factor for LORA.")
    parser.add_argument('--lora_rank', type=int, default=256, help="LORA matrices rank.")
    parser.add_argument('--lora_dropout', type=int, default=0.05, help="LORA dropout value.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for finetuning.")
    parser.add_argument('--device_batch_size', type=int, default=16, help="Batch size per device.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Number steps for gradient accumulation.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for finetuning.")
    parser.add_argument('--max_grad_norm', type=float, default=0.3, help="Value for gradient clipping.")
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Warmup percentage of sheduler.")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help="Learning rate sheduler type.")
    parser.add_argument('--loss_type', type=str, default="dft", help="Loss type for SFT training.")
    parser.add_argument('--work_dir', type=str, required=True, help="Path to tmpdir.")
    return parser.parse_args()

def main():
    args = arg_parser()

    if args.compiled:
        compile_verbose = "only_compiled"
    elif not args.compiled:
        compile_verbose = "compiled_uncompiled"

    if args.debugged:
        debug_verbose = "plus_debugged"
    elif not args.debugged:
        debug_verbose = "not_debugged"

    if args.relative:
        dataset_cache_id = f"{args.model_id}_{args.max_seq_length}_{args.input_variant}_{compile_verbose}_{debug_verbose}_{args.code_length[0]}_{args.code_length[1]}_{args.source_filter}_{args.data_percentage}"
    elif not args.relative:
        dataset_cache_id = f"{args.model_id}_{args.max_seq_length}_{args.input_variant}_{compile_verbose}_{debug_verbose}_{args.code_length[0]}_{args.code_length[1]}_{args.source_filter}_{args.number_samples}"

    if args.use_lora:
        sft_model_id = f"{dataset_cache_id}_{args.learning_rate}_{args.epochs}_{args.device_batch_size}_{args.gradient_accumulation_steps}_lora_{args.lora_rank}_{args.lora_alpha}_{args.loss_type}"
    else:
        sft_model_id = f"{dataset_cache_id}_{args.learning_rate}_{args.epochs}_{args.device_batch_size}_{args.gradient_accumulation_steps}_{args.loss_type}"

    model_path = os.path.join(args.work_dir, "models", args.model_id)
    print(f"Model path: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map="auto",
    )

    if args.use_lora:
        model = prepare_model_for_kbit_training(model)

    full_dataset = get_huggingface_dataset(
        json_dir="data/all", 
        input_variant=args.input_variant, 
        code_length=args.code_length, 
        source_filter=args.source_filter, 
        number_samples=args.number_samples, 
        data_percentage=args.data_percentage, 
        relative=args.relative,
        compiled=args.compiled,
        debugged=args.debugged,
        seed=args.seed
    ).shuffle(args.seed)
    print(f"Dataset length before: {len(full_dataset)}")

    full_dataset_val = get_huggingface_dataset_val(
        json_path=f"captions_new/test_data_gpt4o.json",
    )

    tokenizer_path = os.path.join(args.work_dir, "tokenizer_sft", f"{dataset_cache_id}.arrow")
    print(f"Tokenizer path: {tokenizer_path}")

    preprocessor = make_qwen_base_preprocessor(args.model_id, model_path, args.max_seq_length)
    processed_dataset = full_dataset.map(
        preprocessor,
        batched=True,
        batch_size=256,
        num_proc=32,
        remove_columns=["text", "label"],
        load_from_cache_file=True,
        cache_file_name=tokenizer_path,
        desc="Formatting + Tokenizing + Filtering"
    )
    print(f"Dataset length after: {len(processed_dataset)}")

    processed_dataset_val = full_dataset_val.map(
        preprocessor,
        batched=True,
        batch_size=64,
        num_proc=1,
        remove_columns=["text", "label"],
        desc="Formatting + Tokenizing + Filtering (Validation)"
    )
    print(f"Validation dataset length: {len(processed_dataset_val)}")

    if args.use_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"]
        )
        model = get_peft_model(model, peft_config)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    trained_model_path = os.path.join(args.work_dir, "trained_models_sft", sft_model_id)
    print(f"Trained model path: {trained_model_path}")
    sft_config = SFTConfig(
        output_dir=trained_model_path,
        label_names=["labels"],
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.device_batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
		logging_dir=f"tf_logs/{sft_model_id}",
        report_to="tensorboard",
        eval_strategy="steps",
        eval_steps=500,
        loss_type=args.loss_type
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=processed_dataset,
        eval_dataset=processed_dataset_val
    )
    trainer.train()
    # resume_checkpoint_path = os.path.join(trained_model_path, "checkpoint-15500")
    # trainer.train(resume_checkpoint_path)

if __name__=="__main__":
    main()