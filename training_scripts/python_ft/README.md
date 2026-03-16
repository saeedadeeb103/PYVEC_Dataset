# PYVEC Python Fine-Tuning Pipeline

Train LLMs to generate Python/Matplotlib code from figure descriptions.

**Pipeline:** Caption → LLM → Python code → execute → Figure

## Setup

```bash
conda activate uni  # or your env with torch, transformers, trl, peft, accelerate
export PYVEC_WORK="$(pwd)"
```

## 1. Prepare Data

Render images from the dataset JSONL:

```bash
python prepare_data.py
# or on SLURM:
sbatch scripts/prepare_data.sbatch
```

## 2. SFT (Text-Only)

Fine-tune Qwen2.5-3B-Instruct on caption→code pairs with LoRA:

```bash
accelerate launch sft.py
sbatch scripts/train_sft.sbatch
```

Override defaults:

```bash
accelerate launch sft.py model=qwen25_7b training.lr=1e-4 lora.enabled=true
```

## 3. DeTikZify SFT (Image→Code)

Fine-tune DeTikZify-v2.5-8B on image→code (inverse graphics):

```bash
accelerate launch sft_detikzify.py
sbatch scripts/run_sft_detikzify.sbatch
```

### Rejection Sampling (optional)

Generate N candidates per image, keep the best by DINOv2 score, then re-train:

```bash
# Step 1: Generate filtered dataset
sbatch scripts/run_generate_rsft_detikzify.sbatch

# Step 2: Train on filtered data
sbatch scripts/run_rsft_detikzify.sbatch
```

## 4. GRPO (RL Post-Training)

Reinforce the SFT model with execution + DINOv2 visual similarity rewards:

```bash
# Qwen2.5-3B (full fine-tuning)
SFT_RUN_ID=<your_sft_run> sbatch scripts/train_grpo.sbatch

# DeTikZify-8B (LoRA, memory-efficient)
sbatch scripts/train_grpo_detikzify.sbatch
```

Pre-download reward models on login node (compute nodes may lack internet):

```bash
python download_reward_models.py --out models --models dinov2
```

GRPO config supports optional PEFT LoRA for large models:

```bash
accelerate launch grpo.py model.sft_run_id=<RUN> peft.enable=true peft.r=32
```

## 5. Evaluation

### Internal test set (DINOv2 visual comparison)

```bash
# SFT model
sbatch scripts/run_eval_sft_vis.sbatch

# GRPO model (edit MODEL_PATH in the script)
python eval_sft_vis.py --model_path <path> --render_dir data/rendered_images --out_dir eval_output
```

### VisPlotBench (external benchmark)

```bash
sbatch scripts/run_eval_visplotbench.sbatch
```

### DeTikZify model

```bash
sbatch scripts/run_eval_detikzify_vis.sbatch
```

Eval scripts are incremental — they resume from where they left off if interrupted.

## File Structure

```
python_ft/
├── sft.py                    # SFT training (text-only, inverse, combined)
├── sft_detikzify.py          # DeTikZify SFT (image→code)
├── grpo.py                   # GRPO RL post-training
├── rewards.py                # DINOv2, CLIP, DreamSim scorers
├── preprocessing.py          # Prompt templates, tokenization
├── prepare_data.py           # Dataset rendering
├── download_reward_models.py # Pre-download DINOv2/CLIP for offline nodes
├── generate_rejection_samples.py  # Expert iteration sampling
├── eval_sft.py               # Basic eval (exec pass rate)
├── eval_sft_vis.py           # Visual eval (side-by-side + DINOv2 score)
├── eval_visplotbench.py      # VisPlotBench benchmark eval
├── eval_detikzify_vis.py     # DeTikZify visual eval
├── configs/
│   ├── grpo.yaml             # GRPO defaults (includes peft section)
│   ├── sft.yaml              # SFT defaults
│   ├── sft_detikzify.yaml    # DeTikZify SFT defaults
│   ├── model/                # Model presets (qwen25_3b, detikzify_8b, ...)
│   └── reward/               # Reward presets (dinov2, dinov2_local, clip, ...)
└── scripts/                  # SLURM sbatch scripts for all stages
```

## Config (Hydra)

All training scripts use Hydra. Override any value from CLI:

```bash
accelerate launch sft.py model=qwen25_vl_3b training_mode=inverse_graphics
accelerate launch grpo.py model.sft_run_id=<RUN> reward=clip training.lr=1e-5
```

## Requirements

```
torch, transformers, trl, peft, datasets, accelerate
hydra-core, omegaconf, qwen-vl-utils, orjson
wandb, tensorboard, dreamsim, POT
```
