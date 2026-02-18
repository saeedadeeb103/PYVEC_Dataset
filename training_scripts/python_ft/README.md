# PYVEC Python/Matplotlib Fine-Tuning Pipeline

Train language models to generate scientific figures from natural language descriptions.

**Pipeline:** `Natural Language → LLM → Python/Matplotlib code → execute → Scientific Figure`

## Training Modes

Inspired by [TikZero+](https://arxiv.org/abs/2503.11509), the pipeline supports three training modes:

| Mode | Input → Output | Model | Description |
|------|---------------|-------|-------------|
| `text_only` | caption → code | Text LLM (Qwen2.5-3B) | Standard SFT (AutomaTikZ-style) |
| `inverse_graphics` | image → code | VLM (Qwen2.5-VL-3B) | Self-supervised, DeTikZify-style |
| `combined` | image + caption → code | VLM (Qwen2.5-VL-3B) | TikZero+ style |

### TikZero+ Two-Stage Pipeline

Following TikZero+ Sec. 5.2(iii), the recommended multimodal pipeline is:

1. **Stage 1 — Inverse Graphics**: Train the VLM on `image → code` using ALL samples (self-supervised, no captions needed). This teaches the model the image-to-code mapping.

2. **Stage 2 — Combined**: Fine-tune the Stage 1 model on `image + caption → code`. This adds caption understanding while preserving the inverse graphics capability. Use a lower LR (1e-5) and fewer epochs.

```bash
# Stage 1: Inverse Graphics
sbatch scripts/train_sft_inverse.sbatch

# Stage 2: Combined (pass Stage 1 checkpoint)
INVERSE_CHECKPOINT=trained_models_sft_python/<stage1_run>/final \
  sbatch scripts/train_sft_combined.sbatch
```

## Architecture

```
python_ft/
├── configs/                        # Hydra YAML configuration
│   ├── prepare_data.yaml           #   data preparation defaults
│   ├── sft.yaml                    #   SFT training defaults (all modes)
│   ├── grpo.yaml                   #   GRPO training defaults
│   ├── model/                      #   model presets
│   │   ├── qwen25_3b.yaml          #     text-only LLM
│   │   ├── qwen25_7b.yaml          #     text-only LLM (larger)
│   │   ├── qwen25_vl_3b.yaml       #     VLM for multimodal modes
│   │   └── qwen25_vl_7b.yaml       #     VLM (larger)
│   └── reward/                     #   reward scorer presets
│       ├── dinov2.yaml
│       ├── clip.yaml
│       └── dreamsim.yaml
├── scripts/                        # SLURM sbatch scripts
│   ├── prepare_data.sbatch         #   CPU-only data rendering
│   ├── train_sft.sbatch            #   1x A40 — text-only SFT
│   ├── train_sft_4gpu.sbatch       #   4x A40 — text-only SFT
│   ├── train_sft_inverse.sbatch    #   1x A40 — inverse graphics (Stage 1)
│   ├── train_sft_combined.sbatch   #   1x A40 — combined (Stage 2)
│   ├── train_grpo.sbatch           #   1x A40 — GRPO
│   └── train_grpo_4gpu.sbatch      #   4x A40 — GRPO
├── prepare_data.py                 # JSONL → rendered images + HF Arrow dataset
├── sft.py                          # SFT (text-only, inverse graphics, combined)
├── grpo.py                         # GRPO reinforcement learning
├── rewards.py                      # Reward functions (DINOv2, CLIP, DreamSim)
└── preprocessing.py                # Prompt templates and tokenization
```

## Quick Start

### 1. Prepare Data (render images)

```bash
python prepare_data.py
python prepare_data.py rendering.dpi=200 rendering.workers=8
sbatch scripts/prepare_data.sbatch
```

### 2a. Text-Only SFT (default)

```bash
accelerate launch sft.py
accelerate launch sft.py model=qwen25_7b training.lr=1e-4 lora.enabled=true
sbatch scripts/train_sft.sbatch
```

### 2b. Inverse Graphics SFT (Stage 1)

```bash
accelerate launch sft.py training_mode=inverse_graphics model=qwen25_vl_3b
sbatch scripts/train_sft_inverse.sbatch
```

### 2c. Combined SFT (Stage 2 — TikZero+ style)

```bash
# From scratch (base VLM)
accelerate launch sft.py training_mode=combined model=qwen25_vl_3b

# From inverse graphics checkpoint (recommended TikZero+ pipeline)
accelerate launch sft.py training_mode=combined model=qwen25_vl_3b \
    resume_from=trained_models_sft_python/<inverse_run>/final

# SLURM
INVERSE_CHECKPOINT=<path_to_stage1_final> sbatch scripts/train_sft_combined.sbatch
```

### 3. GRPO Post-Training

```bash
accelerate launch grpo.py model.sft_run_id=<SFT_RUN>
accelerate launch grpo.py model.sft_run_id=<SFT_RUN> reward=clip
SFT_RUN_ID=<your_sft_run> sbatch scripts/train_grpo.sbatch
```

## Hydra Configuration

All scripts use [Hydra](https://hydra.cc/) for composable configuration:

```bash
# Override any value from CLI
python prepare_data.py rendering.dpi=200

# Switch config groups
accelerate launch sft.py model=qwen25_vl_3b training_mode=inverse_graphics

# Multiple overrides
accelerate launch sft.py training_mode=combined model=qwen25_vl_7b training.lr=1e-5

# Show all options
python prepare_data.py --help
accelerate launch sft.py -- --help
```

### Config Groups

| Group    | Options                                           | Default      |
|----------|---------------------------------------------------|--------------|
| `model`  | `qwen25_3b`, `qwen25_7b`, `qwen25_vl_3b`, `qwen25_vl_7b` | `qwen25_3b` |
| `reward` | `dinov2`, `clip`, `dreamsim`                       | `dinov2`     |

### Training Mode Matrix

| `training_mode` | Compatible Models | Packing | Needs `render_dir` |
|-----------------|-------------------|---------|---------------------|
| `text_only`     | `qwen25_3b`, `qwen25_7b` | Yes | No |
| `inverse_graphics` | `qwen25_vl_3b`, `qwen25_vl_7b` | No | Yes |
| `combined`      | `qwen25_vl_3b`, `qwen25_vl_7b` | No | Yes |

### Environment Variables

| Variable      | Used By    | Description                 |
|---------------|------------|-----------------------------|
| `PYVEC_ROOT`  | prepare    | Path to dataset root        |
| `PYVEC_WORK`  | all        | Working directory for outputs |

### Adding a New Model

Create `configs/model/my_model.yaml`:

```yaml
# @package model
id: MyModel-13B
max_seq_len: 4096
is_vlm: false    # true for VLMs
```

Then: `accelerate launch sft.py model=my_model`

## SLURM Submission

### Full Text-Only Pipeline

```bash
JOB1=$(sbatch --parsable scripts/prepare_data.sbatch)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} scripts/train_sft.sbatch)
SFT_RUN_ID=<auto-name> sbatch --dependency=afterok:${JOB2} scripts/train_grpo.sbatch
```

### Full TikZero+ Pipeline

```bash
# Step 1: Prepare data
JOB1=$(sbatch --parsable scripts/prepare_data.sbatch)

# Step 2: Inverse Graphics SFT (Stage 1)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} scripts/train_sft_inverse.sbatch)

# Step 3: Combined SFT (Stage 2 — TikZero+ fine-tuning)
INVERSE_CHECKPOINT=<stage1_path> \
  JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} scripts/train_sft_combined.sbatch)

# Step 4: GRPO (optional, on the combined model)
SFT_RUN_ID=<combined_run> sbatch --dependency=afterok:${JOB3} scripts/train_grpo.sbatch
```

### Passing Extra Hydra Overrides to sbatch

```bash
sbatch scripts/train_sft.sbatch training.lr=1e-4 training.epochs=5
sbatch scripts/train_sft_inverse.sbatch training.epochs=10 lora.enabled=true
```

### GPU Variants

| Script                        | GPUs | Mode              | Effective Batch |
|-------------------------------|------|-------------------|-----------------|
| `train_sft.sbatch`            | 1    | text_only         | 8×4×1 = 32      |
| `train_sft_4gpu.sbatch`       | 4    | text_only         | 4×2×4 = 32      |
| `train_sft_inverse.sbatch`    | 1    | inverse_graphics  | 2×16×1 = 32     |
| `train_sft_combined.sbatch`   | 1    | combined          | 2×16×1 = 32     |
| `train_grpo.sbatch`           | 1    | —                 | 4×8×1 = 32      |
| `train_grpo_4gpu.sbatch`      | 4    | —                 | 2×4×4 = 32      |

## Design Choices

### Why TikZero+ for Python?

The TikZero+ paper shows that decoupling graphics program generation from text understanding yields better results than end-to-end training alone. Our adaptation:

- **Inverse graphics** (Stage 1): The VLM learns to "read" a figure and write the Python code that produced it. This is self-supervised — we already have rendered images from `prepare_data.py`.
- **Combined** (Stage 2): Adds caption understanding by fine-tuning with both image and caption inputs. This bridges the gap between visual understanding and text-guided generation.

### Caption Selection
Prioritizes `original_data.new_caption` over `caption` for higher-quality descriptions.

### DINOv2 as Default Reward
DINOv2 (`facebook/dinov2-large`) captures structural/spatial layout better than CLIP for scientific figure comparison.

### Multi-Signal Reward
```
reward = execution_bonus + (1 - execution_bonus) × visual_similarity
```

### Sequence Packing (text-only SFT only)
Packs multiple short samples into one sequence for ~30-40% better GPU utilization. Disabled for multimodal modes since images have fixed cost.

## Requirements

```
torch
transformers
trl
peft
datasets
accelerate
deepspeed
hydra-core
omegaconf
qwen-vl-utils
orjson
wandb
tensorboard
dreamsim
POT
```
