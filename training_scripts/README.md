# Training Scripts

This directory contains scripts for fine-tuning and reinforcement learning of LLMs for TikZ-to-Python conversion.

## Structure

- `python_ft/`: Helper scripts for data preparation.
- `sft_tikzilla.py`: Script for Supervised Fine-Tuning (SFT).
- `grpo_tikzilla.py`: Script for Group Relative Policy Optimization (GRPO) / Reinforcement Learning.
- `batching.py`: Utilities for batching and preprocessing.
- `Datasets/`: Dataset loading utilities.
- `accelerate_configs/`: Configuration files for HuggingFace Accelerate.
- `download_models.sh`: Script to download necessary models.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install 'detikzify @ git+https://github.com/potamides/DeTikZify'
   ```

2. Download models:
   ```bash
   chmod +x download_models.sh
   ./download_models.sh ./models
   ```

## Usage

### 1. Data Preparation

Convert your `dataset.jsonl` to the format required for training:

```bash
python python_ft/prepare_data.py \
  --input ../../output/dataset.jsonl \
  --output data/python_ft \
  --render
```

### 2. SFT Training

Run SFT training using `accelerate`:

```bash
accelerate launch --config_file accelerate_configs/deepspeed_sft.yaml sft_tikzilla.py \
  --work_dir ./work_dir \
  --model_id Qwen2.5-3B \
  --epochs 5
```

### 3. GRPO Training (RL)

Run GRPO training after SFT:

```bash
accelerate launch --config_file accelerate_configs/deepspeed_grpo.yaml grpo_tikzilla.py \
  --work_dir ./work_dir \
  --tmp_dir ./tmp \
  --model_id Qwen2.5-3B \
  --checkpoint_id <your_sft_checkpoint>
```
