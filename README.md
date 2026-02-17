# PYVEC Dataset & Training Scripts

This repository contains the dataset and training scripts for the PYVEC project (TikZ to Python/Matplotlib conversion).

## Contents

- **dataset.jsonl**: The combined dataset containing TikZ-derived and scraped Python/Matplotlib code samples.
  - Format: JSONL
  - Fields: `id`, `python_code`, `caption`, `source`, `origin_type`.
  - Size: ~15k samples.

- **training_scripts/**: Scripts for Fine-Tuning (SFT) and Reinforcement Learning (GRPO).
  - `sft_tikzilla.py`: Supervised Fine-Tuning.
  - `grpo_tikzilla.py`: GRPO / RL training.
  - `python_ft/`: Data preparation utilities.
  - `download_models.sh`: Helper to download base models.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r training_scripts/requirements.txt
   pip install 'detikzify @ git+https://github.com/potamides/DeTikZify'
   ```

2. **Download Models**:
   ```bash
   cd training_scripts
   chmod +x download_models.sh
   ./download_models.sh ../models
   ```

3. **Train**:
   See `training_scripts/README.md` for detailed training instructions.
