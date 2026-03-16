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

- **scripts/generate_synthetic_tikz_to_python.py**: Generate synthetic TikZ→Python data using LLMs (Gemini-3-pro, Claude, GPT, etc.) for use in PYVEC. See [Synthetic data generation](#synthetic-data-generation) below.

## Synthetic data generation

Synthetic (TikZ→Python) samples are **not** included in this repo by default; they are produced by the parent repository’s conversion pipeline using multimodal LLMs. Training scripts accept a `synthetic` source (e.g. `--source_filter arxiv_github_tex_synthetic_curated`).

**Option 1 – Script (recommended)**  
From the **repository root** (parent of `PYVEC_Dataset`):

```bash
# Gemini (e.g. Gemini-3-pro)
python PYVEC_Dataset/scripts/generate_synthetic_tikz_to_python.py \
  --provider gemini --model google/gemini-3-pro \
  --input cached_data/samples.jsonl --limit 500 \
  --pyvec-output PYVEC_Dataset/synthetic_dataset.jsonl

# OpenAI-compatible API (Claude, GPT)
python PYVEC_Dataset/scripts/generate_synthetic_tikz_to_python.py \
  --provider openai --model anthropic/claude-sonnet-4.5 \
  --input cached_data/samples.jsonl --output ./output_openai --limit 500 \
  --pyvec-output PYVEC_Dataset/synthetic_dataset.jsonl
```

Create `cached_data/samples.jsonl` first with:

```bash
python quick_download.py --num 1000
```

**Option 2 – Parent repo entry points**

- **Gemini**: `python main.py --input cached_data/samples.jsonl --model google/gemini-3-pro --limit 500`  
  Output: `./output/dataset.jsonl`. Add `source: "synthetic"` and `origin_type: "tikz_converted"` when merging into PYVEC.
- **OpenAI/Claude/GPT**: `python main_openai.py --input cached_data/samples.jsonl --output ./output_openai --model anthropic/claude-sonnet-4.5 --limit 500`  
  Output: `./output_openai/dataset.jsonl`. Same PYVEC fields as above.

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
