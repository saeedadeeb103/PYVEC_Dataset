# PYVEC Dataset and Training Pipeline

PYVEC is a repository for building and evaluating models that generate Python/Matplotlib visualizations. It combines a curated training dataset, raw scraping outputs, rule-based data augmentation, training code for SFT and GRPO, a lightweight viewer for qualitative inspection, and human-evaluation artifacts.



## What Is In This Repository?

- `dataset.jsonl`: Main curated JSONL training dataset used by the project. It contains `15,272` examples.
- `Data_augmentation/final_dataset_deduplicated.jsonl`: Larger augmented dataset with `53,846` examples. This file is tracked with Git LFS.
- `Scraping/`: Raw and validated visualization snippets collected from Kaggle, StackOverflow, GitHub, and official galleries.
- `training_scripts/`: Training, preparation, and evaluation code for SFT and GRPO experiments.
- `Viewer/`: Small web app for browsing samples and rendering their code outputs.
- `Human_evaluation/`: Manual evaluation spreadsheets for comparing model variants.
- `EXTRACTION_PIPELINE.md`: Detailed description of the data-collection pipeline.
- `MINDMAP.txt`: Project planning and scope notes.

## Repository Map

```text
.
|-- README.md
|-- dataset.jsonl
|-- Data_augmentation/
|   |-- final_dataset_deduplicated.jsonl
|   |-- parameter_perturbation.py
|   `-- rendered_pairs/
|-- Scraping/
|   |-- all_validated_snippets.json
|   |-- cleaned_snippets_scraped.jsonl
|   |-- Galleries/
|   |-- GitHub/
|   |-- Kaggle/
|   `-- StackOverflow/
|-- training_scripts/
|   |-- README.md
|   |-- sft_tikzilla.py
|   |-- grpo_tikzilla.py
|   `-- python_ft/
|-- Viewer/
|   |-- backend/
|   `-- frontend/
|-- Human_evaluation/
|-- EXTRACTION_PIPELINE.md
`-- MINDMAP.txt
```


## Dataset Overview

### 1. Main training dataset

- File: `dataset.jsonl`
- Format: JSONL
- Size: `15,272` records
- Typical fields:
  - `id`
  - `python_code`
  - `caption`
  - `source`
  - `origin_type`
  - `original_data` (optional nested metadata from the original source)

Representative schema:

```json
{
  "id": "ff56d3a1ca82",
  "python_code": "import matplotlib.pyplot as plt\n...",
  "caption": "A description of the target figure.",
  "source": "arxiv",
  "origin_type": "tikz_conversion",
  "original_data": {
    "code": "\\begin{tikzpicture} ...",
    "new_caption": "A richer rewritten caption.",
    "conversion_status": "success"
  }
}
```

### 2. Augmented dataset

- File: `Data_augmentation/final_dataset_deduplicated.jsonl`
- Format: JSONL
- Size: `53,846` records
- Purpose: expanded training data after rule-based perturbation and deduplication
- Note: this file is stored with Git LFS because of its size

### 3. Scraped visualization corpus

- File: `Scraping/all_validated_snippets.json`
- Format: JSON array
- Size: `3,802` validated visualization snippets
- Sources: Kaggle, StackOverflow, GitHub, and official galleries
- Typical fields:
  - `code`
  - `caption`
  - `library`
  - `source_type`
  - `source`
  - `file_id`

### 4. Human evaluation artifacts

- Directory: `Human_evaluation/`
- Files:
  - `human_eval_samples_sft.ods`
  - `human_eval_samples_grpo_1701.ods`
  - `human_eval_samples_data_aug.ods`

These spreadsheets provide qualitative comparison material for the main model variants.

## Data Pipeline

At a high level, the repository supports the following workflow:

1. Collect runnable plotting code from public sources using the extractors in `Scraping/`.
2. Curate a main JSONL training set in `dataset.jsonl`.
3. Expand the training set with rule-based perturbations using `Data_augmentation/parameter_perturbation.py`.
4. Prepare rendered data for training and evaluation with the scripts in `training_scripts/python_ft/`.
5. Train and evaluate SFT and GRPO models.

For a deeper explanation of the extraction stage, see `EXTRACTION_PIPELINE.md`.

## Quick Start

### 1. Clone and pull large files

```bash
git lfs install
git lfs pull
```

### 2. Run the dataset viewer

```bash
pip install -r Viewer/backend/requirements.txt
python Viewer/backend/app.py
```

Then open `http://127.0.0.1:5001`.

Notes:

- The viewer now auto-detects `dataset.jsonl`, `combined_dataset.jsonl`, `Scraping/all_validated_snippets.json`, or `Data_augmentation/final_dataset_deduplicated.jsonl`.
- You can force a specific dataset with the `PYVEC_DATA_FILE` environment variable.

### 3. Generate augmented data

```bash
python Data_augmentation/parameter_perturbation.py \
  --input dataset.jsonl \
  --output Data_augmentation/final_dataset_augmented.jsonl \
  --variations-per-sample 4
```

The augmentation script performs rule-based perturbations of coordinates, colors, opacity, and size-related parameters. It does not require an LLM.

### 4. Train models

Install training dependencies:

```bash
pip install -r training_scripts/requirements.txt
```

Then use the training pipeline in `training_scripts/python_ft/`. Common entry points are:

```bash
python training_scripts/python_ft/prepare_data.py
accelerate launch training_scripts/python_ft/sft.py
accelerate launch training_scripts/python_ft/grpo.py model.sft_run_id=<RUN_ID>
```

More detailed commands are documented in:

- `training_scripts/README.md`
- `training_scripts/python_ft/README.md`

## Key Code Locations

- `Scraping/Kaggle/kaggle_smart_extractor.py`: Kaggle notebook extraction
- `Scraping/StackOverflow/stackoverflow_extractor.py`: StackOverflow extraction
- `Scraping/GitHub/github_smart_extractor.py`: GitHub extraction
- `Scraping/Galleries/gallery_extractor.py`: official gallery extraction
- `Data_augmentation/parameter_perturbation.py`: rule-based augmentation
- `training_scripts/python_ft/prepare_data.py`: dataset rendering and preparation
- `training_scripts/python_ft/sft.py`: supervised fine-tuning
- `training_scripts/python_ft/grpo.py`: reinforcement learning / GRPO
- `training_scripts/python_ft/eval_*.py`: evaluation scripts
- `Viewer/backend/app.py`: viewer backend and rendering API

## Recommended Reading Order

1. `README.md`
2. `EXTRACTION_PIPELINE.md`
3. `training_scripts/python_ft/README.md`
4. `training_scripts/README.md`

## Reproducibility Notes

- The augmented dataset uses Git LFS, so a plain `git clone` is not enough if you want all data locally.
- The viewer supports both JSONL datasets and JSON array snippet files.
- The augmentation script now accepts CLI arguments, so it can be run directly without editing the file.
- `Data_augmentation/rendered_pairs/` contains visual examples of original and augmented samples for quick inspection.


