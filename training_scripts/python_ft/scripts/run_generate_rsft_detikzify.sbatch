#!/bin/bash
#SBATCH --job-name=rsft_gen
#SBATCH --output=logs/rsft_gen_%j.out
#SBATCH --error=logs/rsft_gen_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate uni

export PYVEC_WORK="${PYVEC_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PYVEC_WORK}"

# We use the previous SFT run with the LoRA adapter
MODEL_PATH="${PYVEC_WORK}/trained_models_sft_python/pyvec_sft_detikzify-v2.5-8b_inverse_lr2e-05_ep3_bs1x32_lora64/final"
BASE_MODEL="nllg/detikzify-v2.5-8b" # Used if the model path is just the LoRA adapter

echo "Start RSFT Generation: $(date)"

python generate_rejection_samples.py \
    --model_path "${MODEL_PATH}" \
    --base_model_path "${BASE_MODEL}" \
    --render_dir "${PYVEC_WORK}/data/rendered_images" \
    --output_dataset_path "${PYVEC_WORK}/data/python_ft_rsft_filtered" \
    --num_candidates 8 \
    --temperature 1.0

echo "Done RSFT Generation: $(date)"
