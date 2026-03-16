#!/bin/bash
#SBATCH --job-name=pyvec_detikz_vis
#SBATCH --output=logs/eval_detikz_vis_%j.out
#SBATCH --error=logs/eval_detikz_vis_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate uni

export PYVEC_WORK="${PYVEC_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PYVEC_WORK}"

MODEL_PATH="trained_models_sft_python/pyvec_sft_detikzify-v2.5-8b_inverse_lr2e-05_ep3_bs1x32_lora64/final"
BASE_MODEL="nllg/detikzify-v2.5-8b"
OUT_DIR="eval_output_detikzify_vis"

echo "Start Eval DeTikZify Vis: $(date)"

python eval_detikzify_vis.py \
    --model_path "${MODEL_PATH}" \
    --base_model_path "${BASE_MODEL}" \
    --data_dir "${PYVEC_WORK}/data/python_ft" \
    --render_dir "${PYVEC_WORK}/data/rendered_images" \
    --out_dir "${OUT_DIR}" \
    --max_new_tokens 1024

echo "Done: $(date)"
