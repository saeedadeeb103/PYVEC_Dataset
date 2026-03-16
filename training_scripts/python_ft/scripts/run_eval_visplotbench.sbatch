#!/bin/bash
#SBATCH --job-name=pyvec_visplot
#SBATCH --output=logs/eval_visplotbench_%j.out
#SBATCH --error=logs/eval_visplotbench_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate uni

export PYVEC_WORK="${PYVEC_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PYVEC_WORK}"

MODEL_PATH="trained_models_sft_python/pyvec_sft_Qwen2.5-3B-Instruct_lr2e-05_ep3_bs2x16_lora128/final"
OUT_DIR="eval_output_visplotbench"

echo "Start Eval VisPlotBench: $(date)"

python eval_visplotbench.py \
    --model_path "${MODEL_PATH}" \
    --out_dir "${OUT_DIR}" \
    --max_new_tokens 2048 \
    --temperature 0.3

echo "Done: $(date)"
