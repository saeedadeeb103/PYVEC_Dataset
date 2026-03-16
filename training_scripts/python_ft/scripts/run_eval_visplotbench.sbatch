#!/bin/bash
#SBATCH --job-name=pyvec_visplot
#SBATCH --output=/home/hpc/v123be/v123be26/pyvec/logs/eval_visplotbench_%j.out
#SBATCH --error=/home/hpc/v123be/v123be26/pyvec/logs/eval_visplotbench_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"
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
