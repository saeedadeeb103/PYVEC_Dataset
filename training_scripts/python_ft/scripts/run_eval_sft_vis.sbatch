#!/bin/bash
#SBATCH --job-name=pyvec_eval_vis
#SBATCH --output=/home/hpc/v123be/v123be26/pyvec/logs/eval_sft_vis_%j.out
#SBATCH --error=/home/hpc/v123be/v123be26/pyvec/logs/eval_sft_vis_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"
cd "${PYVEC_WORK}"

MODEL_PATH="trained_models_sft_python/pyvec_sft_Qwen2.5-3B-Instruct_lr2e-05_ep3_bs2x16_lora128/final"
OUT_DIR="eval_output_vis"

echo "Start Eval SFT Vis: $(date)"

python eval_sft_vis.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${PYVEC_WORK}/data/python_ft" \
    --render_dir "${PYVEC_WORK}/data/rendered_images" \
    --out_dir "${OUT_DIR}" \
    --max_new_tokens 2048 \
    --temperature 0.3

echo "Done: $(date)"
