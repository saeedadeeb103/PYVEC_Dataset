#!/bin/bash
#SBATCH --job-name=pyvec_detikz_vis
#SBATCH --output=/home/hpc/v123be/v123be26/pyvec/logs/eval_detikz_vis_%j.out
#SBATCH --error=/home/hpc/v123be/v123be26/pyvec/logs/eval_detikz_vis_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"
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
