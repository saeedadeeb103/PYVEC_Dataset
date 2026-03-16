#!/bin/bash
#SBATCH --job-name=pyvec_grpo_4gpu
#SBATCH --output=/home/vault/v123be/v123be26/pyvec/logs/grpo_4gpu_%j.out
#SBATCH --error=/home/vault/v123be/v123be26/pyvec/logs/grpo_4gpu_%j.err
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:a40:4
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16
# ---------------------------------------------------------------------------
# GRPO post-training with 4x A40 GPUs (DeepSpeed ZeRO-2)
# ---------------------------------------------------------------------------

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"

SFT_RUN_ID="${SFT_RUN_ID:-pyvec_sft_Qwen2.5-3B-Instruct_lr2e-05_ep3_bs2x16_lora128}"

SCRIPT_DIR="${PYVEC_WORK}"
ACCEL_CONFIG="${SCRIPT_DIR}/../accelerate_configs/deepspeed_grpo.yaml"
cd "${SCRIPT_DIR}"

mkdir -p /home/vault/v123be/v123be26/pyvec/logs

NUM_GPUS=4

echo "=== PYVEC GRPO Training (4 GPU) ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       ${NUM_GPUS}"
echo "SFT model:  ${SFT_RUN_ID}"
echo "Start:      $(date)"
echo "====================================="

accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    grpo.py \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    model=qwen25_3b \
    model.sft_run_id="${SFT_RUN_ID}" \
    reward=dinov2 \
    training.epochs=1 \
    training.lr=5e-6 \
    training.batch_size=2 \
    training.gradient_accumulation=4 \
    logging.report_to=tensorboard \
    "$@"

echo "=== Done: $(date) ==="
