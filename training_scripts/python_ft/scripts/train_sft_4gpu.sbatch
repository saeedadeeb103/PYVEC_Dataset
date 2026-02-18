#!/bin/bash
#SBATCH --job-name=pyvec_sft_4gpu
#SBATCH --output=/home/vault/v123be/v123be26/pyvec/logs/sft_4gpu_%j.out
#SBATCH --error=/home/vault/v123be/v123be26/pyvec/logs/sft_4gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:4
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ---------------------------------------------------------------------------
# SFT training with 4x A40 GPUs (DeepSpeed ZeRO-3)
# ---------------------------------------------------------------------------

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"

SCRIPT_DIR="${PYVEC_WORK}"
ACCEL_CONFIG="${SCRIPT_DIR}/../accelerate_configs/deepspeed_sft.yaml"
cd "${SCRIPT_DIR}"

mkdir -p /home/vault/v123be/v123be26/pyvec/logs

NUM_GPUS=4

echo "=== PYVEC SFT Training (4 GPU) ==="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "GPUs:    ${NUM_GPUS}"
echo "Start:   $(date)"
echo "===================================="

# Reduce per-device batch & increase grad accum to keep effective batch same
accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    sft.py \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    model=qwen25_3b \
    training.epochs=3 \
    training.lr=5e-5 \
    training.batch_size=4 \
    training.gradient_accumulation=2 \
    training.packing=true \
    logging.report_to=tensorboard \
    "$@"

echo "=== Done: $(date) ==="
