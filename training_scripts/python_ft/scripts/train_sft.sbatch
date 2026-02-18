#!/bin/bash
#SBATCH --job-name=pyvec_sft
#SBATCH --output=/home/vault/v123be/v123be26/pyvec/logs/sft_%j.out
#SBATCH --error=/home/vault/v123be/v123be26/pyvec/logs/sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ---------------------------------------------------------------------------
# SFT training: caption → Python/Matplotlib code
# ---------------------------------------------------------------------------

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"

SCRIPT_DIR="${PYVEC_WORK}"
ACCEL_CONFIG="${SCRIPT_DIR}/../accelerate_configs/deepspeed_sft.yaml"
cd "${SCRIPT_DIR}"

mkdir -p /home/vault/v123be/v123be26/pyvec/logs

# Adjust num_processes to match available GPUs
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "=== PYVEC SFT Training ==="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "GPUs:    ${NUM_GPUS}"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"
echo "Start:   $(date)"
echo "=========================="

# Hydra overrides are passed after the script name
accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    sft.py \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    model=qwen25_3b \
    training.epochs=3 \
    training.lr=5e-5 \
    training.batch_size=8 \
    training.gradient_accumulation=4 \
    training.packing=true \
    logging.report_to=tensorboard \
    "$@"

echo "=== Done: $(date) ==="
