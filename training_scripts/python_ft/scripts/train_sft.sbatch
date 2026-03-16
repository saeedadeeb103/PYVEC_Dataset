#!/bin/bash
#SBATCH --job-name=pyvec_sft
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ---------------------------------------------------------------------------
# SFT training: caption → Python/Matplotlib code
# ---------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate uni

export PYVEC_WORK="${PYVEC_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

SCRIPT_DIR="${PYVEC_WORK}"
ACCEL_CONFIG="${SCRIPT_DIR}/../accelerate_configs/deepspeed_sft.yaml"
cd "${SCRIPT_DIR}"

mkdir -p "${PYVEC_WORK}/logs"

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
