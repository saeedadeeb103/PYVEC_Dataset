#!/bin/bash
#SBATCH --job-name=pyvec_sft_inv
#SBATCH --output=logs/sft_inverse_%j.out
#SBATCH --error=logs/sft_inverse_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ---------------------------------------------------------------------------
# Stage 1 — Inverse Graphics SFT (DeTikZify-style)
#   image → Python/Matplotlib code (self-supervised, no captions needed)
#   Uses Qwen2.5-VL-3B as the VLM backbone
# ---------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate uni

export PYVEC_WORK="${PYVEC_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

SCRIPT_DIR="${PYVEC_WORK}"
ACCEL_CONFIG="${SCRIPT_DIR}/../accelerate_configs/deepspeed_sft.yaml"
cd "${SCRIPT_DIR}"

mkdir -p "${PYVEC_WORK}/logs"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "=== PYVEC Inverse Graphics SFT (Stage 1) ==="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "GPUs:    ${NUM_GPUS}"
echo "Mode:    inverse_graphics (image → code)"
echo "Start:   $(date)"
echo "=============================================="

# VLM uses more memory per sample → smaller batch, more grad accum
accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    sft.py \
    training_mode=inverse_graphics \
    model=qwen25_vl_3b \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    paths.render_dir="${PYVEC_WORK}/data/rendered_images" \
    training.epochs=5 \
    training.lr=5e-5 \
    training.batch_size=2 \
    training.gradient_accumulation=16 \
    logging.report_to=tensorboard \
    "$@"

echo "=== Done: $(date) ==="
