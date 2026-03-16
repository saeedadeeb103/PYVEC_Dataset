#!/bin/bash
#SBATCH --job-name=pyvec_prepare
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ---------------------------------------------------------------------------
# Data preparation — CPU only, no GPU needed
# Renders Python/Matplotlib code to images and builds HF Arrow dataset
# ---------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate uni

# Project paths
export PYVEC_WORK="${PYVEC_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYVEC_ROOT="${PYVEC_ROOT:-$(dirname "$(dirname "$PYVEC_WORK")")}"

SCRIPT_DIR="${PYVEC_WORK}"
cd "${SCRIPT_DIR}"

mkdir -p "${PYVEC_WORK}/logs"

echo "=== PYVEC Data Preparation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "Start:  $(date)"
echo "================================"

python prepare_data.py \
    paths.input="${PYVEC_ROOT}/dataset.jsonl" \
    paths.output="${PYVEC_WORK}/data/python_ft" \
    paths.render_dir="${PYVEC_WORK}/data/rendered_images" \
    rendering.dpi=150 \
    rendering.workers=${SLURM_CPUS_PER_TASK}

echo "=== Done: $(date) ==="
