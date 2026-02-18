#!/bin/bash
#SBATCH --job-name=pyvec_prepare
#SBATCH --output=/home/hpc/v123be/v123be26/pyvec/logs/prepare_%j.out
#SBATCH --error=/home/hpc/v123be/v123be26/pyvec/logs/prepare_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ---------------------------------------------------------------------------
# Data preparation — CPU only, no GPU needed
# Renders Python/Matplotlib code to images and builds HF Arrow dataset
# ---------------------------------------------------------------------------

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

# Project paths
export PYVEC_ROOT="/home/hpc/v123be/v123be26/PYVEC_Dataset"
export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"

SCRIPT_DIR="${PYVEC_WORK}"
cd "${SCRIPT_DIR}"

mkdir -p /home/vault/v123be/v123be26/pyvec/logs

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
