#!/bin/bash
#SBATCH --job-name=pyvec_sft_comb
#SBATCH --output=/home/vault/v123be/v123be26/pyvec/logs/sft_combined_%j.out
#SBATCH --error=/home/vault/v123be/v123be26/pyvec/logs/sft_combined_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ---------------------------------------------------------------------------
# Stage 2 — Combined SFT (TikZero+ style)
#   image + caption → Python/Matplotlib code
#   Fine-tunes the inverse graphics model with caption conditioning
#
# Set INVERSE_CHECKPOINT to the Stage 1 model path, or leave empty to
# start from the base VLM.
# ---------------------------------------------------------------------------

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"

SCRIPT_DIR="${PYVEC_WORK}"
ACCEL_CONFIG="${SCRIPT_DIR}/../accelerate_configs/deepspeed_sft.yaml"
cd "${SCRIPT_DIR}"

mkdir -p /home/vault/v123be/v123be26/pyvec/logs

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

# Optional: point to inverse_graphics checkpoint for two-stage TikZero+ pipeline
# e.g. INVERSE_CHECKPOINT="${PYVEC_WORK}/trained_models_sft_python/pyvec_sft_..._inverse_graphics_.../final"
INVERSE_CHECKPOINT="${INVERSE_CHECKPOINT:-}"

echo "=== PYVEC Combined SFT (Stage 2 — TikZero+ style) ==="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "GPUs:        ${NUM_GPUS}"
echo "Mode:        combined (image + caption → code)"
echo "Base model:  ${INVERSE_CHECKPOINT:-base VLM}"
echo "Start:       $(date)"
echo "======================================================"

# Build the resume arg if we have an inverse checkpoint
RESUME_ARG=""
if [ -n "${INVERSE_CHECKPOINT}" ]; then
    RESUME_ARG="resume_from=${INVERSE_CHECKPOINT}"
fi

# TikZero+ Sec. 5.2(iii): lower LR, fewer epochs for fine-tuning stage
accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    sft.py \
    training_mode=combined \
    model=qwen25_vl_3b \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    paths.render_dir="${PYVEC_WORK}/data/rendered_images" \
    training.epochs=3 \
    training.lr=1e-5 \
    training.batch_size=2 \
    training.gradient_accumulation=16 \
    logging.report_to=tensorboard \
    ${RESUME_ARG} \
    "$@"

echo "=== Done: $(date) ==="
