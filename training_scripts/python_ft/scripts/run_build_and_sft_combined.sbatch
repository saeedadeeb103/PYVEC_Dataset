#!/bin/bash
#SBATCH --job-name=pyvec_sft_comb
#SBATCH --output=logs/sft_combined_%j.out
#SBATCH --error=logs/sft_combined_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ===========================================================================
#  PYVEC: Build HF dataset from existing PNGs → Combined SFT with LoRA
#
#  Images already rendered (15193 PNGs on disk). This script:
#    Step 1: Build HF Arrow dataset from cached PNGs (skip rendering)
#    Step 2: Combined SFT (image + caption → code) with LoRA on 1x A40
# ===========================================================================

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate uni

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load cuda 2>/dev/null || true

export PYVEC_WORK="${PYVEC_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYVEC_ROOT="${PYVEC_ROOT:-$(dirname "$(dirname "$PYVEC_WORK")")}"

cd "${PYVEC_WORK}"

mkdir -p "${PYVEC_WORK}/logs"

echo "========================================================"
echo "  PYVEC: Build Dataset + Combined SFT (LoRA) — 1x A40"
echo "========================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-1}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Start:     $(date)"
echo "========================================================"

# -------------------------------------------------------------------
# Step 1: Build HF dataset from existing rendered PNGs (skip rendering)
# -------------------------------------------------------------------
echo ""
echo "========== STEP 1: Build HF Dataset (skip rendering) =========="
echo "Start: $(date)"

python prepare_data.py \
    paths.input="${PYVEC_ROOT}/dataset.jsonl" \
    paths.output="${PYVEC_WORK}/data/python_ft" \
    paths.render_dir="${PYVEC_WORK}/data/rendered_images" \
    rendering.skip=true

PREP_EXIT=$?
if [ ${PREP_EXIT} -ne 0 ]; then
    echo "ERROR: Dataset build failed with exit code ${PREP_EXIT}"
    exit ${PREP_EXIT}
fi
echo "Step 1 done: $(date)"

# -------------------------------------------------------------------
# Step 2: Combined SFT with LoRA (image + caption → code)
# -------------------------------------------------------------------
echo ""
echo "========== STEP 2: Combined SFT with LoRA =========="
echo "Start: $(date)"

python sft.py \
    training_mode=combined \
    model=qwen25_vl_3b \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    paths.render_dir="${PYVEC_WORK}/data/rendered_images" \
    training.epochs=3 \
    training.lr=5e-5 \
    training.batch_size=2 \
    training.gradient_accumulation=16 \
    training.warmup_ratio=0.05 \
    training.scheduler=cosine \
    lora.enabled=true \
    lora.rank=128 \
    lora.alpha=256 \
    lora.dropout=0.05 \
    logging.report_to=tensorboard

SFT_EXIT=$?
echo ""
echo "========================================================"
echo "  DONE"
echo "========================================================"
echo "Prep exit:  ${PREP_EXIT}"
echo "SFT exit:   ${SFT_EXIT}"
echo "End:        $(date)"
echo ""
echo ">>> For GRPO, use the run ID from trained_models_sft_python/"
echo "========================================================"
