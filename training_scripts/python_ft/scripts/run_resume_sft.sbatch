#!/bin/bash
#SBATCH --job-name=pyvec_resume_sft
#SBATCH --output=/home/hpc/v123be/v123be26/pyvec/logs/resume_sft_combined_%j.out
#SBATCH --error=/home/hpc/v123be/v123be26/pyvec/logs/resume_sft_combined_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ===========================================================================
#  PYVEC Pipeline: RESUME Data Preparation -> Combined SFT (TikZero+ style)
#
#  Step 1: Build HF dataset from EXISTING rendered images (rendering.skip=true)
#  Step 2: Combined SFT (image + caption -> code) with LoRA on 1x A40
#
#  After this finishes, run GRPO separately using the SFT run ID printed
#  at the end of this job.
# ===========================================================================

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

export PYVEC_ROOT="/home/hpc/v123be/v123be26/PYVEC_Dataset"
export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"

SCRIPT_DIR="${PYVEC_WORK}"
cd "${SCRIPT_DIR}"

mkdir -p /home/hpc/v123be/v123be26/pyvec/logs

echo "========================================================"
echo "  PYVEC: RESUME Data Prep + Combined SFT (LoRA) on 1x A40"
echo "========================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-1}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Start:     $(date)"
echo "========================================================"

# -------------------------------------------------------------------
# Step 1: Prepare data (SKIP RENDERING -> build HF Arrow dataset)
# -------------------------------------------------------------------
echo ""
echo "========== STEP 1: Data Preparation (Resuming) =========="
echo "Start: $(date)"

python prepare_data.py \
    paths.input="${PYVEC_ROOT}/dataset.jsonl" \
    paths.output="${PYVEC_WORK}/data/python_ft" \
    paths.render_dir="${PYVEC_WORK}/data/rendered_images" \
    rendering.dpi=150 \
    rendering.workers=${SLURM_CPUS_PER_TASK} \
    rendering.skip=true

PREP_EXIT=$?
if [ ${PREP_EXIT} -ne 0 ]; then
    echo "ERROR: Data preparation failed with exit code ${PREP_EXIT}"
    exit ${PREP_EXIT}
fi
echo "Step 1 done: $(date)"

# -------------------------------------------------------------------
# Step 2: Combined SFT with LoRA (image + caption -> code)
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
echo "Next step: run GRPO with the SFT model."
echo "Check trained_models_sft_python/ for the run directory."
echo "========================================================"
