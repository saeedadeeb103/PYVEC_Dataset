#!/bin/bash
#SBATCH --job-name=pyvec_grpo_dtk
#SBATCH --output=/home/hpc/v123be/v123be26/pyvec/logs/grpo_detikzify_%j.out
#SBATCH --error=/home/hpc/v123be/v123be26/pyvec/logs/grpo_detikzify_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ===========================================================================
#  PYVEC: GRPO post-training for DeTikZify-v2.5-8B SFT checkpoint
#
#  Goal: Improve execution rate and DINOv2 visual similarity rewards.
#  Uses PEFT LoRA during GRPO for memory efficiency (8B model on 1x A40).
#  The SFT LoRA adapter is first merged, then a fresh LoRA (r=32) is applied
#  for the GRPO phase — only adapter weights are trained.
# ===========================================================================

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load cuda/11.8.0 2>/dev/null || export CUDA_HOME=/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/cuda-11.8.0-qrwohrlyazyilia56tybqfm5k3k6kiom

export PYVEC_ROOT="/home/hpc/v123be/v123be26/PYVEC_Dataset"
export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"

SFT_RUN_ID="pyvec_sft_detikzify-v2.5-8b_inverse_lr2e-05_ep3_bs1x32_lora64"

cd "${PYVEC_WORK}"
mkdir -p /home/hpc/v123be/v123be26/pyvec/logs

echo "========================================================"
echo "  PYVEC: GRPO RL Training — DeTikZify 8B + LoRA — 1x A40"
echo "========================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-1}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK}"
echo "SFT model:  ${SFT_RUN_ID}"
echo "PEFT:       LoRA r=32, alpha=64"
echo "Start:      $(date)"
echo "========================================================"

python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No CUDA devices')"

# Pre-download DINOv2 to local models/ (compute nodes may have no internet)
if [[ ! -f models/dinov2-large/config.json ]]; then
  echo "Downloading DINOv2 to models/dinov2-large..."
  python download_reward_models.py --out models --models dinov2 || {
    echo "ERROR: Could not download DINOv2. Run on login node: cd ${PYVEC_WORK} && python download_reward_models.py --out models"
    exit 1
  }
fi

python grpo.py \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    model=detikzify_8b \
    model.sft_run_id="${SFT_RUN_ID}" \
    reward=dinov2_local \
    training.epochs=1 \
    training.lr=5e-6 \
    training.batch_size=1 \
    training.num_generations=4 \
    training.gradient_accumulation=32 \
    peft.enable=true \
    peft.r=32 \
    peft.lora_alpha=64 \
    peft.lora_dropout=0.05 \
    logging.report_to=tensorboard \
    "$@"

GRPO_EXIT=$?
echo ""
echo "========================================================"
echo "  DONE"
echo "========================================================"
echo "GRPO exit:  ${GRPO_EXIT}"
echo "End:        $(date)"
echo ""
echo ">>> Model saved to trained_models_grpo_python/"
echo "========================================================"
