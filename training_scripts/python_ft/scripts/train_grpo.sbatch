#!/bin/bash
#SBATCH --job-name=pyvec_grpo
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ===========================================================================
#  PYVEC: GRPO post-training with Python execution rewards
#
#  Uses merged LoRA checkpoint from text-only SFT + DINOv2 reward on 1x A40.
# ===========================================================================

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate uni

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load cuda 2>/dev/null || true

export PYVEC_WORK="${PYVEC_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYVEC_ROOT="${PYVEC_ROOT:-$(dirname "$(dirname "$PYVEC_WORK")")}"

SFT_RUN_ID="${SFT_RUN_ID:-pyvec_sft_Qwen2.5-3B-Instruct_lr2e-05_ep3_bs2x16_lora128}"

cd "${PYVEC_WORK}"

mkdir -p "${PYVEC_WORK}/logs"

echo "========================================================"
echo "  PYVEC: GRPO RL Training — DINOv2 Reward — 1x A40"
echo "========================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-1}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK}"
echo "SFT model:  ${SFT_RUN_ID}"
echo "Start:      $(date)"
echo "========================================================"

python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No CUDA devices')"

# Pre-download DINOv2 to local models/ (compute nodes may have no internet)
# If this fails, run on login node: python download_reward_models.py --out models
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
    model=qwen25_3b \
    model.sft_run_id="${SFT_RUN_ID}" \
    reward=dinov2_local \
    training.epochs=1 \
    training.lr=5e-6 \
    training.batch_size=2 \
    training.num_generations=4 \
    training.gradient_accumulation=16 \
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
