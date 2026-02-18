#!/bin/bash
#SBATCH --job-name=pyvec_grpo
#SBATCH --output=/home/hpc/v123be/v123be26/pyvec/logs/grpo_%j.out
#SBATCH --error=/home/hpc/v123be/v123be26/pyvec/logs/grpo_%j.err
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ===========================================================================
#  PYVEC: GRPO post-training with Python execution rewards
#
#  Uses merged LoRA checkpoint from text-only SFT + DINOv2 reward on 1x A40.
# ===========================================================================

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load cuda/11.8.0 2>/dev/null || export CUDA_HOME=/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/cuda-11.8.0-qrwohrlyazyilia56tybqfm5k3k6kiom

export PYVEC_ROOT="/home/hpc/v123be/v123be26/PYVEC_Dataset"
export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"

SFT_RUN_ID="${SFT_RUN_ID:-pyvec_sft_Qwen2.5-3B-Instruct_lr2e-05_ep3_bs2x16_lora128}"

cd "${PYVEC_WORK}"

mkdir -p /home/hpc/v123be/v123be26/pyvec/logs

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

python grpo.py \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    model=qwen25_3b \
    model.sft_run_id="${SFT_RUN_ID}" \
    reward=dinov2 \
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
