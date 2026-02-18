#!/bin/bash
#SBATCH --job-name=pyvec_sft_txt
#SBATCH --output=/home/hpc/v123be/v123be26/pyvec/logs/sft_textonly_%j.out
#SBATCH --error=/home/hpc/v123be/v123be26/pyvec/logs/sft_textonly_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ===========================================================================
#  PYVEC: Text-Only SFT — caption → Python/Matplotlib code
#
#  Uses Qwen2.5-3B-Instruct with LoRA on 1x A40.
#  Dataset already built at data/python_ft (14235 train / 1005 val).
# ===========================================================================

source /home/hpc/v123be/v123be26/miniconda3/etc/profile.d/conda.sh
conda activate uni

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load cuda/11.8.0 2>/dev/null || export CUDA_HOME=/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/cuda-11.8.0-qrwohrlyazyilia56tybqfm5k3k6kiom

export PYVEC_ROOT="/home/hpc/v123be/v123be26/PYVEC_Dataset"
export PYVEC_WORK="/home/hpc/v123be/v123be26/PYVEC_Dataset/training_scripts/python_ft"

cd "${PYVEC_WORK}"

mkdir -p /home/hpc/v123be/v123be26/pyvec/logs

echo "========================================================"
echo "  PYVEC: Text-Only SFT (LoRA) — 1x A40"
echo "========================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-1}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Start:     $(date)"
echo "========================================================"

# Diagnostics: verify CUDA is available in PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No CUDA devices')"

python sft.py \
    training_mode=text_only \
    model=qwen25_3b \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    training.epochs=3 \
    training.lr=2e-5 \
    training.batch_size=2 \
    training.gradient_accumulation=16 \
    training.warmup_ratio=0.05 \
    training.scheduler=cosine \
    training.packing=true \
    model.max_seq_len=4096 \
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
echo "SFT exit:   ${SFT_EXIT}"
echo "End:        $(date)"
echo ""
echo ">>> Model saved to trained_models_sft_python/"
echo ">>> For GRPO, use the run ID from that directory"
echo "========================================================"
