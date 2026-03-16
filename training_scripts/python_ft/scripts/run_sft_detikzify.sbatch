#!/bin/bash
#SBATCH --job-name=pyvec_detikz
#SBATCH --output=logs/sft_detikzify_%j.out
#SBATCH --error=logs/sft_detikzify_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

# ===========================================================================
#  PYVEC: DeTikZify inverse graphics SFT — image → Python/Matplotlib code
#
#  Fine-tunes nllg/detikzify-v2.5-8b (8B VLM) with LoRA to produce
#  Python/Matplotlib code from rendered figure images.
#  Dataset already built at data/python_ft with rendered images.
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
echo "  PYVEC: DeTikZify Inverse Graphics SFT (LoRA) — 1x A40"
echo "========================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-1}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Start:     $(date)"
echo "========================================================"

# Diagnostics: verify CUDA is available in PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No CUDA devices')"

python sft_detikzify.py \
    model=detikzify_8b \
    paths.data_dir="${PYVEC_WORK}/data/python_ft" \
    paths.work_dir="${PYVEC_WORK}" \
    paths.render_dir="${PYVEC_WORK}/data/rendered_images" \
    training.epochs=3 \
    training.lr=2e-5 \
    training.batch_size=1 \
    training.gradient_accumulation=32 \
    training.warmup_ratio=0.05 \
    training.scheduler=cosine \
    lora.enabled=true \
    lora.rank=64 \
    lora.alpha=128 \
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
echo "========================================================"
