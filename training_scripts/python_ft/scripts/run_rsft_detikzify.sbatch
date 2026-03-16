#!/bin/bash
#SBATCH --job-name=rsft_train
#SBATCH --output=logs/rsft_train_%j.out
#SBATCH --error=logs/rsft_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate uni

export PYVEC_WORK="${PYVEC_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PYVEC_WORK}"

echo "Start RSFT Training: $(date)"

# Use accelerate to launch across 2 GPUs
accelerate launch --multi_gpu --num_processes 2 sft_detikzify.py \
    model.id="detikzify-v2.5-8b" \
    paths.data_dir="${PYVEC_WORK}/data/python_ft_rsft_filtered" \
    lora.enabled=true \
    lora.rank=64 \
    training.epochs=3 \
    training.lr=1e-5 \
    training.batch_size=1 \
    training.gradient_accumulation=16

echo "Done RSFT Training: $(date)"
