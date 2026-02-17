#!/bin/bash
# Download models for SFT and GRPO training

WORK_DIR=${1:-"./models"}
mkdir -p "$WORK_DIR"

echo "Downloading models to $WORK_DIR..."

# 1. Text-only Model (Qwen2.5-3B)
echo "Downloading Qwen/Qwen2.5-3B..."
huggingface-cli download Qwen/Qwen2.5-3B --local-dir "$WORK_DIR/Qwen2.5-3B" --exclude "*.safetensors"  # Optional exclude if you want bin or vice versa

# 2. Multimodal Model (DeTikZify v2)
echo "Downloading nllg/detikzify-v2-8b..."
huggingface-cli download nllg/detikzify-v2-8b --local-dir "$WORK_DIR/detikzify-v2-8b"

# 3. TikZero Adapter (Optional)
echo "Downloading nllg/tikzero-adapter..."
huggingface-cli download nllg/tikzero-adapter --local-dir "$WORK_DIR/tikzero-adapter"

# 4. Reward Model (CLIP)
echo "Downloading openai/clip-vit-large-patch14..."
huggingface-cli download openai/clip-vit-large-patch14 --local-dir "$WORK_DIR/clip-vit-large-patch14"

echo "Done! Models downloaded to $WORK_DIR"
