# PythonFT — Fine-Tuning Pipeline for TikZ → Python/Matplotlib

Fine-tune inverse graphics models to generate Python/Matplotlib code from scientific figure descriptions and images, adapted from the TikZero/DeTikZify/TikZilla architecture.

## Architecture

```
Caption ──→ [TikZero Adapter (frozen)] ──→ Image Embeddings ──┐
                                                               ├──→ [LLM Decoder (trainable)] ──→ Python/Matplotlib Code
Image   ──→ [SigLIP Vision Encoder (frozen)] ────────────────┘
```

## Prerequisites

```bash
pip install torch transformers trl peft datasets orjson accelerate dreamsim POT
pip install 'detikzify @ git+https://github.com/potamides/DeTikZify'
```

Download models to `$WORK_DIR/models/`:
```bash
# Text-only
huggingface-cli download Qwen/Qwen2.5-3B --local-dir $WORK_DIR/models/Qwen2.5-3B

# Multimodal
huggingface-cli download nllg/detikzify-v2-8b --local-dir $WORK_DIR/models/detikzify-v2-8b

# TikZero adapter (optional, for text-conditioned inference)
huggingface-cli download nllg/tikzero-adapter --local-dir $WORK_DIR/models/tikzero-adapter
```

## Pipeline

### Step 1 — Prepare Data

Convert the conversion pipeline output (`output/dataset.jsonl`) to HuggingFace Arrow format:

```bash
python prepare_data.py \
  --input ../../output/dataset.jsonl \
  --images-dir ../../output \
  --output data/python_ft \
  --val-ratio 0.066 \
  --min-code-len 50 \
  --max-code-len 8000
```

With rendered Python images (needed for GRPO):
```bash
python prepare_data.py \
  --input ../../output/dataset.jsonl \
  --images-dir ../../output \
  --output data/python_ft \
  --render \
  --render-dir data/rendered_images
```

### Step 2 — SFT Training

#### Option A: Text-Only (Qwen2.5-3B)

```bash
accelerate launch --config_file ../accelerate_configs/deepspeed_sft.yaml sft.py \
  --mode text \
  --model-id Qwen2.5-3B \
  --data-dir data/python_ft \
  --max-seq-len 2048 \
  --epochs 5 \
  --batch-size 16 \
  --gradient-accumulation 2 \
  --lr 1e-4 \
  --scheduler cosine \
  --work-dir $WORK_DIR
```

With LoRA:
```bash
accelerate launch --config_file ../accelerate_configs/deepspeed_sft.yaml sft.py \
  --mode text \
  --model-id Qwen2.5-3B \
  --data-dir data/python_ft \
  --use-lora \
  --lora-rank 256 \
  --lora-alpha 512 \
  --work-dir $WORK_DIR
```

#### Option B: Multimodal (DeTikZifyv2)

```bash
accelerate launch --config_file ../accelerate_configs/deepspeed_sft.yaml sft.py \
  --mode multimodal \
  --detikzify-model detikzify-v2-8b \
  --data-dir data/python_ft \
  --max-seq-len 2048 \
  --epochs 5 \
  --batch-size 16 \
  --lr 5e-5 \
  --work-dir $WORK_DIR
```

### Step 3 — GRPO Post-Training

After SFT, apply reinforcement learning with Python execution rewards:

```bash
accelerate launch --config_file ../accelerate_configs/deepspeed_grpo.yaml grpo.py \
  --model-id pythonft_text_Qwen2.5-3B_lr0.0001_ep5 \
  --checkpoint checkpoint-XXXX \
  --data-dir data/python_ft \
  --reward-backend clip \
  --reward-model openai/clip-vit-large-patch14 \
  --batch-size 4 \
  --gradient-accumulation 9 \
  --num-generations 8 \
  --lr 5e-6 \
  --scheduler constant \
  --work-dir $WORK_DIR
```

Reward backends:
| Backend | Model | Speed | Quality |
|---------|-------|-------|---------|
| `clip` | `openai/clip-vit-large-patch14` | Fast | Good |
| `dreamsim` | `ensemble` | Medium | Best |
| `detikzify` | `nllg/detikzify-v2-8b` | Slow | Good (EMD-based) |

### Step 4 — Inference (After Training)

#### Text-only model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/checkpoint", torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("path/to/checkpoint")

messages = [
    {"role": "system", "content": "You are an expert at generating Python Matplotlib code."},
    {"role": "user", "content": "Generate Python Matplotlib code for: A bar chart showing GDP of top 5 countries"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, top_p=0.95)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

#### With TikZero adapter (text → image embeddings → Python):
```python
from detikzify.model import load, load_adapter
from detikzify.infer import DetikzifyPipeline

pipeline = DetikzifyPipeline(
    *load_adapter(
        *load(model_name_or_path="path/to/finetuned-detikzify", device_map="auto", torch_dtype="bfloat16"),
        adapter_name_or_path="nllg/tikzero-adapter",
    )
)
fig = pipeline.sample(text="A neural network architecture diagram")
```

## File Structure

```
PythonFT/
├── prepare_data.py    # JSONL → HuggingFace Dataset
├── preprocessing.py   # Tokenization and batching
├── rewards.py         # Python execution + image similarity rewards
├── sft.py             # SFT training (text-only + multimodal)
├── grpo.py            # GRPO post-training
└── README.md          # This file
```

## Reward Mechanism

The GRPO reward replaces TikZ LaTeX compilation with Python execution:

```
TikZ pipeline:   generate TikZ → pdflatex → PDF → pdftoppm → PNG → compare
Python pipeline:  generate Python → exec() → plt.savefig() → PNG → compare
```

Python execution is faster and requires no external dependencies (no TeX Live, ghostscript, or poppler).

## Key Differences from TikZ Pipeline

| Aspect | TikZ (original) | Python (this pipeline) |
|--------|-----------------|----------------------|
| Output format | LaTeX/TikZ code | Python/Matplotlib code |
| Compilation | pdflatex + poppler | exec() + plt.savefig() |
| Dependencies | TeX Live, ghostscript | matplotlib, numpy |
| Reward speed | ~30s (LaTeX compile) | ~2s (Python exec) |
| Adapter | TikZero (frozen, reusable) | Same adapter works |
| Decoder | Retrained for Python | Retrained for Python |

## Monitoring

```bash
tensorboard --logdir tf_logs_python_sft/   # SFT
tensorboard --logdir tf_logs_python_grpo/  # GRPO
```
