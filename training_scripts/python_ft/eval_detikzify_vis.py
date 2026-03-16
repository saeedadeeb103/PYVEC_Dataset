"""Evaluate DeTikZify model: Image -> Code, execute, save side-by-side comparison with DINOv2 score.

Features:
  - Incremental: appends results per sample, resumes from where it left off.
  - Fixed figure layout with centered score and clear GT/Generated labels.
  - Per-sample timing.

Usage:
    python eval_detikzify_vis.py --model_path <path> --base_model_path <path> --render_dir <path> --out_dir <path>
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_from_disk
from transformers import AutoProcessor

from rewards import DINOv2Scorer
from eval_sft_vis import (
    execute_python_to_image,
    create_comparison_image,
    extract_code,
    load_done_ids,
    IMAGE_SIZE,
)

# ---------------------------------------------------------------------------
# DeTikZify Model Loading
# ---------------------------------------------------------------------------
def load_detikzify(model_path: str):
    """Load DeTikZify model + processor, registering custom architecture."""
    detikzify_path = Path(__file__).resolve().parent / "DeTikZify"
    if str(detikzify_path) not in sys.path:
        sys.path.append(str(detikzify_path))
        
    from detikzify.model import load, register
    register()

    model, processor = load(
        model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Force slow (PIL-based) image processor to avoid fast-processor issues
    slow_processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    processor.image_processor = slow_processor.image_processor
    
    return model, processor

def apply_lora(model, model_path):
    from peft import PeftModel
    print(f"Loading LoRA adapter from {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    return model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model_path", required=True, help="Path to base DeTikZify model")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--render_dir", required=True, help="Directory containing GT rendered images")
    parser.add_argument("--out_dir", default="eval_output_detikzify")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    work_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or str(work_dir / "data" / "python_ft")
    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    # Resume support
    done_ids = load_done_ids(jsonl_path)
    if done_ids:
        print(f"Resuming — {len(done_ids)} samples already done, skipping them.")

    print("Loading model...")
    model, processor = load_detikzify(args.base_model_path)
    
    # Apply LoRA if model_path is different and looks like an adapter
    if args.model_path != args.base_model_path and os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        model = apply_lora(model, args.model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading DINOv2 scorer...")
    scorer = DINOv2Scorer(device=device)

    print("Loading validation set...")
    dd = load_from_disk(data_dir)
    val = dd["validation"]
    if args.max_samples:
        val = val.select(range(min(args.max_samples, len(val))))
    n_total = len(val)
    
    print(f"Evaluating on {n_total} samples")

    n_success = 0
    scores = []
    total_time = 0.0
    n_processed = 0

    # Open in append mode for incremental writing
    with open(jsonl_path, "a") as fout:
        for i in range(n_total):
            sample = val[i]
            sample_id = sample["sample_id"]

            # Skip already-done samples
            if sample_id in done_ids:
                continue
            
            t0 = time.time()

            # 1. Load GT Image
            gt_path = os.path.join(args.render_dir, f"{sample_id}.png")
            if not os.path.exists(gt_path):
                print(f"Warning: GT image not found for {sample_id} at {gt_path}")
                continue
            
            try:
                gt_img = Image.open(gt_path).convert("RGB")
            except Exception as e:
                print(f"Error loading GT image {sample_id}: {e}")
                continue

            # 2. Generate Code — processor adds image tokens automatically
            inputs = processor(
                text="",
                images=gt_img,
                return_tensors="pt"
            ).to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            gen_text = processor.decode(out[0], skip_special_tokens=True)
            gen_code = extract_code(gen_text)

            # 3. Execute and Render
            pred_img, error = execute_python_to_image(gen_code)
            
            dino_score = None
            if pred_img:
                gt_img_padded = ImageOps.pad(gt_img, IMAGE_SIZE, color="white")
                dino_score = scorer.score([pred_img], [gt_img_padded])[0]
                scores.append(dino_score)
                n_success += 1

            elapsed = time.time() - t0
            total_time += elapsed
            n_processed += 1
            
            # 5. Create Comparison Image
            gt_img_padded = ImageOps.pad(gt_img, IMAGE_SIZE, color="white")
            comp_img = create_comparison_image(gt_img_padded, pred_img, dino_score, error, elapsed)
            comp_img.save(figs_dir / f"{sample_id}_comparison.png")

            # 6. Append result immediately
            record = {
                "sample_id": sample_id,
                "dino_score": dino_score,
                "error": error,
                "elapsed_s": round(elapsed, 2),
                "figure_path": f"figures/{sample_id}_comparison.png"
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            if n_processed % 10 == 0:
                avg_score = sum(scores) / len(scores) if scores else 0
                avg_time = total_time / n_processed
                print(f"  {n_processed} done | Success: {n_success} | Avg DINO: {avg_score:.4f} | Avg time: {avg_time:.1f}s")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nFinal Results: {n_success}/{n_processed} success. Avg DINOv2 Score: {avg_score:.4f}")
    print(f"Total time: {total_time:.1f}s | Avg per sample: {total_time/max(n_processed,1):.1f}s")

if __name__ == "__main__":
    main()
