"""Evaluate SFT model: generate code, execute, save side-by-side comparison with DINOv2 score.

Features:
  - Incremental: appends results per sample, resumes from where it left off.
  - Fixed figure layout with centered score and clear GT/Generated labels.
  - Per-sample timing.

Usage:
    python eval_sft_vis.py --model_path <path> --data_dir <path> --render_dir <path> --out_dir <path>
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocessing import format_grpo_prompt
from rewards import DINOv2Scorer

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def _is_lora_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))

def load_model_and_tokenizer(model_path: str):
    if _is_lora_adapter(model_path):
        from peft import PeftConfig, PeftModel
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base_id = peft_cfg.base_model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def extract_code(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

IMAGE_SIZE = (448, 448)

def execute_python_to_image(code: str) -> tuple[Image.Image | None, str | None]:
    plt.close("all")
    safe = code.replace("plt.show()", "").replace("plt.ion()", "")
    restricted_builtins = {
        k: v for k, v in __builtins__.__dict__.items()
        if k not in ("exit", "quit", "breakpoint", "input")
    } if hasattr(__builtins__, "__dict__") else __builtins__
    ns = {"__builtins__": restricted_builtins, "__name__": "__main__"}

    try:
        exec(safe, ns)
        fig = plt.gcf()
        if not fig.get_axes():
            plt.close("all")
            return None, "No axes produced"
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white", edgecolor="none")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img.load()
        plt.close("all")
        return ImageOps.pad(img, IMAGE_SIZE, color="white"), None
    except Exception as e:
        plt.close("all")
        return None, str(e)


def _get_font(size: int = 20):
    """Try loading a TrueType font, fall back to default."""
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def create_comparison_image(
    gt_img: Image.Image,
    pred_img: Image.Image | None,
    score: float | None,
    error: str | None,
    elapsed: float | None = None,
) -> Image.Image:
    """Side-by-side: GT (left) | Pred (right) with score header and labels."""
    HEADER_H = 60
    LABEL_H = 30
    W, H = IMAGE_SIZE
    total_w = 2 * W
    total_h = HEADER_H + H + LABEL_H

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    font_big = _get_font(22)
    font_sm = _get_font(16)

    # --- header bar (light gray background) ---
    draw.rectangle([(0, 0), (total_w, HEADER_H)], fill=(240, 240, 240))
    score_str = f"DINOv2 Score: {score:.4f}" if score is not None else "DINOv2 Score: N/A"
    if elapsed is not None:
        score_str += f"  |  Time: {elapsed:.1f}s"
    bbox = draw.textbbox((0, 0), score_str, font=font_big)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((total_w - tw) / 2, (HEADER_H - th) / 2), score_str, fill="black", font=font_big)

    # --- images ---
    gt_padded = ImageOps.pad(gt_img, IMAGE_SIZE, color="white")
    canvas.paste(gt_padded, (0, HEADER_H))

    if pred_img:
        canvas.paste(pred_img, (W, HEADER_H))
    else:
        err_img = Image.new("RGB", IMAGE_SIZE, (255, 240, 240))
        d = ImageDraw.Draw(err_img)
        err_msg = (error or "Unknown Error")[:120]
        d.text((10, 10), f"Error:\n{err_msg}", fill="red", font=font_sm)
        canvas.paste(err_img, (W, HEADER_H))

    # --- bottom labels (centered under each image) ---
    label_y = HEADER_H + H + 4
    for label, x_off in [("Ground Truth", 0), ("Generated", W)]:
        lbox = draw.textbbox((0, 0), label, font=font_sm)
        lw = lbox[2] - lbox[0]
        draw.text((x_off + (W - lw) / 2, label_y), label, fill="gray", font=font_sm)

    # --- vertical separator ---
    draw.line([(W, HEADER_H), (W, HEADER_H + H)], fill="gray", width=2)

    return canvas


def load_done_ids(jsonl_path: Path) -> set[str]:
    """Read already-completed sample IDs from results file."""
    done = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--render_dir", required=True, help="Directory containing GT rendered images")
    parser.add_argument("--out_dir", default="eval_output")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
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
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Loading DINOv2 scorer...")
    scorer = DINOv2Scorer(device=device)

    print("Loading validation set...")
    dd = load_from_disk(data_dir)
    val = dd["validation"]
    if args.max_samples:
        val = val.select(range(min(args.max_samples, len(val))))
    n_total = len(val)
    
    print(f"Evaluating on {n_total} samples")

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None:
        eos_id = tokenizer.eos_token_id

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

            caption = sample["text"].strip()
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

            # 2. Generate Code
            messages = format_grpo_prompt(caption)
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096).to(device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or eos_id,
                    eos_token_id=eos_id,
                )
            
            gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
                "caption": caption,
                "code": gen_code,
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
