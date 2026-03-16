"""Evaluate SFT model on TIGER-Lab/VisPlotBench (python config, matplotlib + plotly + seaborn).

Features:
  - Downloads VisPlotBench from HuggingFace, filters to matplotlib/plotly/seaborn.
  - Writes CSV data to temp files so generated code can read_csv / loadtxt.
  - Generates code from the task description, executes it, computes DINOv2 vs reference.
  - Incremental: appends results per sample, resumes from where it left off.
  - Reports execution pass rate, DINOv2 score, and per-sample timing.

Usage:
    python eval_visplotbench.py --model_path <path> --out_dir eval_output_visplotbench
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocessing import format_grpo_prompt
from rewards import DINOv2Scorer
from eval_sft_vis import (
    create_comparison_image,
    extract_code,
    load_done_ids,
    _get_font,
    IMAGE_SIZE,
)

ALLOWED_LIBS = {"matplotlib", "plotly", "seaborn"}

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


def build_prompt_text(task_desc: str, task_style: str, data_csv: str | None) -> str:
    """Build a natural-language prompt from VisPlotBench task fields."""
    parts = [task_desc.strip()]
    if task_style:
        parts.append(task_style.strip())
    if data_csv:
        # Truncate very long CSV data to avoid exceeding context
        lines = data_csv.strip().split("\n")
        if len(lines) > 30:
            data_csv = "\n".join(lines[:30]) + "\n... (truncated)"
        parts.append(f"Use the following CSV data (available as 'data.csv'):\n```\n{data_csv}\n```")
    return "\n\n".join(parts)


def execute_python_with_data(code: str, data_csv: str | None) -> tuple[Image.Image | None, str | None]:
    """Execute python code in a temp directory with CSV data written to common filenames."""
    plt.close("all")

    # Create a temp dir and write CSV data to various filenames the model might use
    with tempfile.TemporaryDirectory() as tmpdir:
        if data_csv:
            # Write to 'data.csv' and other common names the model might use
            csv_path = os.path.join(tmpdir, "data.csv")
            with open(csv_path, "w") as f:
                f.write(data_csv)

            # Also look for specific filenames the model might reference in the code
            # and create symlinks/copies for them
            common_names = _extract_csv_filenames(code)
            for name in common_names:
                target = os.path.join(tmpdir, name)
                if not os.path.exists(target):
                    with open(target, "w") as f:
                        f.write(data_csv)

        # Build safe code
        safe = code.replace("plt.show()", "").replace("plt.ion()", "")

        restricted_builtins = {
            k: v for k, v in __builtins__.__dict__.items()
            if k not in ("exit", "quit", "breakpoint", "input")
        } if hasattr(__builtins__, "__dict__") else __builtins__
        ns = {"__builtins__": restricted_builtins, "__name__": "__main__"}

        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            exec(safe, ns)
            fig = plt.gcf()
            if not fig.get_axes():
                plt.close("all")
                return None, "No axes produced"
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            img.load()
            plt.close("all")
            return ImageOps.pad(img, IMAGE_SIZE, color="white"), None
        except Exception as e:
            plt.close("all")
            return None, str(e)
        finally:
            os.chdir(old_cwd)


def _extract_csv_filenames(code: str) -> list[str]:
    """Extract potential CSV/data filenames from code strings like read_csv('tips.csv')."""
    # Match patterns like: read_csv("tips.csv"), loadtxt('data.csv'), open("file.csv")
    pattern = r"""(?:read_csv|loadtxt|genfromtxt|read_table|open)\s*\(\s*['\"]([^'\"]+\.(?:csv|tsv|txt|dat))['\"]"""
    matches = re.findall(pattern, code)
    # Only return basenames (no paths)
    return [os.path.basename(m) for m in matches]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", default="eval_output_visplotbench")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    # Resume support
    done_ids = load_done_ids(jsonl_path)
    if done_ids:
        print(f"Resuming — {len(done_ids)} samples already done, skipping them.")

    # ------------------------------------------------------------------
    # Load VisPlotBench – python config, filter to matplotlib/plotly/seaborn
    # ------------------------------------------------------------------
    libs_str = "/".join(sorted(ALLOWED_LIBS))
    print(f"Loading VisPlotBench dataset (python, {libs_str} only)...")
    ds = load_dataset("TIGER-Lab/VisPlotBench", "python", split="test")
    ds = ds.filter(lambda x: any(l in ALLOWED_LIBS for l in x["used_lib"]))
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    n_total = len(ds)
    print(f"  {n_total} samples after filtering")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Loading DINOv2 scorer...")
    scorer = DINOv2Scorer(device=device)

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None:
        eos_id = tokenizer.eos_token_id

    n_exec_pass = 0
    n_success = 0
    scores = []
    total_time = 0.0
    n_processed = 0

    with open(jsonl_path, "a") as fout:
        for i in range(n_total):
            sample = ds[i]
            sample_id = sample["id"]

            if sample_id in done_ids:
                continue

            t0 = time.time()

            # Build prompt from task fields
            task_desc = sample.get("task__plot_description", "")
            task_style = sample.get("task__plot_style", "")
            data_csv = sample.get("data", None)
            used_lib = sample.get("used_lib", [])
            ref_image = sample.get("image", None)  # PIL Image

            description = build_prompt_text(task_desc, task_style, data_csv)

            # Format into chat prompt
            messages = format_grpo_prompt(description)
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

            # Execute with CSV data available as files
            pred_img, error = execute_python_with_data(gen_code, data_csv)
            exec_pass = pred_img is not None
            if exec_pass:
                n_exec_pass += 1

            # Score against reference image
            dino_score = None
            if pred_img and ref_image is not None:
                try:
                    ref_rgb = ref_image.convert("RGB")
                    ref_padded = ImageOps.pad(ref_rgb, IMAGE_SIZE, color="white")
                    dino_score = scorer.score([pred_img], [ref_padded])[0]
                    scores.append(dino_score)
                    n_success += 1
                except Exception as e:
                    error = (error or "") + f" | Scoring error: {e}"

            elapsed = time.time() - t0
            total_time += elapsed
            n_processed += 1

            # Comparison image
            if ref_image is not None:
                ref_padded = ImageOps.pad(ref_image.convert("RGB"), IMAGE_SIZE, color="white")
            else:
                ref_padded = Image.new("RGB", IMAGE_SIZE, (200, 200, 200))
            comp_img = create_comparison_image(ref_padded, pred_img, dino_score, error, elapsed)
            comp_img.save(figs_dir / f"{sample_id}_comparison.png")

            # Append result
            record = {
                "sample_id": sample_id,
                "used_lib": used_lib,
                "vis_cate": sample.get("vis_cate", ""),
                "exec_pass": exec_pass,
                "dino_score": dino_score,
                "error": error,
                "elapsed_s": round(elapsed, 2),
                "code": gen_code,
                "figure_path": f"figures/{sample_id}_comparison.png"
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            if n_processed % 5 == 0:
                exec_rate = n_exec_pass / n_processed * 100
                avg_score = sum(scores) / len(scores) if scores else 0
                avg_time = total_time / n_processed
                print(f"  {n_processed}/{n_total} | ExecPass: {exec_rate:.0f}% | "
                      f"DINOv2: {avg_score:.4f} | Avg time: {avg_time:.1f}s")

    # Final summary
    exec_rate = n_exec_pass / max(n_processed, 1) * 100
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n{'='*60}")
    print(f"VisPlotBench Results ({n_processed} samples)")
    print(f"  Exec Pass Rate: {n_exec_pass}/{n_processed} = {exec_rate:.1f}%")
    print(f"  Avg DINOv2 Score: {avg_score:.4f} (on {n_success} successful)")
    print(f"  Total time: {total_time:.1f}s | Avg: {total_time/max(n_processed,1):.1f}s/sample")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
