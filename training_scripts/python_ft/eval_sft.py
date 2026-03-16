"""Evaluate SFT model on validation set: generate code, execute, save outputs and stats.

Usage:
    python eval_sft.py --model_path <path_to_lora_or_model> --data_dir <path_to_data> --out_dir <output_dir>
    python eval_sft.py --model_path trained_models_sft_python/pyvec_sft_Qwen2.5-3B-Instruct_lr2e-05_ep3_bs2x16_lora128/final

Outputs:
    - {out_dir}/results.jsonl     Caption + code per line, with figure_path linking to figures/
    - {out_dir}/figures/          Rendered figures ({sample_id}.png or {sample_id}_error.png)
    - {out_dir}/summary.json      Summary stats (n_total, n_success, n_error, success_rate)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocessing import SYSTEM_PROMPT, USER_TEMPLATE, format_grpo_prompt


# ---------------------------------------------------------------------------
# Model loading (LoRA-aware)
# ---------------------------------------------------------------------------
def _is_lora_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def load_model_and_tokenizer(model_path: str):
    if _is_lora_adapter(model_path):
        from peft import PeftConfig, PeftModel
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base_id = peft_cfg.base_model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------
def extract_code(text: str) -> str:
    """Extract Python code from model output (handles ```python ... ``` blocks)."""
    text = text.strip()
    # Try code block first
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: assume entire output is code
    return text


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
IMAGE_SIZE = (448, 448)
EXEC_TIMEOUT = 30


def execute_python_to_image(code: str) -> tuple[str | None, str | None]:
    """Execute Python/Matplotlib code. Returns (image, error_message)."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        required=True,
        help="Path to SFT model (LoRA adapter or full checkpoint)",
    )
    ap.add_argument(
        "--data_dir",
        default=None,
        help="Path to HF dataset dir (default: data/python_ft)",
    )
    ap.add_argument(
        "--out_dir",
        default="eval_output",
        help="Output directory for code, figures, results.json",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max validation samples (default: all)",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max tokens to generate",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (lower = more deterministic)",
    )
    args = ap.parse_args()

    work_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or str(work_dir / "data" / "python_ft")
    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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
    n_error = 0

    with open(jsonl_path, "w") as fout:
        for i in range(n_total):
            sample = val[i]
            sample_id = sample["sample_id"]
            caption = sample["text"].strip()
            gt_code = sample["python_code"].strip()

            messages = format_grpo_prompt(caption)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(device)

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

            # Execute and save figure
            img, err = execute_python_to_image(gen_code)
            if img is not None:
                fig_filename = f"{sample_id}.png"
                img.save(figs_dir / fig_filename)
                n_success += 1
                status = "success"
            else:
                fig_filename = f"{sample_id}_error.png"
                err_img = Image.new("RGB", IMAGE_SIZE, color=(255, 240, 240))
                try:
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(err_img)
                    err_short = (err or "unknown")[:60].replace("\n", " ")
                    draw.text((10, 10), f"Error: {err_short}", fill=(180, 0, 0))
                except Exception:
                    pass
                err_img.save(figs_dir / fig_filename)
                n_error += 1
                status = "error"

            # Write jsonl line: caption, code, figure_path
            figure_path = f"figures/{fig_filename}"
            record = {
                "sample_id": sample_id,
                "caption": caption,
                "code": gen_code,
                "figure_path": figure_path,
                "status": status,
            }
            if err:
                record["error"] = err
            fout.write(json.dumps(record) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n_total} (success: {n_success}, error: {n_error})")

    summary = {
        "n_total": n_total,
        "n_success": n_success,
        "n_error": n_error,
        "success_rate": n_success / n_total if n_total else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples:    {n_total}")
    print(f"Produced figure:  {n_success} ({100 * summary['success_rate']:.1f}%)")
    print(f"Code error:       {n_error}")
    print()
    print(f"Outputs saved to: {out_dir}")
    print(f"  - results.jsonl: {jsonl_path}")
    print(f"  - figures/:      {figs_dir}")
    print(f"  - summary.json:  {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
