#!/usr/bin/env python3
"""
Generate synthetic TikZ → Python/Matplotlib data for PYVEC using LLMs.

This script runs the same conversion pipeline used in the parent repo
(Gemini-3-pro, Claude, GPT, etc.) and optionally writes PYVEC-ready
JSONL with source=synthetic and origin_type=tikz_converted.

Run from the repository root (parent of PYVEC_Dataset):

  # Using Gemini (e.g. Gemini-3-pro)
  python PYVEC_Dataset/scripts/generate_synthetic_tikz_to_python.py \\
    --provider gemini --model google/gemini-3-pro \\
    --input cached_data/samples.jsonl --limit 500

  # Using OpenAI-compatible API (Claude, GPT)
  python PYVEC_Dataset/scripts/generate_synthetic_tikz_to_python.py \\
    --provider openai --model anthropic/claude-sonnet-4.5 \\
    --input cached_data/samples.jsonl --output ./output_openai --limit 500

  # Write PYVEC-ready dataset (source=synthetic, origin_type=tikz_converted)
  python PYVEC_Dataset/scripts/generate_synthetic_tikz_to_python.py \\
    --provider gemini --model google/gemini-3-pro \\
    --input cached_data/samples.jsonl --limit 500 \\
    --pyvec-output PYVEC_Dataset/synthetic_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    """Repository root = parent of PYVEC_Dataset (parent of script's parent's parent)."""
    script_dir = Path(__file__).resolve().parent
    # script_dir = .../PYVEC_Dataset/scripts  -> parent = PYVEC_Dataset, parent.parent = repo root
    return script_dir.parent.parent


def run_gemini_conversion(
    input_path: str,
    output_dir: str,
    model: str,
    limit: int,
    start: int,
    workers: int,
    no_resume: bool,
    sequential: bool,
    no_exec_validate: bool,
) -> str:
    """Run conversion via main.py (Gemini pipeline). Returns output_dir."""
    root = repo_root()
    sys.path.insert(0, str(root))
    os.chdir(root)

    import src.config as config

    config.OUTPUT_DIR = output_dir
    config.MODEL_NAME = model
    config.API_PROVIDER = "gemini"
    if "openai" in model.lower() or "gpt-" in model.lower():
        config.API_PROVIDER = "openai"
    if no_exec_validate:
        config.VALIDATE_BY_EXECUTION = False
    config.MAX_CONCURRENT = workers

    from main import load_from_jsonl
    from src.application.pipeline import process_dataset

    dataset = load_from_jsonl(input_path, limit=limit, start=start)
    if not dataset:
        print("No samples to process.")
        return output_dir
    process_dataset(
        dataset,
        resume=not no_resume,
        concurrent=not sequential,
    )
    return output_dir


def run_openai_conversion(
    input_path: str,
    output_dir: str,
    model: str,
    limit: int,
    start: int,
    workers: int,
    no_resume: bool,
    sequential: bool,
) -> str:
    """Run conversion via main_openai.py. Returns output_dir."""
    root = repo_root()
    os.chdir(root)
    cmd = [
        sys.executable,
        "main_openai.py",
        "--input", input_path,
        "--output", output_dir,
        "--model", model,
        "--limit", str(limit),
        "--start", str(start),
        "--workers", str(workers),
    ]
    if no_resume:
        cmd.append("--no-resume")
    if sequential:
        cmd.append("--sequential")
    subprocess.run(cmd, check=True, cwd=root)
    return output_dir


def write_pyvec_synthetic(
    conversion_jsonl: str,
    pyvec_output: str,
    origin_type: str = "tikz_converted",
) -> None:
    """
    Read conversion output and write PYVEC-ready JSONL with source=synthetic,
    origin_type, and optional original_data for new_caption.
    """
    out_path = Path(pyvec_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(conversion_jsonl, "r") as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("conversion_status") != "success" or not rec.get("python_code"):
                continue
            pyvec = {
                "id": rec.get("id"),
                "python_code": rec.get("python_code", ""),
                "caption": rec.get("caption", ""),
                "source": "synthetic",
                "origin_type": origin_type,
            }
            if rec.get("new_caption"):
                pyvec["original_data"] = {"new_caption": rec["new_caption"]}
            if rec.get("code"):
                pyvec["code"] = rec["code"]
            f_out.write(json.dumps(pyvec) + "\n")
            count += 1
    print(f"Wrote {count} PYVEC synthetic records to {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic TikZ→Python data for PYVEC using Gemini or OpenAI-compatible models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--provider", choices=["gemini", "openai"], default="gemini",
                    help="Pipeline: gemini (main.py) or openai (main_openai.py)")
    ap.add_argument("--model", "-m", default="google/gemini-3-pro",
                    help="Model: google/gemini-3-pro, anthropic/claude-sonnet-4.5, openai/gpt-5.2, etc.")
    ap.add_argument("--input", "-i", default="cached_data/samples.jsonl",
                    help="Input JSONL (TikZ samples, e.g. from quick_download.py)")
    ap.add_argument("--output", "-o", default="./output",
                    help="Output directory for conversion (dataset.jsonl + images)")
    ap.add_argument("--limit", "-l", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", "-w", type=int, default=5)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--no-exec-validate", action="store_true",
                    help="Skip execution validation (faster, Gemini only)")
    ap.add_argument("--pyvec-output", type=str, default=None,
                    help="Write PYVEC-ready JSONL with source=synthetic to this path")
    ap.add_argument("--origin-type", type=str, default="tikz_converted",
                    help="Value for origin_type in PYVEC output (default: tikz_converted)")
    args = ap.parse_args()

    if not Path(args.input).exists():
        print(f"Input file not found: {args.input}")
        print("Create it with: python quick_download.py --num 1000")
        sys.exit(1)

    if args.provider == "gemini":
        run_gemini_conversion(
            args.input,
            args.output,
            args.model,
            args.limit,
            args.start,
            args.workers,
            args.no_resume,
            args.sequential,
            args.no_exec_validate,
        )
    else:
        run_openai_conversion(
            args.input,
            args.output,
            args.model,
            args.limit,
            args.start,
            args.workers,
            args.no_resume,
            args.sequential,
        )

    conversion_jsonl = os.path.join(args.output, "dataset.jsonl")
    if args.pyvec_output and os.path.exists(conversion_jsonl):
        write_pyvec_synthetic(
            conversion_jsonl,
            args.pyvec_output,
            origin_type=args.origin_type,
        )


if __name__ == "__main__":
    main()
