"""Pre-download reward models to a local path for GRPO training.

Run this on the login node (where internet is available) before submitting
the GRPO job. Compute nodes often have restricted outbound access.

Usage:
    python download_reward_models.py [--models dinov2] [--out models/]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dinov2"], help="dinov2, clip, or both")
    ap.add_argument("--out", default="models", help="Output directory under CWD")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if "dinov2" in args.models:
        dest = out_root / "dinov2-large"
        if (dest / "config.json").exists():
            print(f"DINOv2 already cached at {dest}")
        else:
            print(f"Downloading facebook/dinov2-large to {dest}...")
            from huggingface_hub import snapshot_download
            snapshot_download("facebook/dinov2-large", local_dir=str(dest))
            print(f"  Done: {dest}")

    if "clip" in args.models:
        dest = out_root / "clip-vit-large-patch14"
        if (dest / "config.json").exists():
            print(f"CLIP already cached at {dest}")
        else:
            print(f"Downloading openai/clip-vit-large-patch14 to {dest}...")
            from huggingface_hub import snapshot_download
            snapshot_download("openai/clip-vit-large-patch14", local_dir=str(dest))
            print(f"  Done: {dest}")

    print("Reward models ready. Run train_grpo.sbatch with reward.model_name=<local_path> or use dinov2_local config.")


if __name__ == "__main__":
    main()
