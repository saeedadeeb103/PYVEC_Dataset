"""Dataset preparation: JSONL → HuggingFace Arrow format for SFT/GRPO training."""

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import orjson
import torch
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare TikZ→Python dataset for training.")
    p.add_argument("--input", "-i", type=Path, required=True, help="Path to dataset.jsonl")
    p.add_argument("--images-dir", type=Path, default=None, help="Image directory (default: input parent)")
    p.add_argument("--output", "-o", type=Path, default=Path("data/python_ft"), help="Output directory")
    p.add_argument("--val-ratio", type=float, default=0.066, help="Validation split ratio")
    p.add_argument("--min-code-len", type=int, default=50, help="Minimum Python code length")
    p.add_argument("--max-code-len", type=int, default=8000, help="Maximum Python code length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--render", action="store_true", help="Render Python codes to images for reward signal")
    p.add_argument("--render-dir", type=Path, default=None, help="Directory for rendered images")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                records.append(orjson.loads(line))
    return records


def filter_records(records: list[dict], min_len: int, max_len: int) -> list[dict]:
    valid = []
    for r in records:
        code = r.get("python_code") or ""
        if r.get("conversion_status") != "success":
            continue
        if not (min_len <= len(code) <= max_len):
            continue
        if "import" not in code or "plt" not in code:
            continue
        valid.append(r)
    return valid


def resolve_image(record: dict, images_dir: Path) -> Optional[PILImage.Image]:
    img_path = record.get("image_path", "")
    if not img_path:
        return None
    full = images_dir / img_path
    if not full.exists():
        return None
    try:
        img = PILImage.open(full).convert("RGB")
        img.load()
        return img
    except Exception:
        return None


def render_python_code(code: str, dpi: int = 100) -> Optional[bytes]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    safe_code = code.replace("plt.show()", "")
    ns = {"plt": plt, "__builtins__": __builtins__, "__name__": "__main__"}
    try:
        exec(safe_code, ns)
        fig = plt.gcf()
        if not fig.get_axes():
            plt.close("all")
            return None
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close("all")
        return buf.getvalue()
    except Exception:
        plt.close("all")
        return None


def build_dataset(records: list[dict], images_dir: Path, do_render: bool, render_dir: Optional[Path]) -> Dataset:
    data = {"image": [], "text": [], "code": [], "python_code": [], "sample_id": []}

    if do_render:
        data["rendered_image"] = []
        if render_dir:
            render_dir.mkdir(parents=True, exist_ok=True)

    for r in records:
        img = resolve_image(r, images_dir)
        if img is None:
            continue

        caption = r.get("caption") or r.get("new_caption") or ""
        if not caption.strip():
            continue

        data["image"].append(img)
        data["text"].append(caption.strip())
        data["code"].append(r.get("code", ""))
        data["python_code"].append(r["python_code"])
        data["sample_id"].append(r.get("id", ""))

        if do_render:
            png_bytes = render_python_code(r["python_code"])
            if png_bytes and render_dir:
                out_path = render_dir / f"{r.get('id', 'unknown')}.png"
                out_path.write_bytes(png_bytes)
                data["rendered_image"].append(PILImage.open(io.BytesIO(png_bytes)).convert("RGB"))
            elif png_bytes:
                data["rendered_image"].append(PILImage.open(io.BytesIO(png_bytes)).convert("RGB"))
            else:
                data["rendered_image"].append(None)

    features = {
        "image": Image(),
        "text": Value("string"),
        "code": Value("string"),
        "python_code": Value("string"),
        "sample_id": Value("string"),
    }
    if do_render:
        features["rendered_image"] = Image()

    return Dataset.from_dict(data, features=Features(features))


def main():
    args = parse_args()
    images_dir = args.images_dir or args.input.parent

    print(f"Loading {args.input}")
    raw = load_jsonl(args.input)
    print(f"  Total records: {len(raw)}")

    filtered = filter_records(raw, args.min_code_len, args.max_code_len)
    print(f"  After filtering: {len(filtered)}")

    ds = build_dataset(filtered, images_dir, args.render, args.render_dir)
    print(f"  Dataset rows: {len(ds)}")

    n_val = max(1, int(len(ds) * args.val_ratio))
    split = ds.train_test_split(test_size=n_val, seed=args.seed)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})

    args.output.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(args.output))
    print(f"  Saved to {args.output}")
    print(f"  Train: {len(dd['train'])}, Validation: {len(dd['validation'])}")


if __name__ == "__main__":
    main()
