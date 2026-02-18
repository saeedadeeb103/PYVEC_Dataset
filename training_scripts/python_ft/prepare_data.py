"""Dataset preparation: JSONL → HuggingFace Arrow format with rendered images.

Renders each sample's Python/Matplotlib code to a PNG image for use as
ground-truth reward signal during GRPO training.  Rendering is parallelized
across CPU cores and cached to disk so re-runs are fast.

Usage:
    python prepare_data.py                           # defaults
    python prepare_data.py rendering.dpi=200         # override DPI
    python prepare_data.py rendering.skip=true       # rebuild from cache
"""

from __future__ import annotations

import hashlib
import io
import logging
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orjson
from datasets import Dataset, DatasetDict, Features, Image, Value
from omegaconf import DictConfig
from PIL import Image as PILImage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading & filtering
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                records.append(orjson.loads(line))
    return records


def get_best_caption(record: dict) -> str:
    """Extract the best available caption: new_caption (from original_data) > caption."""
    od = record.get("original_data", {})
    if isinstance(od, dict):
        nc = (od.get("new_caption") or "").strip()
        if nc:
            return nc
    return (record.get("caption") or "").strip()


def filter_records(records: list[dict], min_len: int, max_len: int) -> list[dict]:
    valid = []
    reasons: Counter = Counter()
    caption_sources: Counter = Counter()

    for r in records:
        code = (r.get("python_code") or "").strip()
        caption = get_best_caption(r)

        if not code:
            reasons["no_code"] += 1; continue
        if not caption:
            reasons["no_caption"] += 1; continue
        if not (min_len <= len(code) <= max_len):
            reasons["code_length"] += 1; continue
        if "import" not in code or "plt" not in code:
            reasons["no_matplotlib"] += 1; continue

        od = r.get("original_data", {})
        if isinstance(od, dict) and (od.get("new_caption") or "").strip():
            caption_sources["new_caption"] += 1
        else:
            caption_sources["caption"] += 1
        valid.append(r)

    log.info(f"Filtering: kept {len(valid)}/{len(records)}")
    for reason, count in reasons.most_common():
        log.info(f"  dropped {count:>5d} — {reason}")
    log.info(f"Caption sources: {dict(caption_sources)}")
    return valid


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _code_id(record: dict) -> str:
    rid = record.get("id", "")
    if rid:
        return rid
    return hashlib.sha256(record["python_code"].encode()).hexdigest()[:12]


def _render_one(code: str, dpi: int = 150) -> Optional[bytes]:
    plt.close("all")
    safe_code = code.replace("plt.show()", "").replace("plt.ion()", "")

    restricted_builtins = {
        k: v for k, v in __builtins__.__dict__.items()
        if k not in ("exit", "quit", "breakpoint", "input")
    } if hasattr(__builtins__, "__dict__") else __builtins__

    ns = {"__builtins__": restricted_builtins, "__name__": "__main__"}
    try:
        exec(safe_code, ns)  # noqa: S102
        fig = plt.gcf()
        if not fig.get_axes():
            plt.close("all"); return None
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close("all")
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        plt.close("all"); return None


def _render_worker(args: tuple) -> tuple[str, bool]:
    sample_id, code, render_dir, dpi, timeout = args
    out_path = render_dir / f"{sample_id}.png"
    if out_path.exists() and out_path.stat().st_size > 0:
        return (sample_id, True)
    png_bytes = _render_one(code, dpi=dpi)
    if png_bytes is None:
        return (sample_id, False)
    try:
        out_path.write_bytes(png_bytes)
        return (sample_id, True)
    except Exception:
        return (sample_id, False)


def render_all(records, render_dir, dpi, timeout, workers):
    render_dir.mkdir(parents=True, exist_ok=True)
    n_workers = workers or max(1, mp.cpu_count())

    tasks = []
    for r in records:
        sid = _code_id(r)
        tasks.append((sid, r["python_code"], render_dir, dpi, timeout))

    log.info(f"Rendering {len(tasks)} samples with {n_workers} workers (dpi={dpi}) ...")
    t0 = time.time()
    results = {}
    success = cached = 0

    with mp.Pool(n_workers, maxtasksperchild=50) as pool:
        for i, (sid, ok) in enumerate(pool.imap_unordered(_render_worker, tasks), 1):
            if ok:
                results[sid] = render_dir / f"{sid}.png"
                if (render_dir / f"{sid}.png").stat().st_mtime < t0:
                    cached += 1
                success += 1
            if i % 500 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                log.info(f"  [{i:>6d}/{len(tasks)}] success={success} "
                         f"cached={cached} failed={i - success} ({rate:.1f}/s)")

    log.info(f"Rendering done: {success}/{len(tasks)} succeeded "
             f"({cached} cached) in {time.time() - t0:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Build HF dataset
# ---------------------------------------------------------------------------
def build_dataset(records, render_dir, rendered_ids=None):
    data = {
        "text": [], "python_code": [], "rendered_image": [],
        "sample_id": [], "source": [], "origin_type": [],
    }
    skipped = 0
    for r in records:
        sid = _code_id(r)
        png_path = (rendered_ids or {}).get(sid, render_dir / f"{sid}.png")
        if not Path(png_path).exists():
            skipped += 1; continue
        try:
            img = PILImage.open(png_path).convert("RGB"); img.load()
        except Exception:
            skipped += 1; continue
        caption = get_best_caption(r)
        if not caption:
            continue
        data["text"].append(caption)
        data["python_code"].append(r["python_code"])
        data["rendered_image"].append(img)
        data["sample_id"].append(sid)
        data["source"].append(r.get("source", "unknown"))
        data["origin_type"].append(r.get("origin_type", "unknown"))

    if skipped:
        log.info(f"Skipped {skipped} samples without rendered images")

    features = Features({
        "text": Value("string"), "python_code": Value("string"),
        "rendered_image": Image(), "sample_id": Value("string"),
        "source": Value("string"), "origin_type": Value("string"),
    })
    return Dataset.from_dict(data, features=features)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra.main(config_path="configs", config_name="prepare_data", version_base="1.3")
def main(cfg: DictConfig):
    input_path = Path(cfg.paths.input)
    output_path = Path(cfg.paths.output)
    render_dir = Path(cfg.paths.render_dir)

    log.info(f"Loading {input_path}")
    raw = load_jsonl(input_path)
    log.info(f"Total records: {len(raw)}")

    filtered = filter_records(raw, cfg.filtering.min_code_len, cfg.filtering.max_code_len)

    rendered_ids = None
    if not cfg.rendering.skip:
        rendered_ids = render_all(
            filtered, render_dir,
            dpi=cfg.rendering.dpi,
            timeout=cfg.rendering.timeout,
            workers=cfg.rendering.workers,
        )

    log.info("Building HuggingFace dataset ...")
    ds = build_dataset(filtered, render_dir, rendered_ids)
    log.info(f"Dataset rows: {len(ds)}")

    src_counts = Counter(ds["source"])
    origin_counts = Counter(ds["origin_type"])
    log.info("Source distribution:")
    for src, cnt in src_counts.most_common(10):
        log.info(f"  {src}: {cnt}")
    log.info("Origin distribution:")
    for origin, cnt in origin_counts.most_common():
        log.info(f"  {origin}: {cnt}")

    n_val = max(1, int(len(ds) * cfg.split.val_ratio))
    split = ds.train_test_split(test_size=n_val, seed=cfg.split.seed)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})

    output_path.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(output_path))
    log.info(f"Saved to {output_path}")
    log.info(f"Train: {len(dd['train'])}, Validation: {len(dd['validation'])}")

    code_lens = np.array([len(r["python_code"]) for r in filtered])
    log.info(f"Code lengths: min={code_lens.min()}, median={int(np.median(code_lens))}, "
             f"mean={int(code_lens.mean())}, p95={int(np.percentile(code_lens, 95))}, "
             f"max={code_lens.max()}")


if __name__ == "__main__":
    main()
