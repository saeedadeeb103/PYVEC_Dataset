"""Reward functions for GRPO training with Python/Matplotlib execution feedback.

Multi-signal rewards:
  1. Execution reward — does the generated code run and produce a figure?
  2. Visual similarity — how close is the rendered figure to the ground truth?

The final reward = execution_bonus + visual_similarity, which gives the model
a gradient signal even for code that runs but looks wrong, and zero reward
for code that crashes.
"""

from __future__ import annotations

import hashlib
import io
import os
import signal
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------
EXEC_TIMEOUT = 30  # seconds per code execution
RENDER_DPI = 100
IMAGE_SIZE = (448, 448)

# Baseline reward for successfully producing a figure (even if it looks wrong)
EXECUTION_BONUS = 0.1


def execute_python_to_image(
    code: str,
    dpi: int = RENDER_DPI,
    timeout: int = EXEC_TIMEOUT,
) -> Optional[Image.Image]:
    """Execute Python/Matplotlib code and capture the figure as a PIL Image.

    Returns None if the code fails to execute or produces no axes.
    """
    plt.close("all")

    safe = code.replace("plt.show()", "").replace("plt.ion()", "")

    restricted_builtins = {
        k: v for k, v in __builtins__.__dict__.items()
        if k not in ("exit", "quit", "breakpoint", "input")
    } if hasattr(__builtins__, "__dict__") else __builtins__

    ns = {"__builtins__": restricted_builtins, "__name__": "__main__"}

    try:
        exec(safe, ns)  # noqa: S102
        fig = plt.gcf()
        if not fig.get_axes():
            plt.close("all")
            return None
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img.load()
        plt.close("all")
        return ImageOps.pad(img, IMAGE_SIZE, color="white")
    except Exception:
        plt.close("all")
        return None


def _exec_worker(code: str) -> Optional[Image.Image]:
    """Wrapper for ProcessPoolExecutor."""
    return execute_python_to_image(code)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
class CLIPScorer:
    """Image similarity using CLIP cosine similarity (fast, no detikzify needed)."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[torch.device] = None,
    ):
        from transformers import AutoModel, AutoProcessor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(model_name)

    def _encode(self, images: list[Image.Image]) -> torch.Tensor:
        images_rgb = [img.convert("RGB") for img in images]
        enc = self.proc(images=images_rgb, return_tensors="pt")
        px = enc["pixel_values"].to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            feats = self.model.get_image_features(pixel_values=px)
        feats = feats.detach().cpu().float()
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def score(self, images_a: list[Image.Image], images_b: list[Image.Image]) -> list[float]:
        fa = self._encode(images_a)
        fb = self._encode(images_b)
        cos = (fa * fb).sum(dim=-1)
        return ((cos + 1.0) / 2.0).clamp(0, 1).tolist()


class DINOv2Scorer:
    """Image similarity using DINOv2 CLS token cosine similarity.

    DINOv2 is ideal for scientific figure comparison because:
    - Self-supervised vision model — captures structural/spatial features
    - Strong at layout, shape, and color matching (better than CLIP for image-image)
    - Fast inference, no text encoder overhead
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        device: Optional[torch.device] = None,
    ):
        from transformers import AutoModel, AutoImageProcessor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoImageProcessor.from_pretrained(model_name)

    def _encode(self, images: list[Image.Image]) -> torch.Tensor:
        images_rgb = [img.convert("RGB") for img in images]
        enc = self.proc(images=images_rgb, return_tensors="pt")
        px = enc["pixel_values"].to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            out = self.model(pixel_values=px)
            # Use CLS token (first token of last_hidden_state)
            feats = out.last_hidden_state[:, 0, :].detach().cpu().float()
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def score(self, images_a: list[Image.Image], images_b: list[Image.Image]) -> list[float]:
        fa = self._encode(images_a)
        fb = self._encode(images_b)
        cos = (fa * fb).sum(dim=-1)
        # Map cosine [-1, 1] → [0, 1]
        return ((cos + 1.0) / 2.0).clamp(0, 1).tolist()


class DreamSimScorer:
    """Image similarity using DreamSim perceptual distance."""

    def __init__(
        self,
        model_name: str = "ensemble",
        cache_dir: str = "models/dreamsim",
        device: Optional[torch.device] = None,
    ):
        from dreamsim import dreamsim

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float32
        )

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        model, processor = dreamsim(
            dreamsim_type=model_name, pretrained=True,
            normalize_embeds=True, device=self.device, cache_dir=cache_dir,
        )
        for ext in model.extractor_list:
            ext.model = ext.model.to(self.dtype)
            ext.proj = ext.proj.to(self.dtype)
        self.model = model.to(self.device, self.dtype)
        self.processor = processor

    def score(self, images_a: list[Image.Image], images_b: list[Image.Image]) -> list[float]:
        scores = []
        for a, b in zip(images_a, images_b):
            try:
                max_dim = max(a.width, a.height, b.width, b.height)
                a_pad = ImageOps.pad(a.convert("RGB"), (max_dim, max_dim), color="white")
                b_pad = ImageOps.pad(b.convert("RGB"), (max_dim, max_dim), color="white")
                ta = self.processor(a_pad).to(self.device, self.dtype)
                tb = self.processor(b_pad).to(self.device, self.dtype)
                with torch.inference_mode():
                    dist = self.model(ta, tb).item()
                scores.append(max(0.0, min(1.0, 1.0 - dist)))
            except Exception:
                scores.append(0.0)
        return scores


class DualScorer:
    """Weighted combination of DINOv2 (structural) + CLIP (semantic).

    DINOv2 captures layout/spatial similarity while CLIP adds semantic
    alignment — useful when the reward should also reflect whether the
    figure *means* the right thing, not just looks alike structurally.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        dino_weight: float = 0.7,
        clip_weight: float = 0.3,
        device: Optional[torch.device] = None,
    ):
        self.dino = DINOv2Scorer(model_name, device=device)
        self.clip = CLIPScorer("openai/clip-vit-large-patch14", device=device)
        self.dw = dino_weight
        self.cw = clip_weight

    def score(self, images_a: list[Image.Image], images_b: list[Image.Image]) -> list[float]:
        ds = self.dino.score(images_a, images_b)
        cs = self.clip.score(images_a, images_b)
        return [self.dw * d + self.cw * c for d, c in zip(ds, cs)]


# ---------------------------------------------------------------------------
# Composite reward function
# ---------------------------------------------------------------------------
_MAX_EXEC_WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))


def make_python_reward(
    scorer: Union[CLIPScorer, DINOv2Scorer, DreamSimScorer, DualScorer],
    execution_bonus: float = EXECUTION_BONUS,
    max_workers: int = _MAX_EXEC_WORKERS,
):
    """Build a GRPO-compatible reward function.

    Reward = execution_bonus (if code runs and produces a figure)
           + (1 - execution_bonus) * visual_similarity

    This gives partial credit for code that runs but produces wrong output,
    and zero for code that crashes — creating a clear gradient signal for the
    model to first learn to produce valid code, then improve visual fidelity.
    """

    def reward_fn(
        completions: list[str],
        ground_truth: list[str],
        reward_signal: list,
        **batch,
    ) -> list[float]:
        # Step 1: Execute all completions in parallel
        rendered: list[Optional[Image.Image]] = []
        with ProcessPoolExecutor(max_workers) as pool:
            futures = [pool.submit(_exec_worker, code) for code in completions]
            for fut in futures:
                try:
                    rendered.append(fut.result(timeout=EXEC_TIMEOUT + 5))
                except Exception:
                    rendered.append(None)

        # Step 2: Collect valid pairs
        valid_rendered, valid_gt = [], []
        valid_indices = []

        for i, (img_r, img_gt) in enumerate(zip(rendered, reward_signal)):
            if img_r is None:
                continue
            try:
                gt_img = (
                    img_gt if isinstance(img_gt, Image.Image)
                    else Image.open(img_gt).convert("RGB")
                )
                gt_img = ImageOps.pad(gt_img, IMAGE_SIZE, color="white")
                valid_rendered.append(img_r)
                valid_gt.append(gt_img)
                valid_indices.append(i)
            except Exception:
                continue

        # Step 3: Compute rewards
        rewards = [0.0] * len(completions)

        # Execution bonus for all samples that produced a figure
        for i in range(len(completions)):
            if rendered[i] is not None:
                rewards[i] = execution_bonus

        # Visual similarity for valid pairs
        if valid_indices:
            sim_scores = scorer.score(valid_rendered, valid_gt)
            for idx, sim in zip(valid_indices, sim_scores):
                rewards[idx] = execution_bonus + (1.0 - execution_bonus) * sim

        return rewards

    reward_fn.__name__ = "python_execution_reward"
    return reward_fn
