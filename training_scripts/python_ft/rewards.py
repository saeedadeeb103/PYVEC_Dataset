"""Reward functions for GRPO training with Python/Matplotlib execution feedback."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchmetrics.functional import pairwise_cosine_similarity


def _file_cache_key(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{path}|{int(st.st_mtime)}|{st.st_size}"
    except Exception:
        return f"{path}|-1|-1"


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:12]


def execute_python_to_image(code: str, dpi: int = 100) -> Optional[Image.Image]:
    plt.close("all")
    safe = code.replace("plt.show()", "")
    ns = {"plt": plt, "__builtins__": __builtins__, "__name__": "__main__"}
    try:
        exec(safe, ns)
        fig = plt.gcf()
        if not fig.get_axes():
            plt.close("all")
            return None
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img.load()
        plt.close("all")
        return ImageOps.pad(img, (448, 448), color="white")
    except Exception:
        plt.close("all")
        return None


class VisionScorer:
    """Image similarity scoring using a frozen vision encoder (DeTikZifyw, SigLIP)."""

    def __init__(self, model_path: str, device: Optional[torch.device] = None, batch_size: int = 16, cache_size: int = 4096):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self.batch_size = batch_size
        self._cache = OrderedDict()
        self._cache_size = cache_size

        from detikzify.model import load as load_model
        from detikzify.util import expand

        model, processor = load_model(model_name_or_path=model_path, torch_dtype=self.dtype)
        self._vision = getattr(getattr(model, "model", model), "vision_model")
        self._vision.to(self.device).eval()
        for p in self._vision.parameters():
            p.requires_grad_(False)
        self._proc = getattr(processor, "image_processor", processor)
        self._expand = expand

    def _preprocess(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        return self._expand(img, max(img.size), do_trim=True)

    def _encode_batch(self, images: list[Image.Image]) -> list[torch.Tensor]:
        processed = [self._preprocess(im) for im in images]
        enc = self._proc(images=processed, return_tensors="pt")
        px = enc["pixel_values"]
        if px.ndim == 3:
            px = px.unsqueeze(0)
        px = px.to(self.device, dtype=self.dtype, non_blocking=True)
        with torch.inference_mode():
            out = self._vision(pixel_values=px)
            feats = out.last_hidden_state.detach().cpu().float()
        return [feats[i] for i in range(feats.shape[0])]

    def _get_features(self, images: list[Image.Image], keys: list[str]) -> list[Optional[torch.Tensor]]:
        feats = [None] * len(images)
        uncached_idx, uncached_imgs = [], []

        for i, (img, key) in enumerate(zip(images, keys)):
            if key in self._cache:
                feats[i] = self._cache[key]
                self._cache.move_to_end(key)
            else:
                uncached_idx.append(i)
                uncached_imgs.append(img)

        for start in range(0, len(uncached_imgs), self.batch_size):
            batch = uncached_imgs[start:start + self.batch_size]
            batch_idx = uncached_idx[start:start + self.batch_size]
            encoded = self._encode_batch(batch)
            for j, feat in enumerate(encoded):
                idx = batch_idx[j]
                feats[idx] = feat
                key = keys[idx]
                self._cache[key] = feat
                self._cache.move_to_end(key)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

        return feats

    @staticmethod
    def _emd_score(x: torch.Tensor, y: torch.Tensor) -> float:
        from ot.lp import emd2
        dists = 1.0 - pairwise_cosine_similarity(x.double(), y.double())
        score = 2 * np.tanh(-emd2(M=dists.numpy(), a=[], b=[])) + 1
        return float(np.clip(score, 0.0, 1.0))

    def score(self, images_a: list[Image.Image], images_b: list[Image.Image]) -> list[float]:
        keys_a = [f"a_{id(img)}" for img in images_a]
        keys_b = [f"b_{id(img)}" for img in images_b]
        feats_a = self._get_features(images_a, keys_a)
        feats_b = self._get_features(images_b, keys_b)
        scores = []
        for fa, fb in zip(feats_a, feats_b):
            if fa is None or fb is None:
                scores.append(0.0)
                continue
            try:
                scores.append(self._emd_score(fa, fb))
            except Exception:
                scores.append(0.0)
        return scores


class CLIPScorer:
    """Image similarity using CLIP cosine similarity."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: Optional[torch.device] = None):
        from transformers import AutoModel, AutoProcessor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(model_name)

    def _encode(self, images: list[Image.Image]) -> torch.Tensor:
        enc = self.proc(images=images, return_tensors="pt")
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


class DreamSimScorer:
    """Image similarity using DreamSim."""

    def __init__(self, model_name: str = "ensemble", cache_dir: str = "models/dreamsim", device: Optional[torch.device] = None):
        from dreamsim import dreamsim

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        model, processor = dreamsim(dreamsim_type=model_name, pretrained=True, normalize_embeds=True, device=self.device, cache_dir=cache_dir)
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


_MAX_EXEC_WORKERS = max(2, min(8, os.cpu_count() // 2))


def make_python_reward(scorer: VisionScorer | CLIPScorer | DreamSimScorer):
    """Build a GRPO-compatible reward function using Python execution + image similarity."""

    def reward_fn(completions: list[str], ground_truth: list[str], reward_signal: list, **batch) -> list[float]:
        rendered, gt_images = [], []
        valid_indices = []

        with ThreadPoolExecutor(_MAX_EXEC_WORKERS) as pool:
            futures = [pool.submit(execute_python_to_image, code) for code in completions]
            rendered = [f.result() for f in futures]

        for i, (img_r, img_gt) in enumerate(zip(rendered, reward_signal)):
            if img_r is None or img_gt is None:
                continue
            try:
                gt_img = img_gt if isinstance(img_gt, Image.Image) else Image.open(img_gt).convert("RGB")
                gt_img = ImageOps.pad(gt_img, (448, 448), color="white")
                gt_images.append(gt_img)
                valid_indices.append(i)
            except Exception:
                continue

        rewards = [0.0] * len(completions)
        if valid_indices:
            valid_rendered = [rendered[i] for i in valid_indices]
            scores = scorer.score(valid_rendered, gt_images)
            for idx, s in zip(valid_indices, scores):
                rewards[idx] = s

        return rewards

    reward_fn.__name__ = "python_execution_reward"
    return reward_fn
