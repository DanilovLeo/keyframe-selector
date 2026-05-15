"""
attention.py

AttentionSaliencyExtractor: selects keyframes at local MAXIMA of frame-to-frame
[CLS]-token cosine distance computed by DINOv2-small.

Large cosine-distance steps correspond to semantic transitions — the visual
scene has changed meaningfully — making them natural keyframe candidates.

Model: vit_small_patch14_dinov2 (ViT-S/14, 384-dim) via timm.
       Downloads ~85 MB on first run, cached under ~/.cache/huggingface/hub.

Notes on model loading:
  - torch.hub (facebookresearch/dinov2) requires Python 3.10+ due to `X | Y`
    union-type annotations in the upstream repo.
  - transformers >= 4.47 requires torch.compiler (PyTorch >= 2.1) for DINOv2.
  - timm 1.x works with Python 3.9 and PyTorch 2.0, so it is used here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax

from .base import KeyframeExtractor


class AttentionSaliencyExtractor(KeyframeExtractor):
    """Selects frames where the visual scene changes most (DINOv2 [CLS] distance).

    Args:
        timm_model:  timm model name.  Defaults to "vit_small_patch14_dinov2"
                     (DINOv2-small, ViT-S/14, 384-dim).
        min_dist:    Minimum number of frames between any two keyframes.
        sigma:       Gaussian smoothing σ applied to the distance signal (frames).
        batch_size:  Number of frames processed per forward pass.
        device:      Torch device string.  Defaults to "cuda" if available, "cpu" otherwise.
    """

    def __init__(
        self,
        timm_model: str = "vit_small_patch14_dinov2",
        min_dist: int = 5,
        sigma: float = 2.0,
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> None:
        self._timm_model = timm_model
        self._min_dist   = min_dist
        self._sigma      = sigma
        self._batch_size = batch_size
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._processor, self._model = self._load_model()

    # ------------------------------------------------------------------
    # KeyframeExtractor interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "attention_dino"

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Select keyframes from a (T, H, W, 3) uint8 frame array.

        Returns:
            Sorted 1-D integer array of keyframe indices, always including
            0 and T-1.
        """
        images = np.asarray(trajectory)
        T = len(images)
        if T <= 2:
            return np.arange(T, dtype=int)

        features = self._extract_features(images)          # (T, D) float32, L2-normalised
        signal = _cosine_distance_signal(features)          # (T,) float32, d[0]=0
        return _select_maxima(signal, T, self._min_dist, self._sigma)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self):
        import timm  # noqa: PLC0415

        # num_classes=0 → forward() returns the pooled CLS embedding (B, D)
        # rather than class logits.
        model = timm.create_model(self._timm_model, pretrained=True, num_classes=0)
        model = model.to(self._device).eval()

        # Use the model's own recommended preprocessing (resize, crop, normalize)
        data_cfg   = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_cfg, is_training=False)
        return preprocess, model

    def _extract_features(self, images: np.ndarray) -> np.ndarray:
        """Return (T, D) float32 array of L2-normalised [CLS] embeddings."""
        all_cls: list[np.ndarray] = []

        for start in range(0, len(images), self._batch_size):
            batch_np = images[start : start + self._batch_size]
            pil_batch = [Image.fromarray(img).convert("RGB") for img in batch_np]
            tensors = torch.stack(
                [self._processor(img) for img in pil_batch]
            ).to(self._device)

            with torch.no_grad():
                # num_classes=0 → model(x) returns (B, D) CLS embedding
                cls = self._model(tensors)
                cls = F.normalize(cls, dim=-1)

            all_cls.append(cls.cpu().float().numpy())

        return np.concatenate(all_cls, axis=0)  # (T, D)


# ------------------------------------------------------------------
# Signal processing helpers
# ------------------------------------------------------------------

def _cosine_distance_signal(features: np.ndarray) -> np.ndarray:
    """Return per-frame cosine distance to the previous frame.

    Features must be L2-normalised; d[t] = 1 - dot(feat[t-1], feat[t]).
    d[0] is set to 0 (no predecessor).
    """
    T = len(features)
    signal = np.zeros(T, dtype=np.float32)
    # Vectorised dot products for consecutive pairs
    dots = np.einsum("td,td->t", features[:-1], features[1:])  # (T-1,)
    signal[1:] = 1.0 - dots
    return signal


def _select_maxima(
    signal: np.ndarray,
    T: int,
    min_dist: int,
    sigma: float,
) -> np.ndarray:
    """Pick local maxima of *signal* with min_dist spacing, always include 0 and T-1."""
    smoothed = gaussian_filter1d(signal.astype(float), sigma=sigma)

    order = max(1, min_dist // 2)
    (maxima_idx,) = argrelmax(smoothed, order=order)

    selected = _apply_min_dist(list(maxima_idx), min_dist)
    selected = sorted(set(selected) | {0, T - 1})
    return np.array(selected, dtype=int)


def _apply_min_dist(candidates: list, min_dist: int) -> list:
    """Greedy filter: keep a candidate only if >= min_dist from the last kept."""
    kept: list[int] = []
    for idx in sorted(candidates):
        if not kept or idx - kept[-1] >= min_dist:
            kept.append(idx)
    return kept
