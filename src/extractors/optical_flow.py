"""
optical_flow.py

OpticalFlowExtractor: selects keyframes at local MINIMA of per-frame optical
flow magnitude (low-flow frames = motion pauses = semantically salient moments).

Model: RAFT-Small from torchvision.models.optical_flow, pretrained on
Sintel + Flying Chairs. Runs on GPU when available, CPU otherwise.

Flow is computed between consecutive frames f[t-1] → f[t] using RAFT-Small.
The resulting per-frame speed signal (mean L2 of the flow field) is Gaussian-
smoothed and local minima are selected subject to a minimum inter-frame
distance guard.  Frame 0 and T-1 are always included.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin

from .base import KeyframeExtractor


class OpticalFlowExtractor(KeyframeExtractor):
    """Selects low-flow frames (motion pauses) as keyframes.

    Args:
        min_dist:   Minimum number of frames between any two keyframes.
        sigma:      Gaussian smoothing σ applied to the flow signal before
                    peak detection (frames).
        device:     Torch device string.  Defaults to "cuda" if available,
                    else "cpu".
        pretrained: If True, load RAFT-Small weights pretrained on Sintel.
    """

    def __init__(
        self,
        min_dist: int = 5,
        sigma: float = 2.0,
        device: Optional[str] = None,
        pretrained: bool = True,
    ) -> None:
        self._min_dist = min_dist
        self._sigma = sigma
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = self._load_model(pretrained)

    # ------------------------------------------------------------------
    # KeyframeExtractor interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "optical_flow"

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

        flow_signal = self._compute_flow_signal(images)  # length T
        return _select_minima(flow_signal, T, self._min_dist, self._sigma)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self, pretrained: bool):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        weights = Raft_Small_Weights.DEFAULT if pretrained else None
        model = raft_small(weights=weights).to(self._device)
        model.eval()
        return model

    def _compute_flow_signal(self, images: np.ndarray) -> np.ndarray:
        """Return per-frame mean flow magnitude array of length T.

        Flow at t=0 is defined as 0 (no predecessor frame).
        """
        T = images.shape[0]
        signal = np.zeros(T, dtype=np.float32)

        # RAFT expects float32 tensors in [0, 255], shape (N, 3, H, W).
        # Images must be divisible by 8; pad if necessary.
        H, W = images.shape[1], images.shape[2]
        H_pad = ((H + 7) // 8) * 8
        W_pad = ((W + 7) // 8) * 8

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(img).float()          # (H, W, 3)
            t = t.permute(2, 0, 1).unsqueeze(0)        # (1, 3, H, W)
            if H_pad != H or W_pad != W:
                t = F.pad(t, (0, W_pad - W, 0, H_pad - H))
            return t.to(self._device)

        with torch.no_grad():
            prev = to_tensor(images[0])
            for t in range(1, T):
                curr = to_tensor(images[t])
                # raft_small returns a list of flow predictions; last = finest
                flow_preds = self._model(prev, curr)
                flow = flow_preds[-1]  # (1, 2, H_pad, W_pad)
                # Crop back to original resolution before computing magnitude
                flow = flow[:, :, :H, :W]
                mag = flow.norm(dim=1).mean().item()  # scalar
                signal[t] = mag
                prev = curr

        return signal


def _select_minima(
    signal: np.ndarray,
    T: int,
    min_dist: int,
    sigma: float,
) -> np.ndarray:
    """Pick local minima of *signal* with min_dist spacing, always include 0 and T-1."""
    smoothed = gaussian_filter1d(signal.astype(float), sigma=sigma)

    # argrelmin uses an order window (half-width = min_dist // 2, at least 1)
    order = max(1, min_dist // 2)
    (minima_idx,) = argrelmin(smoothed, order=order)

    # Enforce min_dist between any two selected frames
    selected = _apply_min_dist(list(minima_idx), min_dist)

    # Always include endpoints
    endpoints = {0, T - 1}
    selected = sorted(set(selected) | endpoints)
    return np.array(selected, dtype=int)


def _apply_min_dist(candidates: list[int], min_dist: int) -> list[int]:
    """Greedy filter: keep a candidate only if it is >= min_dist from the last kept."""
    kept: list[int] = []
    for idx in sorted(candidates):
        if not kept or idx - kept[-1] >= min_dist:
            kept.append(idx)
    return kept
