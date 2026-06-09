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

Two operating modes
-------------------
* ``n_keyframes=None`` (default) — variable-N: every qualifying local minimum
  (plus endpoints) is returned, so the count is data-dependent.
* ``n_keyframes=k`` — matched budget: return EXACTLY k frames (endpoints plus
  the top-(k-2) lowest-flow interior frames), so it is directly comparable to
  the uniform/random baselines at the same K.

Efficiency
----------
The RAFT model is loaded once per process and shared across all instances
(see ``_get_raft``). The k-INDEPENDENT flow signal is memoised per trajectory
(see ``_signal_memo_get``), so a K-sweep that builds one instance per K computes
the flow ONCE per episode and only re-runs the cheap top-k selection per K.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin

from .base import KeyframeExtractor


# Process-wide caches (shared across every OpticalFlowExtractor instance).
_RAFT_CACHE: dict = {}                       # (device_str, pretrained) -> model
_SIGNAL_MEMO: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (fingerprint, signal)
_SIGNAL_MEMO_MAX = 256                        # bound; eviction is FIFO, affects speed only


class OpticalFlowExtractor(KeyframeExtractor):
    """Selects low-flow frames (motion pauses) as keyframes.

    Args:
        n_keyframes: If given, return EXACTLY this many frames (endpoints + the
                     top-(k-2) lowest-flow interior frames). If None (default),
                     return all qualifying local minima (variable-N, legacy).
        min_dist:    Minimum number of frames between any two keyframes. In
                     matched-budget mode this is a soft constraint, relaxed only
                     as far as needed to reach k.
        sigma:       Gaussian smoothing σ applied to the flow signal before
                     peak detection (frames).
        device:      Torch device string.  Defaults to "cuda" if available,
                     else "cpu".
        pretrained:  If True, load RAFT-Small weights pretrained on Sintel.
        batch_size:  Number of consecutive frame-pairs run through RAFT per
                     forward pass. Larger keeps the GPU busier; the result is
                     numerically identical to per-pair processing (RAFT has no
                     cross-sample normalisation). Affects speed/VRAM only.
    """

    def __init__(
        self,
        n_keyframes: Optional[int] = None,
        min_dist: int = 5,
        sigma: float = 2.0,
        device: Optional[str] = None,
        pretrained: bool = True,
        batch_size: int = 16,
    ) -> None:
        self._n = n_keyframes
        self._min_dist = min_dist
        self._sigma = sigma
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._pretrained = pretrained
        self._batch_size = max(1, int(batch_size))
        # Model is loaded lazily on first extract() via the shared cache.

    # ------------------------------------------------------------------
    # KeyframeExtractor interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"optical_flow_k{self._n}" if self._n is not None else "optical_flow"

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

        # k-INDEPENDENT signal: compute once per trajectory, reuse across K.
        cfg_key = ("optical_flow", str(self._device), bool(self._pretrained))
        flow_signal = _signal_memo_get(cfg_key, images, self._compute_flow_signal)

        if self._n is not None:
            return _select_topk(flow_signal, T, self._n, self._min_dist, self._sigma, pick="min")
        return _select_minima(flow_signal, T, self._min_dist, self._sigma)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_flow_signal(self, images: np.ndarray) -> np.ndarray:
        """Return per-frame mean flow magnitude array of length T.

        Flow at t=0 is defined as 0 (no predecessor frame). Consecutive pairs
        (f[t-1], f[t]) are run through RAFT in mini-batches of
        ``self._batch_size`` so the GPU stays saturated and there is a single
        GPU->CPU sync at the end, instead of one ``.item()`` per pair. RAFT
        has no cross-sample normalisation (the feature/context encoders use
        InstanceNorm, which is per-sample), so the batched result is
        numerically equivalent to the per-pair loop.
        """
        model = _get_raft(self._device, self._pretrained)

        T = images.shape[0]
        signal = np.zeros(T, dtype=np.float32)
        if T < 2:
            return signal

        # RAFT expects float32 tensors in [0, 255], shape (N, 3, H, W).
        # Images must be divisible by 8; pad once for the whole stack.
        H, W = images.shape[1], images.shape[2]
        H_pad = ((H + 7) // 8) * 8
        W_pad = ((W + 7) // 8) * 8

        frames = torch.from_numpy(np.ascontiguousarray(images)).float()  # (T, H, W, 3)
        frames = frames.permute(0, 3, 1, 2)                              # (T, 3, H, W)
        if H_pad != H or W_pad != W:
            frames = F.pad(frames, (0, W_pad - W, 0, H_pad - H))
        frames = frames.to(self._device)

        prev_all = frames[:-1]          # (T-1, 3, H_pad, W_pad)
        curr_all = frames[1:]           # (T-1, 3, H_pad, W_pad)
        n_pairs = T - 1
        bs = self._batch_size

        mags = torch.empty(n_pairs, device=self._device, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, n_pairs, bs):
                stop = min(start + bs, n_pairs)
                # raft_small returns a list of flow predictions; last = finest.
                flow_preds = model(prev_all[start:stop], curr_all[start:stop])
                flow = flow_preds[-1][:, :, :H, :W]          # (b, 2, H, W)
                mags[start:stop] = flow.norm(dim=1).mean(dim=(1, 2))

        # Single GPU->CPU transfer; pair j -> signal[j+1].
        signal[1:] = mags.cpu().numpy()
        return signal


# ------------------------------------------------------------------
# Shared model loading (once per process)
# ------------------------------------------------------------------

def _get_raft(device, pretrained: bool):
    """Return a RAFT-Small model, loaded once per (device, pretrained) per process."""
    key = (str(device), bool(pretrained))
    model = _RAFT_CACHE.get(key)
    if model is None:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        weights = Raft_Small_Weights.DEFAULT if pretrained else None
        model = raft_small(weights=weights).to(device)
        model.eval()
        _RAFT_CACHE[key] = model
    return model


# ------------------------------------------------------------------
# k-independent signal memoisation (compute once per trajectory)
# ------------------------------------------------------------------

def _fingerprint(a: np.ndarray) -> str:
    """Cheap, deterministic content fingerprint (strided sample → constant cost)."""
    flat = np.ascontiguousarray(a).reshape(-1)
    step = max(1, flat.size // 8192)
    return hashlib.sha1(flat[::step].tobytes()).hexdigest()


def _signal_memo_get(cfg_key: tuple, images: np.ndarray, compute_fn):
    """Return the k-independent signal for *images*, computing it at most once.

    Keyed on (cfg_key, id(images), shape, nbytes) for speed, with a content
    fingerprint stored alongside and re-verified on every hit so a reused
    Python id() (after an array is GC'd) can never return another episode's
    signal. The memo only affects performance: on any miss the signal is
    recomputed identically, so determinism is unaffected.
    """
    a = np.asarray(images)
    key = (cfg_key, id(images), a.shape, int(a.nbytes))
    fp = _fingerprint(a)

    hit = _SIGNAL_MEMO.get(key)
    if hit is not None and hit[0] == fp:
        _SIGNAL_MEMO.move_to_end(key)
        return hit[1]

    signal = compute_fn(a)
    _SIGNAL_MEMO[key] = (fp, signal)
    _SIGNAL_MEMO.move_to_end(key)
    while len(_SIGNAL_MEMO) > _SIGNAL_MEMO_MAX:
        _SIGNAL_MEMO.popitem(last=False)
    return signal


# ------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------

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


def _select_topk(
    signal: np.ndarray,
    T: int,
    k: int,
    min_dist: int,
    sigma: float,
    pick: str,
) -> np.ndarray:
    """Select EXACTLY k frames: endpoints + top-(k-2) interior by saliency.

    pick="min": salient = low signal (flow pauses); pick="max": high signal.
    Spacing: min_dist is enforced in a first pass, then relaxed (next-best
    saliency) only as far as needed to reach k. Ties are broken by ascending
    index via np.lexsort — fully deterministic, no RNG (seeds only ever affect
    the RandomExtractor).
    """
    if k >= T:
        return np.arange(T, dtype=int)
    if k <= 2:
        return np.array(sorted({0, T - 1}), dtype=int)

    smoothed = gaussian_filter1d(signal.astype(float), sigma=sigma)
    interior = np.arange(1, T - 1)
    # primary sort key = saliency (ascending), secondary = index (ascending)
    sal_key = smoothed[interior] if pick == "min" else -smoothed[interior]
    order = interior[np.lexsort((interior, sal_key))]

    need = k - 2
    selected = [0, T - 1]

    # Pass 1: take highest-saliency interior frames that respect min_dist.
    for idx in order:
        if len(selected) - 2 >= need:
            break
        if all(abs(int(idx) - s) >= min_dist for s in selected):
            selected.append(int(idx))

    # Pass 2: if spacing blocked too many, relax it to reach exactly k.
    if len(selected) - 2 < need:
        chosen = set(selected)
        for idx in order:
            if len(selected) - 2 >= need:
                break
            if int(idx) not in chosen:
                selected.append(int(idx))
                chosen.add(int(idx))

    return np.array(sorted(selected), dtype=int)


def _apply_min_dist(candidates: list[int], min_dist: int) -> list[int]:
    """Greedy filter: keep a candidate only if it is >= min_dist from the last kept."""
    kept: list[int] = []
    for idx in sorted(candidates):
        if not kept or idx - kept[-1] >= min_dist:
            kept.append(idx)
    return kept
