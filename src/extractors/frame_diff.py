"""
frame_diff.py

FrameDiffExtractor: selects keyframes at frames of largest pixel-level change —
the mean absolute difference (MAD) between consecutive frames. Pure numpy: no
model download, no GPU, no torch. This is the cheapest motion proxy in the
suite and a useful low-complexity counterpart to the RAFT optical-flow and
DINOv2 attention extractors.

Large MAD steps mean the raw pixels changed a lot between two consecutive
frames (fast motion / scene transition), making them natural keyframe
candidates. This mirrors the attention extractor's "scene change" criterion
(pick local MAXIMA), but on raw pixels rather than CLS embeddings.

Two operating modes
-------------------
* ``n_keyframes=None`` (default) — variable-N: every qualifying local maximum
  (plus endpoints) is returned, so the count is data-dependent. This matches
  the attention extractor's variable-N default.
* ``n_keyframes=k`` — matched budget: return EXACTLY k frames (endpoints plus
  the top-(k-2) largest-change interior frames), directly comparable to the
  uniform/random baselines (and the other heuristics) at the same K.

Efficiency
----------
The k-INDEPENDENT MAD signal is memoised per trajectory (see
``_signal_memo_get``), so a K-sweep that builds one instance per K computes the
diff ONCE per episode and only re-runs the cheap top-k selection per K. This
mirrors the optical_flow / attention extractors so the three heuristics share
one structure.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax

from .base import KeyframeExtractor


# Process-wide cache (shared across every FrameDiffExtractor instance).
_SIGNAL_MEMO: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (fingerprint, signal)
_SIGNAL_MEMO_MAX = 256                        # bound; eviction is FIFO, affects speed only


class FrameDiffExtractor(KeyframeExtractor):
    """Selects high-change frames (pixel MAD spikes) as keyframes.

    Args:
        n_keyframes: If given, return EXACTLY this many frames (endpoints + the
                     top-(k-2) largest-change interior frames). If None
                     (default), return all qualifying local maxima (variable-N).
        min_dist:    Minimum number of frames between any two keyframes. In
                     matched-budget mode this is a soft constraint, relaxed only
                     as far as needed to reach k.
        sigma:       Gaussian smoothing σ applied to the MAD signal before peak
                     detection (frames).
    """

    def __init__(
        self,
        n_keyframes: Optional[int] = None,
        min_dist: int = 5,
        sigma: float = 2.0,
    ) -> None:
        self._n = n_keyframes
        self._min_dist = min_dist
        self._sigma = sigma

    # ------------------------------------------------------------------
    # KeyframeExtractor interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"frame_diff_k{self._n}" if self._n is not None else "frame_diff"

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
        cfg_key = ("frame_diff",)
        diff_signal = _signal_memo_get(cfg_key, images, _compute_diff_signal)

        if self._n is not None:
            return _select_topk(diff_signal, T, self._n, self._min_dist, self._sigma, pick="max")
        return _select_maxima(diff_signal, T, self._min_dist, self._sigma)


# ------------------------------------------------------------------
# Signal computation (pure numpy)
# ------------------------------------------------------------------

def _compute_diff_signal(images: np.ndarray) -> np.ndarray:
    """Return per-frame mean absolute difference array of length T.

    diff[t] = mean(|images[t] - images[t-1]|) over every pixel/channel.
    diff[0] is defined as 0 (no predecessor frame). Differences are taken in
    int16 to avoid uint8 wraparound (255 - 0 must read as 255, not -1).
    """
    T = images.shape[0]
    signal = np.zeros(T, dtype=np.float32)
    if T < 2:
        return signal
    cur  = images[1:].astype(np.int16)
    prev = images[:-1].astype(np.int16)
    axes = tuple(range(1, images.ndim))          # average over all non-time axes
    signal[1:] = np.abs(cur - prev).mean(axis=axes).astype(np.float32)
    return signal


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


def _select_topk(
    signal: np.ndarray,
    T: int,
    k: int,
    min_dist: int,
    sigma: float,
    pick: str,
) -> np.ndarray:
    """Select EXACTLY k frames: endpoints + top-(k-2) interior by saliency.

    pick="max": salient = high signal (large pixel change); pick="min": low.
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


def _apply_min_dist(candidates: list, min_dist: int) -> list:
    """Greedy filter: keep a candidate only if >= min_dist from the last kept."""
    kept: list[int] = []
    for idx in sorted(candidates):
        if not kept or idx - kept[-1] >= min_dist:
            kept.append(idx)
    return kept
