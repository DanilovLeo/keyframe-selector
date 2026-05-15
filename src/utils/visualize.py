"""Visualization helpers for keyframe selection experiments."""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

_RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def _ensure_results_dir() -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def plot_frame_grid(
    images: np.ndarray,
    keyframe_indices: np.ndarray,
    title: str = "",
    max_frames: int = 16,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Thumbnail grid of selected keyframe images.

    Args:
        images:           (T, H, W, 3) uint8 frame array.
        keyframe_indices: 1-D integer array of selected frame indices.
        title:            Figure suptitle.
        max_frames:       Cap on how many thumbnails to show (leftmost subset).
        save_name:        Filename stem for saving; skipped if None.

    Returns:
        matplotlib Figure.
    """
    idx = keyframe_indices[:max_frames]
    n = len(idx)
    ncols = min(n, 8)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6))
    axes = np.array(axes).reshape(-1)  # always 1-D

    for ax_i, frame_i in enumerate(idx):
        axes[ax_i].imshow(images[frame_i])
        axes[ax_i].set_title(f"t={frame_i}", fontsize=7)
        axes[ax_i].axis("off")

    for ax_i in range(len(idx), len(axes)):
        axes[ax_i].axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    if save_name:
        out = _ensure_results_dir() / f"{save_name}_frame_grid.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")

    return fig


def plot_signal_with_keyframes(
    signal: np.ndarray,
    keyframe_indices: np.ndarray,
    ylabel: str = "signal",
    title: str = "",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Line plot of a 1-D per-frame signal with keyframe positions marked.

    Suitable for optical-flow magnitude, attention distance, or any
    scalar derived from the video.

    Args:
        signal:           1-D float array of length T.
        keyframe_indices: 1-D integer array of selected frame indices.
        ylabel:           Y-axis label.
        title:            Figure title.
        save_name:        Filename stem for saving; skipped if None.

    Returns:
        matplotlib Figure.
    """
    T = len(signal)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, signal, color="steelblue", linewidth=1.2, label=ylabel)
    ax.vlines(
        keyframe_indices,
        ymin=signal.min(),
        ymax=signal.max() * 1.05,
        colors="crimson",
        linewidth=0.8,
        alpha=0.7,
        label="keyframes",
    )
    ax.scatter(keyframe_indices, signal[keyframe_indices],
               color="crimson", zorder=5, s=30)
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()

    if save_name:
        out = _ensure_results_dir() / f"{save_name}_signal.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig
