"""Visualization helpers for trajectory and keyframe analysis."""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

_RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def _ensure_results_dir() -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def plot_trajectory_with_keyframes(
    ee_pos: np.ndarray,
    keyframe_indices: np.ndarray,
    title: str = "",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """3-panel plot of XYZ EE position with keyframe markers overlaid.

    Args:
        ee_pos:           EE position array of shape (T, 3).
        keyframe_indices: 1-D integer array of keyframe frame indices.
        title:            Figure suptitle. Also used as default save filename.
        save_name:        Override filename (without extension). If None,
                          derived from title or skipped if title is empty.

    Returns:
        The matplotlib Figure object for inline display in notebooks.
    """
    T = len(ee_pos)
    t = np.arange(T)
    labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    if title:
        fig.suptitle(title, fontsize=13)

    for i, ax in enumerate(axes):
        ax.plot(t, ee_pos[:, i], color="steelblue", linewidth=1.2, label="EE pos")
        ax.scatter(
            keyframe_indices,
            ee_pos[keyframe_indices, i],
            color="crimson",
            zorder=5,
            s=40,
            label="keyframes" if i == 0 else None,
        )
        ax.set_ylabel(f"{labels[i]} (m)", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    fname = save_name or (title.replace(" ", "_").lower() if title else None)
    if fname:
        out = _ensure_results_dir() / f"{fname}_trajectory.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_velocity_profile(
    ee_vel: np.ndarray,
    keyframe_indices: np.ndarray,
    title: str = "",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Plot EE speed profile with keyframe positions marked.

    Args:
        ee_vel:           EE velocity array of shape (T, 3).
        keyframe_indices: 1-D integer array of keyframe frame indices.
        title:            Figure title.
        save_name:        Override save filename (without extension).

    Returns:
        The matplotlib Figure object.
    """
    T = len(ee_vel)
    t = np.arange(T)
    speed = np.linalg.norm(ee_vel, axis=1)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, speed, color="steelblue", linewidth=1.2, label="speed")
    ax.vlines(
        keyframe_indices,
        ymin=0,
        ymax=speed.max() * 1.05,
        colors="crimson",
        linewidth=0.8,
        alpha=0.7,
        label="keyframes",
    )
    ax.scatter(
        keyframe_indices,
        speed[keyframe_indices],
        color="crimson",
        zorder=5,
        s=30,
    )
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylabel("Speed (m/frame)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()

    fname = save_name or (title.replace(" ", "_").lower() if title else None)
    if fname:
        out = _ensure_results_dir() / f"{fname}_velocity.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig
