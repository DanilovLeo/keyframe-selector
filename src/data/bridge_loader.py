"""
bridge_loader.py

Episode-level loader for BridgeData v2 using HuggingFace datasets directly.
No lerobot dependency — uses the same parquet files that lerobot would download,
but accesses them via the stable `datasets` library API.

Typical usage
-------------
    from src.data.bridge_loader import BridgeDataLoader

    loader = BridgeDataLoader(root="~/.cache/hf_bridge")
    tasks  = loader.list_tasks(min_demos=20)
    eps    = loader.list_episodes(tasks[0])
    ep     = loader.load_episode(eps[0])
    # ep["images"]    → np.ndarray (T, H, W, 3) uint8
    # ep["task_name"] → str
    # ep["n_frames"]  → int
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_DATASET_REPO = "lerobot/bridgedata_v2"


class BridgeDataLoader:
    """Episode-level accessor for BridgeData v2.

    Downloads parquet files on first construction (HuggingFace cache);
    images are decoded per-episode on demand. A task-index JSON cache is
    written next to the dataset root to avoid re-scanning on subsequent runs.

    Args:
        root:         Local HuggingFace cache directory. Defaults to the
                      HuggingFace default (~/.cache/huggingface/datasets).
        dataset_name: HuggingFace repo ID (default: "lerobot/bridgedata_v2").
    """

    def __init__(
        self,
        root: Optional[str] = None,
        dataset_name: str = _DATASET_REPO,
    ) -> None:
        from datasets import load_dataset  # noqa: PLC0415

        self._root = Path(root).expanduser() if root else None

        kwargs: dict = {"split": "train"}
        if self._root is not None:
            kwargs["cache_dir"] = str(self._root)

        print(f"Loading {dataset_name} (parquet scan, first run may take a few minutes)...")
        self._ds = load_dataset(dataset_name, **kwargs)

        self._task_to_episodes, self._episode_slices = _build_indexes(
            self._ds, self._root
        )

    # ------------------------------------------------------------------
    # Public API (unchanged from the lerobot-based version)
    # ------------------------------------------------------------------

    def list_tasks(self, min_demos: int = 20) -> List[str]:
        """Task strings that have at least *min_demos* episodes."""
        return sorted(
            t for t, eps in self._task_to_episodes.items() if len(eps) >= min_demos
        )

    def list_episodes(self, task: str) -> List[int]:
        """Episode indices that belong to *task*."""
        return list(self._task_to_episodes.get(task, []))

    def num_episodes_for(self, task: str) -> int:
        return len(self._task_to_episodes.get(task, []))

    def load_episode(self, episode_index: int) -> dict:
        """Load all frames for *episode_index*.

        Returns:
            dict with:
              "images"    — np.ndarray (T, H, W, 3) uint8
              "task_name" — str
              "n_frames"  — int
        """
        start, end = self._episode_slices[episode_index]
        episode_ds = self._ds.select(range(start, end))

        frames: List[np.ndarray] = []
        task_name = ""
        for i in range(len(episode_ds)):
            row = episode_ds[i]
            if not task_name:
                task_name = _extract_task(row)
            frames.append(_extract_image(row))

        return {
            "images":    np.stack(frames, axis=0),
            "task_name": task_name,
            "n_frames":  len(frames),
        }


# ------------------------------------------------------------------
# Index building
# ------------------------------------------------------------------

def _build_indexes(
    ds,
    root: Optional[Path],
) -> Tuple[Dict[str, List[int]], Dict[int, Tuple[int, int]]]:
    """Return (task_to_episodes, episode_slices) with a JSON cache on disk.

    task_to_episodes : {task_str: [episode_index, ...]}
    episode_slices   : {episode_index: (first_row, last_row_exclusive)}

    Uses Arrow columnar access — only reads episode_index and task columns,
    not image data, so the scan is fast (~9M integers).
    """
    cache_path = (root / "bridge_task_index.json") if root is not None else None

    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        task_to_eps   = {k: [int(x) for x in v] for k, v in raw["task_to_eps"].items()}
        episode_slices = {int(k): tuple(v) for k, v in raw["episode_slices"].items()}
        return task_to_eps, episode_slices

    print("Building task/episode index (one-time scan)...")
    ep_col   = ds["episode_index"]       # list[int], one per frame
    task_col = _read_task_column(ds)     # list[str], one per frame

    task_to_eps: Dict[str, List[int]] = {}
    episode_slices: Dict[int, List[int]] = {}  # {ep_idx: [start, end]}
    seen_ep_task: set = set()

    for row_i, (ep_idx, task) in enumerate(zip(ep_col, task_col)):
        ep_idx = int(ep_idx)
        task   = str(task)

        key = (ep_idx, task)
        if key not in seen_ep_task:
            seen_ep_task.add(key)
            task_to_eps.setdefault(task, []).append(ep_idx)

        if ep_idx not in episode_slices:
            episode_slices[ep_idx] = [row_i, row_i + 1]
        else:
            episode_slices[ep_idx][1] = row_i + 1

    episode_slices_final = {k: tuple(v) for k, v in episode_slices.items()}

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "task_to_eps":    task_to_eps,
                "episode_slices": {str(k): list(v) for k, v in episode_slices_final.items()},
            }, f)

    return task_to_eps, episode_slices_final


# ------------------------------------------------------------------
# Column helpers
# ------------------------------------------------------------------

def _read_task_column(ds) -> list:
    """Extract the task string column from the HuggingFace dataset."""
    cols = ds.column_names
    if "task" in cols:
        return ds["task"]
    if "annotation.task" in cols:
        return ds["annotation.task"]
    if "task_index" in cols:
        # Integer task IDs with no lookup table — convert to strings as-is.
        return [str(x) for x in ds["task_index"]]
    raise RuntimeError(
        f"Cannot locate task column. Available columns: {cols}"
    )


def _extract_task(row: dict) -> str:
    """Pull task string from a single frame dict."""
    for key in ("task", "annotation.task"):
        if key in row:
            v = row[key]
            return str(v) if not isinstance(v, str) else v
    if "task_index" in row:
        return str(row["task_index"])
    return ""


def _extract_image(row: dict) -> np.ndarray:
    """Return (H, W, 3) uint8 from a frame dict.

    Handles PIL Images, torch Tensors, and numpy arrays.
    Tries common lerobot image column names in order.
    """
    for key in ("observation.image", "observation.images.image_0", "image"):
        if key not in row:
            continue
        img = row[key]

        # PIL Image (most common when loaded via HuggingFace datasets)
        if hasattr(img, "convert"):
            return np.array(img.convert("RGB"), dtype=np.uint8)

        # torch Tensor
        if hasattr(img, "numpy"):
            arr = img.numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):   # (C, H, W) → (H, W, C)
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            return arr

        # numpy array
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            return img.astype(np.uint8)

    raise KeyError(
        f"No image key found in frame dict. Available keys: {sorted(row.keys())}"
    )
