"""
bridge_loader.py

Episode-level loader for BridgeData v2 via lerobot's LeRobotDataset.

Typical usage
-------------
    from src.data.bridge_loader import BridgeDataLoader

    loader = BridgeDataLoader(root="~/.cache/lerobot")
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
from typing import Dict, List, Optional

import numpy as np


_DATASET_REPO = "lerobot/bridgedata_v2"


class BridgeDataLoader:
    """Episode-level accessor for BridgeData v2.

    Downloads parquet metadata on first construction; images are loaded
    per-episode on demand.  A task-index JSON cache is written next to the
    dataset root to avoid re-scanning on subsequent runs.

    Args:
        root:         Local cache directory (passed to LeRobotDataset).
                      Defaults to HuggingFace's default cache location.
        dataset_name: HuggingFace repo ID (default: "lerobot/bridgedata_v2").
    """

    def __init__(
        self,
        root: Optional[str] = None,
        dataset_name: str = _DATASET_REPO,
    ) -> None:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        self._root = Path(root).expanduser() if root else None
        kwargs: dict = {}
        if self._root is not None:
            kwargs["root"] = self._root
        self._ds = LeRobotDataset(dataset_name, **kwargs)
        self._task_to_episodes: Dict[str, List[int]] = _build_task_index(
            self._ds, self._root
        )

    # ------------------------------------------------------------------
    # Public API
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
        ep_idx_map = self._ds.episode_data_index
        start = int(ep_idx_map["from"][episode_index])
        end   = int(ep_idx_map["to"][episode_index])

        frames: List[np.ndarray] = []
        task_name = ""
        for flat_i in range(start, end):
            row = self._ds[flat_i]
            if not task_name:
                task_name = _extract_task(row, self._ds)
            frames.append(_extract_image(row))

        return {
            "images":    np.stack(frames, axis=0),
            "task_name": task_name,
            "n_frames":  len(frames),
        }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_task_index(ds, root: Optional[Path]) -> Dict[str, List[int]]:
    """Return {task_str: [episode_index, …]}, with a JSON cache on disk."""
    cache_path = (root / "bridge_task_index.json") if root is not None else None

    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        # JSON keys are strings; episode indices are ints
        return {k: [int(x) for x in v] for k, v in raw.items()}

    # Read just the two lightweight columns from the Arrow/parquet dataset.
    # For BridgeData v2 (~9M rows) this is a vectorised parquet scan — fast.
    hf = ds.hf_dataset
    task_col = _read_task_column(hf, ds)   # list[str], one per frame
    ep_col   = hf["episode_index"]          # list[int], one per frame

    task_to_eps: Dict[str, List[int]] = {}
    seen: set = set()
    for task, ep in zip(task_col, ep_col):
        task = str(task)
        key = (task, int(ep))
        if key not in seen:
            seen.add(key)
            task_to_eps.setdefault(task, []).append(int(ep))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(task_to_eps, f)

    return task_to_eps


def _read_task_column(hf, ds) -> list:
    """Extract the task string column, handling both lerobot API versions."""
    # lerobot >= 0.2: flat "task" string column
    if "task" in hf.column_names:
        return hf["task"]
    # lerobot 0.1.x: integer "task_index" + ds.meta.tasks lookup
    if "task_index" in hf.column_names and hasattr(ds, "meta") and hasattr(ds.meta, "tasks"):
        idx_to_name: dict = ds.meta.tasks
        return [idx_to_name.get(int(i), str(i)) for i in hf["task_index"]]
    # Fallback: annotation.task
    if "annotation.task" in hf.column_names:
        return hf["annotation.task"]
    raise RuntimeError(
        f"Cannot locate task column in dataset. Available columns: {hf.column_names}"
    )


def _extract_task(row: dict, ds) -> str:
    """Pull task string from a single frame dict."""
    if "task" in row:
        v = row["task"]
        return str(v) if not isinstance(v, str) else v
    if "task_index" in row and hasattr(ds, "meta") and hasattr(ds.meta, "tasks"):
        return ds.meta.tasks.get(int(row["task_index"]), "")
    if "annotation.task" in row:
        return str(row["annotation.task"])
    return ""


def _extract_image(row: dict) -> np.ndarray:
    """Return (H, W, 3) uint8 from a lerobot frame dict."""
    for key in ("observation.image", "observation.images.image_0", "image"):
        if key not in row:
            continue
        img = row[key]

        # torch Tensor
        if hasattr(img, "numpy"):
            arr = img.numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # (C, H, W) → (H, W, C)
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            return arr

        # PIL Image
        if hasattr(img, "convert"):
            return np.array(img.convert("RGB"), dtype=np.uint8)

        # numpy array
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            return img.astype(np.uint8)

    raise KeyError(
        f"No image key found in frame dict. Available keys: {sorted(row.keys())}"
    )
