"""
cache.py

Disk-backed cache for CLIP frame embeddings.

Key: (episode_index, model_id)  →  (T, D) float32 numpy array.
Files: <cache_dir>/clip_<model_id_slug>_ep<episode_index>.npy

A cache miss triggers CLIP inference; a cache hit loads the .npy file.
If the model changes, pass a different model_id and the old files are
ignored (not deleted — different slug).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class FrameEmbeddingCache:
    """Persistent per-episode CLIP frame embedding cache.

    Args:
        cache_dir:  Directory for .npy cache files.
        model_id:   String that uniquely identifies the model (e.g.
                    "ViT-L-14_openai").  Included in every filename so
                    switching models never silently serves stale data.
    """

    def __init__(self, cache_dir: str, model_id: str) -> None:
        self._dir = Path(cache_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        # Slugify: replace characters that are awkward in filenames
        self._slug = model_id.replace("/", "_").replace("-", "_").replace(" ", "_")

    def _path(self, episode_index: int) -> Path:
        return self._dir / f"clip_{self._slug}_ep{episode_index}.npy"

    def get(self, episode_index: int) -> Optional[np.ndarray]:
        """Return cached (T, D) array or None if not cached."""
        p = self._path(episode_index)
        return np.load(str(p)) if p.exists() else None

    def put(self, episode_index: int, embs: np.ndarray) -> None:
        """Save (T, D) float32 array to disk."""
        np.save(str(self._path(episode_index)), embs)

    def contains(self, episode_index: int) -> bool:
        return self._path(episode_index).exists()

    def __repr__(self) -> str:
        n = sum(1 for _ in self._dir.glob(f"clip_{self._slug}_ep*.npy"))
        return f"FrameEmbeddingCache(slug={self._slug!r}, cached={n}, dir={self._dir})"
