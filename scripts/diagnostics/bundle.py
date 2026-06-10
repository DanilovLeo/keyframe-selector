"""
bundle.py

Shared, numpy-only reader for the analysis bundle produced by
`scripts/export_eval_bundle.py`.  Every Tier-1 diagnostic imports this; none of
them touch torch, a GPU, the dataset, or any model — they operate purely on the
exported per-frame CLIP embeddings and keyframe indices.

The pooling and retrieval helpers are deliberately copied (not imported) from
`src/evaluation/retrieval.py` so this module stays dependency-light (importing
retrieval.py would drag in torch).  They are byte-for-byte equivalent; keep them
in sync if the pinned protocol changes.

Bundle layout (see export_eval_bundle.py):
  frame_embeddings.npz   keys "ep{idx}" -> (T, D) float32 L2-normalised
  bundle_meta.json       config + per-episode metadata + label list
  keyframes.jsonl        {"episode_index","label","indices"} per row (optional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Pooling + retrieval (copied from src/evaluation/retrieval.py for parity)
# --------------------------------------------------------------------------- #
def pool_demo_embedding(frame_embs: np.ndarray, keyframe_indices) -> np.ndarray:
    """Mean-pool selected frame embeddings and L2-normalise -> (D,)."""
    sel = frame_embs[np.asarray(keyframe_indices, dtype=int)]
    pooled = sel.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    return pooled / max(norm, 1e-8)


def retrieval_accuracy(
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    top_k: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """Multi-class top-k retrieval accuracy (scalar means)."""
    correct = per_query_correct(gallery_embs, gallery_labels,
                                query_embs, query_labels, top_k=top_k)
    return {f"top_{k}": float(correct[k].mean()) for k in top_k}


def per_query_correct(
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    top_k: Tuple[int, ...] = (1, 5),
) -> Dict[int, np.ndarray]:
    """Per-query boolean correctness for each k (needed for bootstrap/permutation).

    Returns {k: bool array of shape (N_q,)}.
    """
    sim = query_embs @ gallery_embs.T               # (N_q, N_g)
    ranked = np.argsort(-sim, axis=1)
    out: Dict[int, np.ndarray] = {}
    for k in top_k:
        k_eff = min(k, len(gallery_labels))
        top_labels = gallery_labels[ranked[:, :k_eff]]
        out[k] = (top_labels == query_labels[:, None]).any(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
class Bundle:
    """In-memory view over an exported analysis bundle."""

    def __init__(self, bundle_dir: str) -> None:
        d = Path(bundle_dir)
        with open(d / "bundle_meta.json") as f:
            self.meta = json.load(f)
        self.config: dict = self.meta.get("config", {})
        self.episodes: List[dict] = self.meta["episodes"]
        self.labels: List[str] = self.meta.get("labels", [])

        # Lazy-loaded npz: arrays are read per key on access.
        self._npz = np.load(d / "frame_embeddings.npz")

        # Keyframe indices, if exported.
        self._kf: Dict[Tuple[int, str], List[int]] = {}
        kf_path = d / "keyframes.jsonl"
        if kf_path.exists():
            with open(kf_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    self._kf[(int(r["episode_index"]), r["label"])] = r["indices"]

        # Convenience index maps.
        self.ep_split = {e["episode_index"]: e["split"] for e in self.episodes}
        self.ep_task = {e["episode_index"]: int(e["task_id"]) for e in self.episodes}
        self.ep_T = {e["episode_index"]: int(e["T"]) for e in self.episodes}
        self.ep_task_name = {e["episode_index"]: e["task"] for e in self.episodes}

    # -- episode access ----------------------------------------------------- #
    @property
    def episode_indices(self) -> List[int]:
        return [e["episode_index"] for e in self.episodes]

    def frames(self, ep: int) -> np.ndarray:
        """(T, D) float32 L2-normalised per-frame embeddings."""
        return np.asarray(self._npz[f"ep{ep}"], dtype=np.float32)

    def episode_mean(self, ep: int) -> np.ndarray:
        """Full-episode mean embedding, L2-normalised -> (D,)."""
        E = self.frames(ep)
        m = E.mean(axis=0)
        return m / max(float(np.linalg.norm(m)), 1e-8)

    def indices(self, ep: int, label: str) -> List[int]:
        return self._kf[(ep, label)]

    def has_indices(self) -> bool:
        return bool(self._kf)

    # -- splits ------------------------------------------------------------- #
    def gallery_eps(self) -> List[int]:
        return [e["episode_index"] for e in self.episodes if e["split"] == "gallery"]

    def query_eps(self) -> List[int]:
        return [e["episode_index"] for e in self.episodes if e["split"] == "query"]

    # -- demo embeddings ---------------------------------------------------- #
    def demo_embeddings(self, select_fn) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build pooled (gallery, query) demo embeddings + labels.

        Args:
            select_fn: callable (ep_index, T, frame_embs) -> 1-D index array.
                       Lets a diagnostic define an arbitrary selection rule.

        Returns:
            (g_embs, g_labels, q_embs, q_labels)
        """
        g_e, g_l, q_e, q_l = [], [], [], []
        for ep in self.episode_indices:
            E = self.frames(ep)
            idx = np.asarray(select_fn(ep, E.shape[0], E), dtype=int)
            dem = pool_demo_embedding(E, idx)
            if self.ep_split[ep] == "gallery":
                g_e.append(dem); g_l.append(self.ep_task[ep])
            else:
                q_e.append(dem); q_l.append(self.ep_task[ep])
        return (np.stack(g_e), np.array(g_l), np.stack(q_e), np.array(q_l))

    def demo_embeddings_for_label(self, label: str):
        """(gallery, query) demo embeddings using an exported extractor's indices."""
        return self.demo_embeddings(lambda ep, T, E: self.indices(ep, label))
