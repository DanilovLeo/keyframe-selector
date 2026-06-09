"""
retrieval.py

Task-retrieval evaluation utilities.

Protocol
--------
For each task with >= min_demos episodes:
  - Shuffle episodes with seed=42 and split 80/20 (gallery / query).
  - Gallery and query sets are pooled across ALL tasks into one multi-class
    retrieval problem.
  - For each query episode, rank all gallery episodes by cosine similarity.
  - Top-1 accuracy: is the nearest gallery episode from the correct task?
  - Top-5 accuracy: does at least one of the top-5 gallery episodes come from
    the correct task?

Demo embedding
--------------
  All T frames of an episode are embedded with CLIP ViT-L/14 in one pass
  (precomputed once per episode, reused across extractors).  For a given
  extractor, the keyframe subset is pooled:

      per-frame embeds (L2-normalised) → index by keyframe_indices
      → mean pool → L2-normalise → (D,) demo embedding
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image


# ------------------------------------------------------------------
# CLIP loading
# ------------------------------------------------------------------

def load_clip(
    model_name: str = "ViT-L-14-quickgelu",
    pretrained: str = "openai",
    device: Optional[str] = None,
):
    """Load CLIP model + preprocess + tokenizer.

    Use the "-quickgelu" variant with the openai weights: those weights were
    trained with QuickGELU, and the plain "ViT-L-14" builds a standard-GELU
    model, so pairing it with openai weights silently degrades every embedding.
    Experiment scripts override these from configs/models.yaml.

    Returns:
        (model, preprocess, tokenizer, device)
    """
    import open_clip  # noqa: PLC0415

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(dev).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, dev


# ------------------------------------------------------------------
# Embedding helpers
# ------------------------------------------------------------------

def embed_all_frames(
    images: np.ndarray,
    model,
    preprocess,
    device,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed every frame with CLIP and return (T, D) float32 L2-normalised array.

    Calling this once per episode and then pooling with pool_demo_embedding()
    is far cheaper than re-running CLIP for every extractor/K combination.
    """
    T = len(images)
    all_embs: List[np.ndarray] = []

    for start in range(0, T, batch_size):
        batch_np = images[start : start + batch_size]
        tensors = torch.stack(
            [preprocess(Image.fromarray(img).convert("RGB")) for img in batch_np]
        ).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensors)           # (B, D)
            emb = Fn.normalize(emb.float(), dim=-1)
        all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)            # (T, D)


def pool_demo_embedding(
    frame_embs: np.ndarray,
    keyframe_indices: np.ndarray,
) -> np.ndarray:
    """Mean-pool selected frame embeddings and L2-normalise.

    Args:
        frame_embs:       (T, D) L2-normalised per-frame embeddings.
        keyframe_indices: 1-D int array of selected frame indices.

    Returns:
        (D,) float32 L2-normalised demo embedding.
    """
    selected = frame_embs[keyframe_indices]    # (K, D)
    pooled   = selected.mean(axis=0)           # (D,)
    norm     = float(np.linalg.norm(pooled))
    return pooled / max(norm, 1e-8)


# ------------------------------------------------------------------
# Gallery / query split
# ------------------------------------------------------------------

def gallery_query_split(
    episode_indices: List[int],
    gallery_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Return (gallery_eps, query_eps) with a reproducible 80/20 split.

    At least one episode ends up in each set regardless of set size.
    """
    arr = np.array(episode_indices, dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    n_gallery = max(1, int(gallery_frac * len(arr)))
    n_gallery = min(n_gallery, len(arr) - 1)   # ensure at least one query
    return arr[:n_gallery].tolist(), arr[n_gallery:].tolist()


# ------------------------------------------------------------------
# Retrieval metric
# ------------------------------------------------------------------

def retrieval_accuracy(
    gallery_embs:   np.ndarray,
    gallery_labels: np.ndarray,
    query_embs:     np.ndarray,
    query_labels:   np.ndarray,
    top_k: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """Multi-class top-k retrieval accuracy.

    Args:
        gallery_embs:   (N_g, D) L2-normalised embeddings.
        gallery_labels: (N_g,)   integer task IDs.
        query_embs:     (N_q, D) L2-normalised embeddings.
        query_labels:   (N_q,)   integer task IDs.
        top_k:          k values to evaluate.

    Returns:
        Dict like {"top_1": 0.73, "top_5": 0.91}.
    """
    # Cosine similarity: both sides are already L2-normalised
    sim = query_embs @ gallery_embs.T          # (N_q, N_g)

    out: Dict[str, float] = {}
    for k in top_k:
        k_eff = min(k, len(gallery_labels))
        ranked = np.argsort(-sim, axis=1)[:, :k_eff]          # (N_q, k)
        top_labels = gallery_labels[ranked]                    # (N_q, k)
        correct = (top_labels == query_labels[:, None]).any(axis=1)
        out[f"top_{k}"] = float(correct.mean())
    return out
