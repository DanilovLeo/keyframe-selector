"""
clip_similarity.py

CLIP text-image cosine similarity metric.

Measures how semantically aligned keyframe-based demo embeddings are with
the task instruction.  Computed separately from retrieval accuracy and
provides a complementary signal: retrieval measures inter-task discrimination,
CLIP similarity measures absolute semantic alignment.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as Fn


def embed_text(
    instruction: str,
    model,
    tokenizer,
    device,
) -> np.ndarray:
    """Encode a task instruction string with the CLIP text encoder.

    Returns:
        (D,) float32 L2-normalised text embedding.
    """
    tokens = tokenizer([instruction]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)              # (1, D)
        emb = Fn.normalize(emb.float(), dim=-1)
    return emb.cpu().numpy()[0]                      # (D,)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


def mean_clip_similarity(
    demo_embeddings: List[np.ndarray],
    task_instruction: str,
    model,
    tokenizer,
    device,
) -> float:
    """Mean CLIP cosine similarity between demos and their task instruction.

    Args:
        demo_embeddings:  List of (D,) L2-normalised demo embeddings.
        task_instruction: The language instruction for these demos.

    Returns:
        Mean cosine similarity across all provided demo embeddings.
    """
    text_emb = embed_text(task_instruction, model, tokenizer, device)
    sims = [cosine_sim(dem, text_emb) for dem in demo_embeddings]
    return float(np.mean(sims))
