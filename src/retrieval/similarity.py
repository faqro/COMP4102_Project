"""Vector normalization and similarity computation utilities."""

from __future__ import annotations

import numpy as np


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 1D or 2D array along the last axis."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / max(norm, eps)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms


def cosine_similarity(query_vec: np.ndarray, db_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector (D,) and database matrix (N, D)."""
    q = l2_normalize(query_vec)
    db = l2_normalize(db_vecs)
    return db @ q


def euclidean_similarity(query_vec: np.ndarray, db_vecs: np.ndarray) -> np.ndarray:
    """Compute negative Euclidean distance so higher scores mean closer matches."""
    q = np.asarray(query_vec, dtype=np.float32)
    db = np.asarray(db_vecs, dtype=np.float32)
    dists = np.linalg.norm(db - q[None, :], axis=1)
    return -dists


def compute_similarity(
    query_vec: np.ndarray, db_vecs: np.ndarray, metric: str = "cosine"
) -> np.ndarray:
    if metric == "cosine":
        return cosine_similarity(query_vec, db_vecs)
    if metric == "euclidean":
        return euclidean_similarity(query_vec, db_vecs)
    raise ValueError(f"Unsupported similarity metric: {metric}")
