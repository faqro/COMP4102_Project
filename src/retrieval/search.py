"""Batch query search utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.retrieval.index import ImageRetriever


def run_retrieval(
    retriever: ImageRetriever,
    query_images: Sequence[Any],
    top_k: int | None = None,
) -> Dict[str, Any]:
    """Run retrieval for a sequence of queries and collect rankings and timings."""
    n_gallery = len(retriever.get_representations())
    k = n_gallery if top_k is None else min(top_k, n_gallery)

    ranked_indices: List[np.ndarray] = []
    ranked_scores: List[np.ndarray] = []
    query_feature_times: List[float] = []
    query_search_times: List[float] = []

    for q_img in query_images:
        results: List[Tuple[int, float]] = retriever.retrieve(q_img, top_k=k)
        idxs = np.array([r[0] for r in results], dtype=np.int64)
        scores = np.array([r[1] for r in results], dtype=np.float32)
        ranked_indices.append(idxs)
        ranked_scores.append(scores)
        query_feature_times.append(retriever.last_query_feature_time_s)
        query_search_times.append(retriever.last_query_search_time_s)

    return {
        "ranked_indices": ranked_indices,
        "ranked_scores": ranked_scores,
        "query_feature_times_s": query_feature_times,
        "query_search_times_s": query_search_times,
    }
