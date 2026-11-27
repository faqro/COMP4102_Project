"""End-to-end benchmark runner measuring accuracy, latency, and memory."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from src.evaluation.metrics import evaluate_all_protocols
from src.retrieval.index import ImageRetriever
from src.retrieval.search import run_retrieval


def benchmark_retriever(
    retriever: ImageRetriever,
    dataset,
    query_ids: Optional[Sequence[int]] = None,
    gallery_indices: Optional[Sequence[int]] = None,
    query_transform: Optional[Callable[[Image.Image, int], Image.Image]] = None,
    skip_fit: bool = False,
) -> Dict[str, Any]:
    """Fit (unless skip_fit) and evaluate a retriever on ROxford5k."""
    if gallery_indices is None:
        gallery_idx_arr = np.arange(len(dataset.gallery_paths), dtype=np.int64)
    else:
        gallery_idx_arr = np.array(list(gallery_indices), dtype=np.int64)

    if not skip_fit:
        gallery_paths = [dataset.gallery_paths[int(i)] for i in gallery_idx_arr]
        retriever.fit(gallery_paths)

    if query_ids is None:
        eval_qids = list(range(len(dataset.queries)))
    else:
        eval_qids = list(query_ids)

    query_images: List[Image.Image] = []
    eval_queries = []
    for qid in eval_qids:
        q_img = dataset.get_query_image(qid)
        if query_transform is not None:
            q_img = query_transform(q_img, qid)
        query_images.append(q_img)
        eval_queries.append(dataset.queries[qid])

    search_out = run_retrieval(retriever, query_images, top_k=len(gallery_idx_arr))

    mapped_rankings = [gallery_idx_arr[r] for r in search_out["ranked_indices"]]

    valid_mask = np.zeros(len(dataset.gallery_paths), dtype=bool)
    valid_mask[gallery_idx_arr] = True

    metrics = evaluate_all_protocols(
        mapped_rankings,
        eval_queries,
        ks=(1, 5, 10),
        valid_gallery_mask=valid_mask,
    )

    q_feat_ms = float(np.mean(search_out["query_feature_times_s"]) * 1000.0)
    q_search_ms = float(np.mean(search_out["query_search_times_s"]) * 1000.0)
    metrics.update(
        {
            "feature_extraction_time_s": float(retriever.feature_extraction_time_s),
            "indexing_time_s": float(retriever.indexing_time_s),
            "query_feature_time_ms": q_feat_ms,
            "query_search_time_ms": q_search_ms,
            "query_latency_ms": q_feat_ms + q_search_ms,
            "memory_mb": float(retriever.memory_bytes()) / (1024.0 * 1024.0),
            "n_gallery": int(len(gallery_idx_arr)),
            "n_queries": int(len(eval_qids)),
        }
    )

    return {
        "metrics": metrics,
        "rankings": mapped_rankings,
        "scores": search_out["ranked_scores"],
        "query_ids": eval_qids,
    }
