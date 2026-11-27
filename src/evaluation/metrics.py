"""Official ROxford5k mAP, Precision@K, and Recall@K evaluation metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def compute_ap(ranks: np.ndarray, nres: int) -> float:
    """Compute Average Precision following the official revisitop ROxford5k protocol."""
    if nres <= 0 or len(ranks) == 0:
        return 0.0

    ap = 0.0
    old_recall = 0.0
    old_precision = 1.0
    for j, r in enumerate(ranks):
        recall = (j + 1.0) / float(nres)
        precision = (j + 1.0) / float(r + 1.0)
        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
        old_recall = recall
        old_precision = precision
    return float(ap)


def _filter_ranking(
    ranking: np.ndarray,
    positives: np.ndarray,
    junk: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Remove junk indices from ranking and return (positive_positions, n_positives)."""
    if len(positives) == 0:
        return np.array([], dtype=np.int64), 0

    junk_set = set(int(x) for x in junk)
    pos_set = set(int(x) for x in positives)

    filtered = [int(idx) for idx in ranking if int(idx) not in junk_set]
    pos_positions = [pos_i for pos_i, idx in enumerate(filtered) if idx in pos_set]
    return np.array(pos_positions, dtype=np.int64), len(pos_set)


def evaluate_protocol(
    rankings: Sequence[np.ndarray],
    queries: Sequence[Any],
    protocol: str = "medium",
    ks: Iterable[int] = (1, 5, 10),
    valid_gallery_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate mAP, Precision@K, and Recall@K for easy, medium, or hard protocol."""
    aps: List[float] = []
    precisions: Dict[int, List[float]] = {k: [] for k in ks}
    recalls: Dict[int, List[float]] = {k: [] for k in ks}

    for ranking, q in zip(rankings, queries):
        if protocol == "easy":
            pos = q.easy
            junk = np.concatenate([q.junk, q.hard])
        elif protocol == "medium":
            pos = np.concatenate([q.easy, q.hard])
            junk = q.junk
        elif protocol == "hard":
            pos = q.hard
            junk = np.concatenate([q.easy, q.junk])
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        if valid_gallery_mask is not None:
            pos = pos[valid_gallery_mask[pos]]
            junk = junk[valid_gallery_mask[junk]]

        if len(pos) == 0:
            continue

        pos_positions, nres = _filter_ranking(ranking, pos, junk)
        aps.append(compute_ap(pos_positions, nres))

        for k in ks:
            hits = int(np.sum(pos_positions < k))
            precisions[k].append(hits / float(k))
            recalls[k].append(hits / float(nres))

    out: Dict[str, float] = {
        f"mAP_{protocol}": float(np.mean(aps)) if aps else 0.0,
        f"n_eval_queries_{protocol}": float(len(aps)),
    }
    for k in ks:
        out[f"P@{k}_{protocol}"] = float(np.mean(precisions[k])) if precisions[k] else 0.0
        out[f"R@{k}_{protocol}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
    return out


def evaluate_all_protocols(
    rankings: Sequence[np.ndarray],
    queries: Sequence[Any],
    ks: Iterable[int] = (1, 5, 10),
    valid_gallery_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for proto in ("easy", "medium", "hard"):
        metrics.update(
            evaluate_protocol(
                rankings,
                queries,
                protocol=proto,
                ks=ks,
                valid_gallery_mask=valid_gallery_mask,
            )
        )
    return metrics
