"""Experiment D: Gallery scalability and optional FAISS indexing comparison."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np

from src.data.dataset import ROxford5kDataset
from src.evaluation.benchmark import benchmark_retriever
from src.experiments.common import ALL_METHODS, ExperimentTracker, build_retriever, load_config
from src.experiments.tuning import load_best_configs

GALLERY_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def run_scalability(
    methods: Optional[List[str]] = None,
    dataset: Optional[ROxford5kDataset] = None,
    tracker: Optional[ExperimentTracker] = None,
    use_tuned: bool = True,
) -> Dict[str, List[Dict[str, float]]]:
    methods = methods or ALL_METHODS
    dataset = dataset or ROxford5kDataset()
    tracker = tracker or ExperimentTracker()
    best_configs = load_best_configs() if use_tuned else {}

    n_total = len(dataset.gallery_paths)
    rng = np.random.default_rng(42)
    shuffled_indices = rng.permutation(n_total)

    all_results: Dict[str, List[Dict[str, float]]] = {}

    for method in methods:
        cfg = copy.deepcopy(best_configs.get(method) or load_config(method))
        method_rows: List[Dict[str, float]] = []

        for frac in GALLERY_FRACTIONS:
            n_sub = max(10, int(round(n_total * frac)))
            sub_idx = np.sort(shuffled_indices[:n_sub])
            retriever = build_retriever(method, cfg)
            res = benchmark_retriever(retriever, dataset, gallery_indices=sub_idx)
            m = res["metrics"]
            method_rows.append(m)

            exp_id = tracker.log_run(
                experiment_type="scalability",
                method=method,
                metrics=m,
                cfg=cfg,
                split="all",
                gallery_fraction=frac,
                notes=f"Scalability test at {int(frac * 100)}% gallery ({n_sub} images)",
                write_markdown=(frac == 1.0),
            )
            print(
                f"[{exp_id}] Scalability {method:6s} | frac={frac:.2f} ({n_sub:4d}) | "
                f"idx={m['indexing_time_s']:.2f}s | lat={m['query_latency_ms']:.2f}ms | "
                f"mem={m['memory_mb']:.2f}MB | mAP_med={m['mAP_medium']:.4f}"
            )

        if method in ("resnet", "dinov2"):
            for faiss_type in ("flat", "ivf"):
                f_cfg = copy.deepcopy(cfg)
                f_cfg.setdefault("retrieval", {})["use_faiss"] = True
                f_cfg["retrieval"]["faiss_type"] = faiss_type
                retriever = build_retriever(method, f_cfg)
                res = benchmark_retriever(retriever, dataset)
                m = res["metrics"]
                exp_id = tracker.log_run(
                    experiment_type="faiss_comparison",
                    method=method,
                    metrics=m,
                    cfg=f_cfg,
                    split="all",
                    gallery_fraction=1.0,
                    notes=f"FAISS ({faiss_type}) indexing comparison for {method}",
                    write_markdown=True,
                )
                print(
                    f"[{exp_id}] FAISS ({faiss_type:4s}) {method:6s} | "
                    f"search={m['query_search_time_ms']:.3f}ms | mAP_med={m['mAP_medium']:.4f}"
                )

        all_results[method] = method_rows

    return all_results
