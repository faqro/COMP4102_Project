"""Experiment A: Baseline evaluation across all four retrieval methods."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.data.dataset import ROxford5kDataset
from src.evaluation.benchmark import benchmark_retriever
from src.experiments.common import ALL_METHODS, ExperimentTracker, build_retriever, load_config


def run_baseline(
    methods: Optional[List[str]] = None,
    dataset: Optional[ROxford5kDataset] = None,
    tracker: Optional[ExperimentTracker] = None,
) -> Dict[str, Dict[str, float]]:
    methods = methods or ALL_METHODS
    dataset = dataset or ROxford5kDataset()
    tracker = tracker or ExperimentTracker()

    out_dir = Path("results/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    for method in methods:
        cfg = load_config(method)
        retriever = build_retriever(method, cfg)
        res = benchmark_retriever(retriever, dataset, query_ids=None)
        metrics = res["metrics"]
        results[method] = metrics

        exp_id = tracker.log_run(
            experiment_type="baseline",
            method=method,
            metrics=metrics,
            cfg=cfg,
            split="all",
            notes=f"Baseline configuration for {method}",
            objective=f"Measure baseline ROxford5k retrieval accuracy, latency, and memory for {method}.",
            hypothesis="Deep embeddings (ResNet-50, DINOv2) will outperform classical local features on global landmark retrieval.",
        )

        top10 = np.array([r[:10] for r in res["rankings"]], dtype=np.int64)
        top10_scores = np.array([s[:10] for s in res["scores"]], dtype=np.float32)
        np.savez(
            out_dir / f"{method}_baseline_rankings.npz",
            top10_indices=top10,
            top10_scores=top10_scores,
            query_ids=np.array(res["query_ids"], dtype=np.int64),
        )
        print(
            f"[{exp_id}] Baseline {method:6s} | "
            f"mAP(E/M/H): {metrics['mAP_easy']:.4f}/{metrics['mAP_medium']:.4f}/{metrics['mAP_hard']:.4f} | "
            f"P@1: {metrics['P@1_medium']:.4f} | Latency: {metrics['query_latency_ms']:.2f} ms | "
            f"Mem: {metrics['memory_mb']:.2f} MB"
        )

    return results
