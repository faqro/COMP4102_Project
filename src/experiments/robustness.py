"""Experiment C: Query transformation robustness evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.data.dataset import ROxford5kDataset
from src.data.preprocessing import TRANSFORMATION_SPECS, transform_image
from src.evaluation.benchmark import benchmark_retriever
from src.experiments.common import ALL_METHODS, ExperimentTracker, build_retriever, load_config
from src.experiments.tuning import load_best_configs


def run_robustness(
    methods: Optional[List[str]] = None,
    dataset: Optional[ROxford5kDataset] = None,
    tracker: Optional[ExperimentTracker] = None,
    use_tuned: bool = True,
) -> Dict[str, List[Dict[str, float]]]:
    methods = methods or ALL_METHODS
    dataset = dataset or ROxford5kDataset()
    tracker = tracker or ExperimentTracker()
    best_configs = load_best_configs() if use_tuned else {}

    all_results: Dict[str, List[Dict[str, float]]] = {}

    for method in methods:
        cfg = best_configs.get(method) or load_config(method)
        retriever = build_retriever(method, cfg)

        gallery_paths = dataset.gallery_paths
        retriever.fit(gallery_paths)

        method_rows: List[Dict[str, float]] = []
        for t_type, level, param in TRANSFORMATION_SPECS:
            seed = int(cfg.get("seed", 42))

            def _tx(img, qid, tt=t_type, p=param, s=seed):
                return transform_image(img, tt, p, seed=s + qid)

            res = benchmark_retriever(
                retriever,
                dataset,
                query_ids=None,
                query_transform=_tx,
                skip_fit=True,
            )
            m = res["metrics"]
            method_rows.append(m)
            exp_id = tracker.log_run(
                experiment_type="robustness",
                method=method,
                metrics=m,
                cfg=cfg,
                split="all",
                transformation=t_type,
                transform_level=level,
                notes=f"Robustness test {t_type}={level} for {method}",
                write_markdown=(t_type == "none"),
            )
            print(
                f"[{exp_id}] Robustness {method:6s} | {t_type:10s} {level:8s} | "
                f"mAP_med: {m['mAP_medium']:.4f} | P@1: {m['P@1_medium']:.4f}"
            )
        all_results[method] = method_rows

    return all_results
