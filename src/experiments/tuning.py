"""Experiment B: Coarse-to-fine parameter tuning on validation split + held-out test evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.dataset import ROxford5kDataset
from src.evaluation.benchmark import benchmark_retriever
from src.experiments.common import ALL_METHODS, ExperimentTracker, build_retriever, load_config


def _save_best_configs(best_configs: Dict[str, Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2)


def load_best_configs(path: str | Path = "results/processed/best_configs.json") -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _tune_sift(dataset: ROxford5kDataset, tracker: ExperimentTracker) -> Dict[str, Any]:
    base = load_config("sift")
    best_cfg = copy.deepcopy(base)
    best_score = -1.0

    best_kp = base["features"]["n_keypoints"]
    best_raw_desc = None
    best_feat_time = 0.0
    for kp in [100, 250, 500, 1000]:
        cfg = copy.deepcopy(base)
        cfg["features"]["n_keypoints"] = kp
        ret = build_retriever("sift", cfg)
        res = benchmark_retriever(ret, dataset, query_ids=dataset.val_ids)
        m = res["metrics"]
        tracker.log_run(
            "tuning",
            "sift",
            m,
            cfg,
            split="val",
            notes=f"SIFT sweep n_keypoints={kp}, vocab_size={cfg['bovw']['vocab_size']}",
            write_markdown=False,
        )
        if m["mAP_medium"] > best_score:
            best_score = m["mAP_medium"]
            best_kp = kp
            best_cfg = cfg
            best_raw_desc = ret.raw_descriptors
            best_feat_time = ret.feature_extraction_time_s

    for vs in [256, 512, 1024, 2048]:
        if vs == base["bovw"]["vocab_size"]:
            continue
        cfg = copy.deepcopy(base)
        cfg["features"]["n_keypoints"] = best_kp
        cfg["bovw"]["vocab_size"] = vs
        ret = build_retriever("sift", cfg)
        if best_raw_desc is not None:
            ret.fit_from_descriptors(best_raw_desc, feature_extraction_time_s=best_feat_time)
            res = benchmark_retriever(ret, dataset, query_ids=dataset.val_ids, skip_fit=True)
        else:
            res = benchmark_retriever(ret, dataset, query_ids=dataset.val_ids)
        m = res["metrics"]
        tracker.log_run(
            "tuning",
            "sift",
            m,
            cfg,
            split="val",
            notes=f"SIFT sweep n_keypoints={best_kp}, vocab_size={vs}",
            write_markdown=False,
        )
        if m["mAP_medium"] > best_score:
            best_score = m["mAP_medium"]
            best_cfg = cfg

    return best_cfg


def _tune_orb(dataset: ROxford5kDataset, tracker: ExperimentTracker) -> Dict[str, Any]:
    base = load_config("orb")
    best_cfg = copy.deepcopy(base)
    best_score = -1.0

    best_kp = base["features"]["n_keypoints"]
    for kp in [100, 250, 500, 1000]:
        cfg = copy.deepcopy(base)
        cfg["features"]["n_keypoints"] = kp
        ret = build_retriever("orb", cfg)
        res = benchmark_retriever(ret, dataset, query_ids=dataset.val_ids)
        m = res["metrics"]
        tracker.log_run(
            "tuning",
            "orb",
            m,
            cfg,
            split="val",
            notes=f"ORB sweep n_keypoints={kp}, fast_threshold={cfg['features']['fast_threshold']}",
            write_markdown=False,
        )
        if m["mAP_medium"] > best_score:
            best_score = m["mAP_medium"]
            best_kp = kp
            best_cfg = cfg

    for ft in [5, 10, 40]:
        cfg = copy.deepcopy(base)
        cfg["features"]["n_keypoints"] = best_kp
        cfg["features"]["fast_threshold"] = ft
        ret = build_retriever("orb", cfg)
        res = benchmark_retriever(ret, dataset, query_ids=dataset.val_ids)
        m = res["metrics"]
        tracker.log_run(
            "tuning",
            "orb",
            m,
            cfg,
            split="val",
            notes=f"ORB sweep n_keypoints={best_kp}, fast_threshold={ft}",
            write_markdown=False,
        )
        if m["mAP_medium"] > best_score:
            best_score = m["mAP_medium"]
            best_cfg = cfg

    return best_cfg


def _tune_resnet(dataset: ROxford5kDataset, tracker: ExperimentTracker) -> Dict[str, Any]:
    base = load_config("resnet")
    best_cfg = copy.deepcopy(base)
    best_score = -1.0

    best_size = base["model"]["input_size"]
    for sz in [160, 224, 320]:
        cfg = copy.deepcopy(base)
        cfg["model"]["input_size"] = sz
        ret = build_retriever("resnet", cfg)
        res = benchmark_retriever(ret, dataset, query_ids=dataset.val_ids)
        m = res["metrics"]
        tracker.log_run(
            "tuning",
            "resnet",
            m,
            cfg,
            split="val",
            notes=f"ResNet-50 sweep input_size={sz}, similarity=cosine",
            write_markdown=False,
        )
        if m["mAP_medium"] > best_score:
            best_score = m["mAP_medium"]
            best_size = sz
            best_cfg = cfg

    for sim in ["euclidean"]:
        cfg = copy.deepcopy(base)
        cfg["model"]["input_size"] = best_size
        cfg["retrieval"]["similarity"] = sim
        ret = build_retriever("resnet", cfg)
        res = benchmark_retriever(ret, dataset, query_ids=dataset.val_ids)
        m = res["metrics"]
        tracker.log_run(
            "tuning",
            "resnet",
            m,
            cfg,
            split="val",
            notes=f"ResNet-50 sweep input_size={best_size}, similarity={sim}",
            write_markdown=False,
        )
        if m["mAP_medium"] > best_score:
            best_score = m["mAP_medium"]
            best_cfg = cfg

    return best_cfg


def _tune_dinov2(dataset: ROxford5kDataset, tracker: ExperimentTracker) -> Dict[str, Any]:
    base = load_config("dinov2")
    best_cfg = copy.deepcopy(base)
    best_score = -1.0

    best_size = base["model"]["input_size"]
    for sz in [224, 336, 448]:
        cfg = copy.deepcopy(base)
        cfg["model"]["input_size"] = sz
        ret = build_retriever("dinov2", cfg)
        res = benchmark_retriever(ret, dataset, query_ids=dataset.val_ids)
        m = res["metrics"]
        tracker.log_run(
            "tuning",
            "dinov2",
            m,
            cfg,
            split="val",
            notes=f"DINOv2 sweep input_size={sz}, pooling=cls",
            write_markdown=False,
        )
        if m["mAP_medium"] > best_score:
            best_score = m["mAP_medium"]
            best_size = sz
            best_cfg = cfg

    for pool in ["mean", "gem"]:
        cfg = copy.deepcopy(base)
        cfg["model"]["input_size"] = best_size
        cfg["model"]["pooling"] = pool
        ret = build_retriever("dinov2", cfg)
        res = benchmark_retriever(ret, dataset, query_ids=dataset.val_ids)
        m = res["metrics"]
        tracker.log_run(
            "tuning",
            "dinov2",
            m,
            cfg,
            split="val",
            notes=f"DINOv2 sweep input_size={best_size}, pooling={pool}",
            write_markdown=False,
        )
        if m["mAP_medium"] > best_score:
            best_score = m["mAP_medium"]
            best_cfg = cfg

    return best_cfg


def run_tuning(
    methods: Optional[List[str]] = None,
    dataset: Optional[ROxford5kDataset] = None,
    tracker: Optional[ExperimentTracker] = None,
) -> Dict[str, Dict[str, Any]]:
    methods = methods or ALL_METHODS
    dataset = dataset or ROxford5kDataset()
    tracker = tracker or ExperimentTracker()

    best_path = Path("results/processed/best_configs.json")
    best_configs = load_best_configs(best_path)

    tuners = {
        "sift": _tune_sift,
        "orb": _tune_orb,
        "resnet": _tune_resnet,
        "dinov2": _tune_dinov2,
    }

    for method in methods:
        best_cfg = tuners[method](dataset, tracker)
        best_configs[method] = best_cfg
        _save_best_configs(best_configs, best_path)

        ret = build_retriever(method, best_cfg)
        test_res = benchmark_retriever(ret, dataset, query_ids=dataset.test_ids)
        exp_id = tracker.log_run(
            "tuned_test",
            method,
            test_res["metrics"],
            best_cfg,
            split="test",
            notes=f"Best validation-selected configuration for {method} evaluated on held-out test split.",
            objective=f"Evaluate tuned {method} configuration on held-out test queries.",
        )
        m = test_res["metrics"]
        print(
            f"[{exp_id}] Tuned Test {method:6s} | "
            f"mAP(E/M/H): {m['mAP_easy']:.4f}/{m['mAP_medium']:.4f}/{m['mAP_hard']:.4f}"
        )

    return best_configs
