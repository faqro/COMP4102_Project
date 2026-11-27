"""Shared configuration loading, retriever factory, and experiment tracking."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.features.dinov2 import DINOv2Retriever
from src.features.orb import ORBRetriever
from src.features.resnet import ResNetRetriever
from src.features.sift import SIFTRetriever
from src.retrieval.index import ImageRetriever

ALL_METHODS = ["sift", "orb", "resnet", "dinov2"]

CSV_COLUMNS = [
    "experiment_id",
    "timestamp",
    "experiment_type",
    "method",
    "dataset",
    "split",
    "transformation",
    "transform_level",
    "gallery_fraction",
    "n_gallery",
    "n_queries",
    "mAP_easy",
    "mAP_medium",
    "mAP_hard",
    "P@1_medium",
    "P@5_medium",
    "R@5_medium",
    "R@10_medium",
    "feature_extraction_time_s",
    "indexing_time_s",
    "query_feature_time_ms",
    "query_search_time_ms",
    "query_latency_ms",
    "memory_mb",
    "config_json",
    "notes",
]


def load_config(method: str, config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else Path("configs") / f"{method}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_retriever(method: str, cfg: Dict[str, Any]) -> ImageRetriever:
    seed = int(cfg.get("seed", 42))
    if method == "sift":
        feat = cfg.get("features", {})
        bovw = cfg.get("bovw", {})
        ret = cfg.get("retrieval", {})
        ds = cfg.get("dataset", {})
        return SIFTRetriever(
            n_keypoints=int(feat.get("n_keypoints", 500)),
            contrast_threshold=float(feat.get("contrast_threshold", 0.04)),
            edge_threshold=float(feat.get("edge_threshold", 10.0)),
            sigma=float(feat.get("sigma", 1.6)),
            vocab_size=int(bovw.get("vocab_size", 1024)),
            use_tfidf=bool(bovw.get("use_tfidf", True)),
            sample_size_for_kmeans=int(bovw.get("sample_size_for_kmeans", 150000)),
            kmeans_batch_size=int(bovw.get("kmeans_batch_size", 4096)),
            kmeans_max_iter=int(bovw.get("kmeans_max_iter", 100)),
            mode=str(ret.get("mode", "bovw")),
            similarity=str(ret.get("similarity", "cosine")),
            ratio_threshold=float(ret.get("ratio_threshold", 0.75)),
            max_image_size=ds.get("max_image_size", 1024),
            seed=seed,
        )
    if method == "orb":
        feat = cfg.get("features", {})
        ret = cfg.get("retrieval", {})
        ds = cfg.get("dataset", {})
        return ORBRetriever(
            n_keypoints=int(feat.get("n_keypoints", 500)),
            scale_factor=float(feat.get("scale_factor", 1.2)),
            n_levels=int(feat.get("n_levels", 8)),
            fast_threshold=int(feat.get("fast_threshold", 20)),
            mode=str(ret.get("mode", "direct")),
            ratio_threshold=float(ret.get("ratio_threshold", 0.75)),
            vocab_size=int(ret.get("vocab_size", 512)),
            max_image_size=ds.get("max_image_size", 1024),
            seed=seed,
        )
    if method == "resnet":
        m = cfg.get("model", {})
        ret = cfg.get("retrieval", {})
        return ResNetRetriever(
            input_size=int(m.get("input_size", 224)),
            batch_size=int(m.get("batch_size", 64)),
            device=str(m.get("device", "auto")),
            normalize=bool(m.get("normalize", True)),
            pca_dim=m.get("pca_dim", None),
            weights=str(m.get("weights", "IMAGENET1K_V2")),
            similarity=str(ret.get("similarity", "cosine")),
            use_faiss=bool(ret.get("use_faiss", False)),
            faiss_type=str(ret.get("faiss_type", "flat")),
            seed=seed,
        )
    if method == "dinov2":
        m = cfg.get("model", {})
        ret = cfg.get("retrieval", {})
        return DINOv2Retriever(
            model_name=str(m.get("model_name", "facebook/dinov2-base")),
            input_size=int(m.get("input_size", 224)),
            pooling=str(m.get("pooling", "cls")),
            batch_size=int(m.get("batch_size", 32)),
            device=str(m.get("device", "auto")),
            normalize=bool(m.get("normalize", True)),
            pca_dim=m.get("pca_dim", None),
            similarity=str(ret.get("similarity", "cosine")),
            use_faiss=bool(ret.get("use_faiss", False)),
            faiss_type=str(ret.get("faiss_type", "flat")),
            seed=seed,
        )
    raise ValueError(f"Unknown method: {method}")


class ExperimentTracker:
    """Appends quantitative CSV rows and writes qualitative Markdown experiment logs."""

    def __init__(
        self,
        csv_path: str | Path = "results/raw/experiments.csv",
        logs_dir: str | Path = "results/experiment_logs",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.logs_dir = Path(logs_dir)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def _next_id(self) -> str:
        with open(self.csv_path, "r", encoding="utf-8") as f:
            n_rows = max(0, sum(1 for _ in f) - 1)
        return f"EXP_{n_rows + 1:03d}"

    def log_run(
        self,
        experiment_type: str,
        method: str,
        metrics: Dict[str, Any],
        cfg: Dict[str, Any],
        split: str = "all",
        transformation: str = "none",
        transform_level: str = "0",
        gallery_fraction: float = 1.0,
        notes: str = "",
        write_markdown: bool = True,
        objective: str = "",
        hypothesis: str = "",
    ) -> str:
        exp_id = self._next_id()
        row = {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment_type": experiment_type,
            "method": method,
            "dataset": cfg.get("dataset", {}).get("name", "roxford5k"),
            "split": split,
            "transformation": transformation,
            "transform_level": transform_level,
            "gallery_fraction": f"{gallery_fraction:.2f}",
            "n_gallery": metrics.get("n_gallery", 0),
            "n_queries": metrics.get("n_queries", 0),
            "mAP_easy": f"{metrics.get('mAP_easy', 0.0):.6f}",
            "mAP_medium": f"{metrics.get('mAP_medium', 0.0):.6f}",
            "mAP_hard": f"{metrics.get('mAP_hard', 0.0):.6f}",
            "P@1_medium": f"{metrics.get('P@1_medium', 0.0):.6f}",
            "P@5_medium": f"{metrics.get('P@5_medium', 0.0):.6f}",
            "R@5_medium": f"{metrics.get('R@5_medium', 0.0):.6f}",
            "R@10_medium": f"{metrics.get('R@10_medium', 0.0):.6f}",
            "feature_extraction_time_s": f"{metrics.get('feature_extraction_time_s', 0.0):.4f}",
            "indexing_time_s": f"{metrics.get('indexing_time_s', 0.0):.4f}",
            "query_feature_time_ms": f"{metrics.get('query_feature_time_ms', 0.0):.4f}",
            "query_search_time_ms": f"{metrics.get('query_search_time_ms', 0.0):.4f}",
            "query_latency_ms": f"{metrics.get('query_latency_ms', 0.0):.4f}",
            "memory_mb": f"{metrics.get('memory_mb', 0.0):.4f}",
            "config_json": json.dumps(cfg, sort_keys=True),
            "notes": notes,
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        if write_markdown:
            md_path = self.logs_dir / f"{exp_id}_{experiment_type}_{method}.md"
            md = (
                f"**Experiment ID:** {exp_id}\n\n"
                f"**Objective:** {objective or f'Evaluate {method} under {experiment_type} ({transformation}:{transform_level})'}\n\n"
                f"**Hypothesis:** {hypothesis or 'Compare retrieval accuracy, robustness, and efficiency under controlled conditions.'}\n\n"
                f"**Configuration:**\n```json\n{json.dumps(cfg, indent=2)}\n```\n\n"
                f"**Results:**\n"
                f"- Easy mAP: {metrics.get('mAP_easy', 0.0):.4f}\n"
                f"- Medium mAP: {metrics.get('mAP_medium', 0.0):.4f}\n"
                f"- Hard mAP: {metrics.get('mAP_hard', 0.0):.4f}\n"
                f"- Medium P@1 / P@5: {metrics.get('P@1_medium', 0.0):.4f} / {metrics.get('P@5_medium', 0.0):.4f}\n"
                f"- Medium R@5 / R@10: {metrics.get('R@5_medium', 0.0):.4f} / {metrics.get('R@10_medium', 0.0):.4f}\n"
                f"- Feature Extraction Time: {metrics.get('feature_extraction_time_s', 0.0):.2f} s\n"
                f"- Indexing Time: {metrics.get('indexing_time_s', 0.0):.2f} s\n"
                f"- Query Latency: {metrics.get('query_latency_ms', 0.0):.2f} ms "
                f"(feat: {metrics.get('query_feature_time_ms', 0.0):.2f} ms, search: {metrics.get('query_search_time_ms', 0.0):.2f} ms)\n"
                f"- Memory Usage: {metrics.get('memory_mb', 0.0):.2f} MB\n\n"
                f"**Observations:** {notes or 'Recorded automatically by benchmark pipeline.'}\n\n"
                f"**Failure cases:** See qualitative analysis in results/processed/.\n\n"
                f"**Interpretation:** Evaluated on {metrics.get('n_gallery', 0)} gallery images and {metrics.get('n_queries', 0)} queries ({split} split).\n\n"
                f"**Next steps:** Proceed to subsequent experiment phase or visualization.\n"
            )
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)

        return exp_id
