"""Automated figure generation and qualitative failure analysis from experiment CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps

from src.data.dataset import ROxford5kDataset

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

METHOD_LABELS = {
    "sift": "SIFT (BoVW)",
    "orb": "ORB (Hamming)",
    "resnet": "ResNet-50",
    "dinov2": "DINOv2 ViT-B/14",
}
METHOD_ORDER = ["sift", "orb", "resnet", "dinov2"]
PALETTE = {
    "sift": "#1f77b4",
    "orb": "#ff7f0e",
    "resnet": "#2ca02c",
    "dinov2": "#d62728",
}


def generate_all_figures(
    csv_path: str | Path = "results/raw/experiments.csv",
    figures_dir: str | Path = "figures",
    dataset: ROxford5kDataset | None = None,
) -> List[Path]:
    df = pd.read_csv(csv_path)
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    generated.append(plot_figure1_baseline(df, out_dir))
    generated.append(plot_figure2_robustness(df, out_dir))
    generated.append(plot_figure3_accuracy_vs_latency(df, out_dir))
    generated.append(plot_figure4_parameter_sensitivity(df, out_dir))
    generated.append(plot_figure6_computational_requirements(df, out_dir))
    generated.append(plot_scalability_curves(df, out_dir))

    if dataset is not None:
        p5 = plot_figure5_qualitative_grid(dataset, out_dir)
        if p5 is not None:
            generated.append(p5)
        analyze_failure_cases(dataset, Path("results/processed/failure_analysis.md"))

    return [p for p in generated if p is not None]


def plot_figure1_baseline(df: pd.DataFrame, out_dir: Path) -> Path:
    base = df[df["experiment_type"] == "baseline"].drop_duplicates(subset=["method"], keep="last")
    base = base.set_index("method").reindex([m for m in METHOD_ORDER if m in base["method"].values])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(base))
    width = 0.25
    ax.bar(x - width, base["mAP_easy"], width, label="Easy mAP", color="#4c72b0")
    ax.bar(x, base["mAP_medium"], width, label="Medium mAP", color="#55a868")
    ax.bar(x + width, base["mAP_hard"], width, label="Hard mAP", color="#c44e52")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in base.index])
    ax.set_ylim(0, min(1.05, max(0.3, float(base["mAP_easy"].max()) * 1.18)))
    ax.set_ylabel("Mean Average Precision (mAP)")
    ax.set_title("Figure 1: Baseline Retrieval Performance on ROxford5k")
    ax.legend(frameon=True)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

    fig.tight_layout()
    out = out_dir / "baseline_performance.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_figure2_robustness(df: pd.DataFrame, out_dir: Path) -> Path:
    rob = df[df["experiment_type"] == "robustness"].copy()
    if rob.empty:
        return out_dir / "robustness_transformations.png"

    baseline_map = (
        rob[rob["transformation"] == "none"]
        .drop_duplicates(subset=["method"], keep="last")
        .set_index("method")["mAP_medium"]
        .to_dict()
    )

    families = [
        ("rotation", ["15", "30", "45"], "Rotation (degrees)"),
        ("blur", ["k3", "k5", "k9"], "Gaussian Blur (kernel)"),
        ("noise", ["sigma10", "sigma25", "sigma50"], "Gaussian Noise (σ)"),
        ("brightness", ["0.50", "0.75", "1.25"], "Brightness Factor"),
        ("jpeg", ["q80", "q50", "q20"], "JPEG Quality"),
        ("crop", ["90pct", "70pct", "50pct"], "Center Crop Retained"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    axes_flat = axes.ravel()

    for ax, (fam, levels, title) in zip(axes_flat, families):
        sub = rob[rob["transformation"] == fam]
        x_labels = ["Orig"] + levels
        x_pos = np.arange(len(x_labels))

        for method in METHOD_ORDER:
            if method not in baseline_map:
                continue
            msub = sub[sub["method"] == method]
            vals = [baseline_map[method]]
            for lv in levels:
                row = msub[msub["transform_level"].astype(str) == lv]
                vals.append(float(row["mAP_medium"].iloc[-1]) if not row.empty else np.nan)
            vals_arr = np.array(vals, dtype=float)
            valid = ~np.isnan(vals_arr)
            ax.plot(
                x_pos[valid],
                vals_arr[valid],
                marker="o",
                linewidth=2,
                label=METHOD_LABELS[method],
                color=PALETTE[method],
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes_flat[0].set_ylabel("Medium mAP")
    axes_flat[3].set_ylabel("Medium mAP")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Figure 2: Retrieval Robustness Under Query Transformations (ROxford5k)", y=1.03, fontsize=13)
    fig.tight_layout()
    out = out_dir / "robustness_transformations.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_figure3_accuracy_vs_latency(df: pd.DataFrame, out_dir: Path) -> Path:
    base = df[df["experiment_type"] == "baseline"].drop_duplicates(subset=["method"], keep="last")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for _, row in base.iterrows():
        m = row["method"]
        x = float(row["query_latency_ms"])
        y = float(row["mAP_medium"])
        ax.scatter(x, y, s=140, color=PALETTE.get(m, "#333333"), zorder=3)
        ax.annotate(
            f"{METHOD_LABELS.get(m, m)}\n({x:.1f} ms, {y:.3f})",
            (x, y),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Average Query Latency (ms, log scale)")
    ax.set_ylabel("Medium mAP")
    ax.set_title("Figure 3: Accuracy vs. Query Latency Trade-off")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "accuracy_vs_latency.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_figure4_parameter_sensitivity(df: pd.DataFrame, out_dir: Path) -> Path:
    tune = df[df["experiment_type"] == "tuning"].copy()
    if tune.empty:
        return out_dir / "parameter_sensitivity.png"

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    sift_rows = tune[tune["method"] == "sift"].copy()
    if not sift_rows.empty:
        sift_rows["kp"] = sift_rows["config_json"].apply(lambda s: json.loads(s)["features"]["n_keypoints"])
        sift_rows["vs"] = sift_rows["config_json"].apply(lambda s: json.loads(s)["bovw"]["vocab_size"])
        sub_kp = sift_rows[sift_rows["vs"] == 1024].sort_values("kp")
        if not sub_kp.empty:
            ax.plot(sub_kp["kp"], sub_kp["mAP_medium"], marker="o", label="vocab=1024 (vary kp)", color="#1f77b4")
        ax.set_title("SIFT: Keypoints Sensitivity (Val Split)")
        ax.set_xlabel("Number of Keypoints")
        ax.set_ylabel("Validation Medium mAP")
        ax.legend()

    ax = axes[0, 1]
    orb_rows = tune[tune["method"] == "orb"].copy()
    if not orb_rows.empty:
        orb_rows["kp"] = orb_rows["config_json"].apply(lambda s: json.loads(s)["features"]["n_keypoints"])
        orb_rows["ft"] = orb_rows["config_json"].apply(lambda s: json.loads(s)["features"]["fast_threshold"])
        sub_kp = orb_rows[orb_rows["ft"] == 20].sort_values("kp")
        if not sub_kp.empty:
            ax.plot(sub_kp["kp"], sub_kp["mAP_medium"], marker="s", label="FAST=20 (vary kp)", color="#ff7f0e")
        ax.set_title("ORB: Keypoints Sensitivity (Val Split)")
        ax.set_xlabel("Number of Keypoints")
        ax.set_ylabel("Validation Medium mAP")
        ax.legend()

    ax = axes[1, 0]
    res_rows = tune[tune["method"] == "resnet"].copy()
    if not res_rows.empty:
        res_rows["sz"] = res_rows["config_json"].apply(lambda s: json.loads(s)["model"]["input_size"])
        res_rows["sim"] = res_rows["config_json"].apply(lambda s: json.loads(s)["retrieval"]["similarity"])
        sub = res_rows[res_rows["sim"] == "cosine"].sort_values("sz")
        ax.plot(sub["sz"], sub["mAP_medium"], marker="^", color="#2ca02c", label="Cosine")
        ax.set_title("ResNet-50: Input Resolution Sensitivity (Val Split)")
        ax.set_xlabel("Input Size (px)")
        ax.set_ylabel("Validation Medium mAP")
        ax.legend()

    ax = axes[1, 1]
    dino_rows = tune[tune["method"] == "dinov2"].copy()
    if not dino_rows.empty:
        dino_rows["sz"] = dino_rows["config_json"].apply(lambda s: json.loads(s)["model"]["input_size"])
        dino_rows["pool"] = dino_rows["config_json"].apply(lambda s: json.loads(s)["model"]["pooling"])
        sub = dino_rows[dino_rows["pool"] == "cls"].sort_values("sz")
        ax.plot(sub["sz"], sub["mAP_medium"], marker="D", color="#d62728", label="CLS token")
        ax.set_title("DINOv2: Input Resolution Sensitivity (Val Split)")
        ax.set_xlabel("Input Size (px)")
        ax.set_ylabel("Validation Medium mAP")
        ax.legend()

    fig.suptitle("Figure 4: Hyperparameter Sensitivity Analysis", fontsize=13)
    fig.tight_layout()
    out = out_dir / "parameter_sensitivity.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_figure6_computational_requirements(df: pd.DataFrame, out_dir: Path) -> Path:
    base = df[df["experiment_type"] == "baseline"].drop_duplicates(subset=["method"], keep="last")
    base = base.set_index("method").reindex([m for m in METHOD_ORDER if m in base["method"].values])

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    metrics_info = [
        ("feature_extraction_time_s", "Feature Extraction (s)"),
        ("indexing_time_s", "Indexing Time (s)"),
        ("query_latency_ms", "Query Latency (ms)"),
        ("memory_mb", "Memory Footprint (MB)"),
    ]
    labels = [METHOD_LABELS.get(m, m) for m in base.index]
    colors = [PALETTE.get(m, "#333333") for m in base.index]

    for ax, (col, title) in zip(axes, metrics_info):
        vals = base[col].astype(float).values
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=22)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    fig.suptitle("Figure 6: Computational Resource Comparison (ROxford5k, 4,993 Gallery Images)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "computational_requirements.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_scalability_curves(df: pd.DataFrame, out_dir: Path) -> Path:
    sc = df[df["experiment_type"] == "scalability"].copy()
    if sc.empty:
        return out_dir / "scalability.png"

    sc["gallery_pct"] = sc["gallery_fraction"].astype(float) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for method in METHOD_ORDER:
        sub = sc[sc["method"] == method].sort_values("gallery_pct")
        if sub.empty:
            continue
        axes[0].plot(
            sub["gallery_pct"],
            sub["query_latency_ms"],
            marker="o",
            label=METHOD_LABELS[method],
            color=PALETTE[method],
        )
        axes[1].plot(
            sub["gallery_pct"],
            sub["mAP_medium"],
            marker="o",
            label=METHOD_LABELS[method],
            color=PALETTE[method],
        )

    axes[0].set_title("Query Latency vs. Gallery Size")
    axes[0].set_xlabel("Gallery Size (%)")
    axes[0].set_ylabel("Query Latency (ms)")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].set_title("Medium mAP vs. Gallery Size")
    axes[1].set_xlabel("Gallery Size (%)")
    axes[1].set_ylabel("Medium mAP")
    axes[1].legend()

    fig.suptitle("Experiment D: Scalability with Increasing Database Size", fontsize=12)
    fig.tight_layout()
    out = out_dir / "scalability.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_figure5_qualitative_grid(dataset: ROxford5kDataset, out_dir: Path) -> Path | None:
    rankings: Dict[str, np.ndarray] = {}
    for m in METHOD_ORDER:
        p = Path("results/processed") / f"{m}_baseline_rankings.npz"
        if p.exists():
            rankings[m] = np.load(p)["top10_indices"]
    if len(rankings) < 4:
        return None

    sample_qids = [0, min(35, len(dataset.queries) - 1)]
    n_cols = 6
    n_rows = len(sample_qids) * len(METHOD_ORDER)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.0 * n_rows))
    for q_idx, qid in enumerate(sample_qids):
        q_item = dataset.queries[qid]
        pos_set = set(int(x) for x in np.concatenate([q_item.easy, q_item.hard]))
        junk_set = set(int(x) for x in q_item.junk)
        q_img = dataset.get_query_image(qid).resize((160, 160))

        for m_idx, method in enumerate(METHOD_ORDER):
            row_i = q_idx * len(METHOD_ORDER) + m_idx
            ax_q = axes[row_i, 0]
            ax_q.imshow(q_img)
            ax_q.set_ylabel(f"Q{qid}\n{METHOD_LABELS[method]}", fontsize=8, rotation=0, labelpad=45, va="center")
            ax_q.set_xticks([])
            ax_q.set_yticks([])
            if row_i == 0:
                ax_q.set_title("Query", fontsize=9)

            top5 = rankings[method][qid][:5]
            for k in range(5):
                ax = axes[row_i, k + 1]
                g_idx = int(top5[k])
                g_img = dataset.get_gallery_image(g_idx).resize((160, 160))
                if g_idx in pos_set:
                    color = "green"
                elif g_idx in junk_set:
                    color = "gold"
                else:
                    color = "red"
                bordered = ImageOps.expand(g_img, border=6, fill=color)
                ax.imshow(bordered)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_i == 0:
                    ax.set_title(f"Top {k + 1}", fontsize=9)

    fig.suptitle("Figure 5: Qualitative Top-5 Retrieval Examples (Green=Relevant, Gold=Junk, Red=Non-Relevant)", fontsize=11)
    fig.tight_layout()
    out = out_dir / "qualitative_examples.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def analyze_failure_cases(dataset: ROxford5kDataset, out_path: Path) -> None:
    """Identify hardest queries and disagreement cases across all four methods."""
    rankings: Dict[str, np.ndarray] = {}
    for m in METHOD_ORDER:
        p = Path("results/processed") / f"{m}_baseline_rankings.npz"
        if p.exists():
            rankings[m] = np.load(p)["top10_indices"]
    if not rankings:
        return

    lines = ["# Qualitative Failure-Case Analysis (ROxford5k)\n"]
    for m in METHOD_ORDER:
        if m not in rankings:
            continue
        r_mat = rankings[m]
        p1_failures = []
        for qid, q in enumerate(dataset.queries):
            pos = set(int(x) for x in np.concatenate([q.easy, q.hard]))
            junk = set(int(x) for x in q.junk)
            filtered = [int(idx) for idx in r_mat[qid] if int(idx) not in junk]
            if filtered and filtered[0] not in pos:
                p1_failures.append((qid, q.name, filtered[0], dataset.imlist[filtered[0]]))
        lines.append(f"## {METHOD_LABELS[m]} ({len(p1_failures)}/{len(dataset.queries)} P@1 misses)")
        for qid, qname, g_idx, gname in p1_failures[:5]:
            lines.append(f"- Query {qid} (`{qname}`): Top-1 retrieved `{gname}` (idx={g_idx})")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
