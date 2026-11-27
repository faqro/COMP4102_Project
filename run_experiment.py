"""CLI entry point for running COMP4102 image retrieval experiments."""

from __future__ import annotations

import argparse
import sys

from src.data.dataset import ROxford5kDataset, download_roxford5k
from src.experiments.baseline import run_baseline
from src.experiments.common import ALL_METHODS, ExperimentTracker
from src.experiments.robustness import run_robustness
from src.experiments.scalability import run_scalability
from src.experiments.tuning import run_tuning


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Run COMP4102 Image Retrieval Experiments")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["sift", "orb", "resnet", "dinov2", "all"],
        help="Retrieval method to evaluate",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "tuning", "robustness", "scalability", "all"],
        help="Experiment suite to run",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Ensure ROxford5k dataset is downloaded before running",
    )
    args = parser.parse_args()

    if args.download:
        download_roxford5k()

    methods = ALL_METHODS if args.method == "all" else [args.method]
    dataset = ROxford5kDataset()
    tracker = ExperimentTracker()

    experiments = (
        ["baseline", "tuning", "robustness", "scalability"]
        if args.experiment == "all"
        else [args.experiment]
    )

    for exp in experiments:
        print(f"\n=== Running experiment: {exp} for methods: {methods} ===")
        if exp == "baseline":
            run_baseline(methods=methods, dataset=dataset, tracker=tracker)
        elif exp == "tuning":
            run_tuning(methods=methods, dataset=dataset, tracker=tracker)
        elif exp == "robustness":
            run_robustness(methods=methods, dataset=dataset, tracker=tracker)
        elif exp == "scalability":
            run_scalability(methods=methods, dataset=dataset, tracker=tracker)


if __name__ == "__main__":
    main()
