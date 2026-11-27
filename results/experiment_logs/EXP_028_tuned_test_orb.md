**Experiment ID:** EXP_028

**Objective:** Evaluate tuned orb configuration on held-out test queries.

**Hypothesis:** Compare retrieval accuracy, robustness, and efficiency under controlled conditions.

**Configuration:**
```json
{
  "method": "orb",
  "seed": 42,
  "dataset": {
    "name": "roxford5k",
    "root": "data/raw/roxford5k",
    "max_image_size": 1024
  },
  "features": {
    "n_keypoints": 1000,
    "scale_factor": 1.2,
    "n_levels": 8,
    "fast_threshold": 10
  },
  "retrieval": {
    "mode": "direct",
    "ratio_threshold": 0.75,
    "vocab_size": 512,
    "top_k": 100
  }
}
```

**Results:**
- Easy mAP: 0.3648
- Medium mAP: 0.2221
- Hard mAP: 0.1483
- Medium P@1 / P@5: 0.4898 / 0.4367
- Medium R@5 / R@10: 0.0755 / 0.1009
- Feature Extraction Time: 22.61 s
- Indexing Time: 0.00 s
- Query Latency: 5937.06 ms (feat: 14.27 ms, search: 5922.80 ms)
- Memory Usage: 151.36 MB

**Observations:** Best validation-selected configuration for orb evaluated on held-out test split.

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 49 queries (test split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
