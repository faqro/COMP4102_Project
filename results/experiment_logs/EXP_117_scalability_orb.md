**Experiment ID:** EXP_117

**Objective:** Evaluate orb under scalability (none:0)

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
- Easy mAP: 0.3153
- Medium mAP: 0.2012
- Hard mAP: 0.1254
- Medium P@1 / P@5: 0.4571 / 0.4057
- Medium R@5 / R@10: 0.0684 / 0.0932
- Feature Extraction Time: 23.08 s
- Indexing Time: 0.00 s
- Query Latency: 6146.67 ms (feat: 14.72 ms, search: 6131.94 ms)
- Memory Usage: 151.36 MB

**Observations:** Scalability test at 100% gallery (4993 images)

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
