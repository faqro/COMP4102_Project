**Experiment ID:** EXP_003

**Objective:** Measure baseline ROxford5k retrieval accuracy, latency, and memory for orb.

**Hypothesis:** Deep embeddings (ResNet-50, DINOv2) will outperform classical local features on global landmark retrieval.

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
    "n_keypoints": 500,
    "scale_factor": 1.2,
    "n_levels": 8,
    "fast_threshold": 20
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
- Easy mAP: 0.2041
- Medium mAP: 0.1251
- Hard mAP: 0.1146
- Medium P@1 / P@5: 0.0571 / 0.2143
- Medium R@5 / R@10: 0.0405 / 0.0600
- Feature Extraction Time: 41.10 s
- Indexing Time: 0.00 s
- Query Latency: 4664.31 ms (feat: 13.00 ms, search: 4651.31 ms)
- Memory Usage: 75.20 MB

**Observations:** Baseline configuration for orb

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
