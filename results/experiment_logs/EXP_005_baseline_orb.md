**Experiment ID:** EXP_005

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
- Easy mAP: 0.2061
- Medium mAP: 0.1264
- Hard mAP: 0.1151
- Medium P@1 / P@5: 0.0571 / 0.2143
- Medium R@5 / R@10: 0.0405 / 0.0602
- Feature Extraction Time: 316.64 s
- Indexing Time: 0.00 s
- Query Latency: 15113.22 ms (feat: 15.66 ms, search: 15097.56 ms)
- Memory Usage: 75.20 MB

**Observations:** Baseline configuration for orb

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
