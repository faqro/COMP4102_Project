**Experiment ID:** EXP_014

**Objective:** Evaluate tuned resnet configuration on held-out test queries.

**Hypothesis:** Compare retrieval accuracy, robustness, and efficiency under controlled conditions.

**Configuration:**
```json
{
  "method": "resnet",
  "seed": 42,
  "dataset": {
    "name": "roxford5k",
    "root": "data/raw/roxford5k"
  },
  "model": {
    "architecture": "resnet50",
    "weights": "IMAGENET1K_V2",
    "input_size": 224,
    "batch_size": 64,
    "device": "auto",
    "normalize": true,
    "pca_dim": null
  },
  "retrieval": {
    "similarity": "cosine",
    "use_faiss": false,
    "top_k": 100
  }
}
```

**Results:**
- Easy mAP: 0.3752
- Medium mAP: 0.2582
- Hard mAP: 0.1652
- Medium P@1 / P@5: 0.3673 / 0.3306
- Medium R@5 / R@10: 0.0682 / 0.0917
- Feature Extraction Time: 39.57 s
- Indexing Time: 0.08 s
- Query Latency: 71.28 ms (feat: 18.99 ms, search: 52.29 ms)
- Memory Usage: 39.01 MB

**Observations:** Best validation-selected configuration for resnet evaluated on held-out test split.

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 49 queries (test split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
