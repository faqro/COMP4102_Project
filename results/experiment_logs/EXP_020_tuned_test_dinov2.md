**Experiment ID:** EXP_020

**Objective:** Evaluate tuned dinov2 configuration on held-out test queries.

**Hypothesis:** Compare retrieval accuracy, robustness, and efficiency under controlled conditions.

**Configuration:**
```json
{
  "method": "dinov2",
  "seed": 42,
  "dataset": {
    "name": "roxford5k",
    "root": "data/raw/roxford5k"
  },
  "model": {
    "model_name": "facebook/dinov2-base",
    "hub_repo": "facebookresearch/dinov2",
    "hub_model": "dinov2_vitb14",
    "input_size": 448,
    "pooling": "cls",
    "batch_size": 32,
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
- Easy mAP: 0.8892
- Medium mAP: 0.7378
- Hard mAP: 0.5567
- Medium P@1 / P@5: 0.9388 / 0.7551
- Medium R@5 / R@10: 0.2902 / 0.3819
- Feature Extraction Time: 149.47 s
- Indexing Time: 0.03 s
- Query Latency: 99.47 ms (feat: 84.66 ms, search: 14.81 ms)
- Memory Usage: 14.63 MB

**Observations:** Best validation-selected configuration for dinov2 evaluated on held-out test split.

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 49 queries (test split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
