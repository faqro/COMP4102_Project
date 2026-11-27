**Experiment ID:** EXP_056

**Objective:** Evaluate dinov2 under robustness (none:0)

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
- Easy mAP: 0.8534
- Medium mAP: 0.7059
- Hard mAP: 0.5225
- Medium P@1 / P@5: 0.9429 / 0.7857
- Medium R@5 / R@10: 0.2423 / 0.3281
- Feature Extraction Time: 125.85 s
- Indexing Time: 0.02 s
- Query Latency: 96.44 ms (feat: 84.94 ms, search: 11.50 ms)
- Memory Usage: 14.63 MB

**Observations:** Robustness test none=0 for dinov2

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
