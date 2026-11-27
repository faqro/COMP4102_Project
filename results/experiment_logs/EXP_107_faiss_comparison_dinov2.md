**Experiment ID:** EXP_107

**Objective:** Evaluate dinov2 under faiss_comparison (none:0)

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
    "use_faiss": true,
    "top_k": 100,
    "faiss_type": "ivf"
  }
}
```

**Results:**
- Easy mAP: 0.8444
- Medium mAP: 0.6894
- Hard mAP: 0.5040
- Medium P@1 / P@5: 0.9429 / 0.7829
- Medium R@5 / R@10: 0.2422 / 0.3267
- Feature Extraction Time: 128.86 s
- Indexing Time: 0.08 s
- Query Latency: 89.20 ms (feat: 88.11 ms, search: 1.10 ms)
- Memory Usage: 14.63 MB

**Observations:** FAISS (ivf) indexing comparison for dinov2

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
