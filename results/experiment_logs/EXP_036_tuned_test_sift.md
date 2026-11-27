**Experiment ID:** EXP_036

**Objective:** Evaluate tuned sift configuration on held-out test queries.

**Hypothesis:** Compare retrieval accuracy, robustness, and efficiency under controlled conditions.

**Configuration:**
```json
{
  "method": "sift",
  "seed": 42,
  "dataset": {
    "name": "roxford5k",
    "root": "data/raw/roxford5k",
    "max_image_size": 1024
  },
  "features": {
    "n_keypoints": 1000,
    "contrast_threshold": 0.04,
    "edge_threshold": 10,
    "sigma": 1.6
  },
  "bovw": {
    "vocab_size": 2048,
    "use_tfidf": true,
    "max_descriptors_per_image": 500,
    "sample_size_for_kmeans": 150000,
    "kmeans_batch_size": 4096,
    "kmeans_max_iter": 100
  },
  "retrieval": {
    "mode": "bovw",
    "similarity": "cosine",
    "ratio_threshold": 0.75,
    "top_k": 100
  }
}
```

**Results:**
- Easy mAP: 0.2704
- Medium mAP: 0.1857
- Hard mAP: 0.1564
- Medium P@1 / P@5: 0.3673 / 0.3143
- Medium R@5 / R@10: 0.0800 / 0.1019
- Feature Extraction Time: 223.79 s
- Indexing Time: 52.14 s
- Query Latency: 76.22 ms (feat: 40.07 ms, search: 36.15 ms)
- Memory Usage: 39.01 MB

**Observations:** Best validation-selected configuration for sift evaluated on held-out test split.

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 49 queries (test split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
