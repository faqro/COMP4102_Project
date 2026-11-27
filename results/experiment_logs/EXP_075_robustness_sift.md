**Experiment ID:** EXP_075

**Objective:** Evaluate sift under robustness (none:0)

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
- Easy mAP: 0.2399
- Medium mAP: 0.1706
- Hard mAP: 0.1355
- Medium P@1 / P@5: 0.3857 / 0.3114
- Medium R@5 / R@10: 0.0695 / 0.0888
- Feature Extraction Time: 216.99 s
- Indexing Time: 47.77 s
- Query Latency: 65.84 ms (feat: 30.84 ms, search: 35.01 ms)
- Memory Usage: 39.01 MB

**Observations:** Robustness test none=0 for sift

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
