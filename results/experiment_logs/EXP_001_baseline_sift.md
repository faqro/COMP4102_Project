**Experiment ID:** EXP_001

**Objective:** Measure baseline ROxford5k retrieval accuracy, latency, and memory for sift.

**Hypothesis:** Deep embeddings (ResNet-50, DINOv2) will outperform classical local features on global landmark retrieval.

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
    "n_keypoints": 500,
    "contrast_threshold": 0.04,
    "edge_threshold": 10,
    "sigma": 1.6
  },
  "bovw": {
    "vocab_size": 1024,
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
- Easy mAP: 0.2092
- Medium mAP: 0.1359
- Hard mAP: 0.1228
- Medium P@1 / P@5: 0.2429 / 0.2029
- Medium R@5 / R@10: 0.0139 / 0.0349
- Feature Extraction Time: 536.92 s
- Indexing Time: 53.14 s
- Query Latency: 88.33 ms (feat: 64.80 ms, search: 23.53 ms)
- Memory Usage: 19.50 MB

**Observations:** Baseline configuration for sift

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
