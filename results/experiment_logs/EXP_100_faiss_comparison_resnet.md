**Experiment ID:** EXP_100

**Objective:** Evaluate resnet under faiss_comparison (none:0)

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
    "use_faiss": true,
    "top_k": 100,
    "faiss_type": "ivf"
  }
}
```

**Results:**
- Easy mAP: 0.3477
- Medium mAP: 0.2386
- Hard mAP: 0.1154
- Medium P@1 / P@5: 0.3571 / 0.3486
- Medium R@5 / R@10: 0.0575 / 0.0831
- Feature Extraction Time: 23.80 s
- Indexing Time: 0.20 s
- Query Latency: 10.01 ms (feat: 8.67 ms, search: 1.34 ms)
- Memory Usage: 39.01 MB

**Observations:** FAISS (ivf) indexing comparison for resnet

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
