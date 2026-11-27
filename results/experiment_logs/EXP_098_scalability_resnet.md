**Experiment ID:** EXP_098

**Objective:** Evaluate resnet under scalability (none:0)

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
- Easy mAP: 0.3603
- Medium mAP: 0.2562
- Hard mAP: 0.1614
- Medium P@1 / P@5: 0.3571 / 0.3486
- Medium R@5 / R@10: 0.0575 / 0.0831
- Feature Extraction Time: 23.63 s
- Indexing Time: 0.04 s
- Query Latency: 37.15 ms (feat: 10.45 ms, search: 26.71 ms)
- Memory Usage: 39.01 MB

**Observations:** Scalability test at 100% gallery (4993 images)

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
