**Experiment ID:** EXP_099

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
    "faiss_type": "flat"
  }
}
```

**Results:**
- Easy mAP: 0.3603
- Medium mAP: 0.2562
- Hard mAP: 0.1614
- Medium P@1 / P@5: 0.3571 / 0.3486
- Medium R@5 / R@10: 0.0575 / 0.0831
- Feature Extraction Time: 24.16 s
- Indexing Time: 0.06 s
- Query Latency: 13.50 ms (feat: 9.51 ms, search: 3.99 ms)
- Memory Usage: 39.01 MB

**Observations:** FAISS (flat) indexing comparison for resnet

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
