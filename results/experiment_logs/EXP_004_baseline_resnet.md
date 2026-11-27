**Experiment ID:** EXP_004

**Objective:** Measure baseline ROxford5k retrieval accuracy, latency, and memory for resnet.

**Hypothesis:** Deep embeddings (ResNet-50, DINOv2) will outperform classical local features on global landmark retrieval.

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
- Feature Extraction Time: 142.50 s
- Indexing Time: 0.13 s
- Query Latency: 112.28 ms (feat: 38.82 ms, search: 73.46 ms)
- Memory Usage: 39.01 MB

**Observations:** Baseline configuration for resnet

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
