**Experiment ID:** EXP_007

**Objective:** Measure baseline ROxford5k retrieval accuracy, latency, and memory for dinov2.

**Hypothesis:** Deep embeddings (ResNet-50, DINOv2) will outperform classical local features on global landmark retrieval.

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
    "input_size": 224,
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
- Easy mAP: 0.8194
- Medium mAP: 0.6729
- Hard mAP: 0.4860
- Medium P@1 / P@5: 0.9000 / 0.7914
- Medium R@5 / R@10: 0.2499 / 0.3153
- Feature Extraction Time: 201.43 s
- Indexing Time: 0.04 s
- Query Latency: 59.00 ms (feat: 35.21 ms, search: 23.78 ms)
- Memory Usage: 14.63 MB

**Observations:** Baseline configuration for dinov2

**Failure cases:** See qualitative analysis in results/processed/.

**Interpretation:** Evaluated on 4993 gallery images and 70 queries (all split).

**Next steps:** Proceed to subsequent experiment phase or visualization.
