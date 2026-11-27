"""Unit tests for metrics, preprocessing, vector index, and retrievers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.data.preprocessing import TRANSFORMATION_SPECS, transform_image
from src.evaluation.metrics import compute_ap, evaluate_all_protocols
from src.features.orb import ORBRetriever
from src.features.resnet import ResNetRetriever
from src.features.sift import SIFTRetriever
from src.retrieval.index import VectorIndex


@dataclass
class _DummyQuery:
    easy: np.ndarray
    hard: np.ndarray
    junk: np.ndarray


def _make_synthetic_image(seed: int, size: int = 128) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    arr[20:60, 20:60] = (seed * 53) % 255
    arr[70:105, 65:100] = (seed * 97 + 40) % 255
    return Image.fromarray(arr)


def test_metrics_perfect_and_junk_handling() -> None:
    q = _DummyQuery(
        easy=np.array([2, 4]),
        hard=np.array([1]),
        junk=np.array([0]),
    )
    ranking = np.array([0, 2, 4, 1, 3])
    metrics = evaluate_all_protocols([ranking], [q], ks=(1, 5))
    assert np.isclose(metrics["mAP_easy"], 1.0)
    assert np.isclose(metrics["mAP_medium"], 1.0)
    assert np.isclose(metrics["mAP_hard"], 1.0)
    assert np.isclose(metrics["P@1_medium"], 1.0)
    assert np.isclose(metrics["R@5_medium"], 1.0)


def test_compute_ap_known_order() -> None:
    ap = compute_ap(np.array([0, 2]), nres=2)
    assert 0.8 < ap < 0.95


def test_transformations_preserve_valid_rgb() -> None:
    img = _make_synthetic_image(1, size=160)
    for t_type, _, param in TRANSFORMATION_SPECS:
        out = transform_image(img, t_type, param, seed=42)
        assert out.mode == "RGB"
        assert out.size[0] > 0 and out.size[1] > 0


def test_vector_index_exact_and_faiss() -> None:
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(20, 32)).astype(np.float32)
    q = vecs[7].copy()

    exact = VectorIndex(metric="cosine", use_faiss=False)
    exact.build(vecs)
    res_exact = exact.search(q, top_k=3)

    faiss_idx = VectorIndex(metric="cosine", use_faiss=True, faiss_type="flat")
    faiss_idx.build(vecs)
    res_faiss = faiss_idx.search(q, top_k=3)

    assert res_exact[0][0] == 7
    assert res_faiss[0][0] == 7


def test_sift_and_orb_retrievers() -> None:
    gallery = [_make_synthetic_image(i) for i in range(6)]
    query = gallery[3]

    sift = SIFTRetriever(n_keypoints=100, vocab_size=16, max_image_size=128)
    sift.fit(gallery)
    sift_res = sift.retrieve(query, top_k=3)
    assert len(sift_res) == 3
    assert sift_res[0][0] == 3

    orb = ORBRetriever(n_keypoints=100, mode="direct", max_image_size=128)
    orb.fit(gallery)
    orb_res = orb.retrieve(query, top_k=3)
    assert len(orb_res) == 3
    assert orb_res[0][0] == 3


def test_resnet_retriever() -> None:
    gallery = [_make_synthetic_image(i, size=160) for i in range(4)]
    query = gallery[2]
    resnet = ResNetRetriever(input_size=160, batch_size=2, device="cpu")
    resnet.fit(gallery)
    res = resnet.retrieve(query, top_k=2)
    assert len(res) == 2
    assert res[0][0] == 2
