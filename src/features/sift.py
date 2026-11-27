"""SIFT feature extractor and Bag-of-Visual-Words (BoVW) / direct retriever."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import faiss
import numpy as np
from PIL import Image

from src.data.preprocessing import pil_to_gray
from src.retrieval.index import ImageRetriever, VectorIndex
from src.retrieval.similarity import l2_normalize


def _load_gray(image: Union[str, Path, Image.Image, np.ndarray], max_size: Optional[int]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            if max_size is not None and max(gray.shape[:2]) > max_size:
                h, w = gray.shape[:2]
                scale = max_size / float(max(h, w))
                gray = cv2.resize(
                    gray,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            return gray
        with Image.open(image) as im:
            return pil_to_gray(im, max_size=max_size)
    if isinstance(image, Image.Image):
        return pil_to_gray(image, max_size=max_size)
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        if max_size is not None and max(gray.shape[:2]) > max_size:
            h, w = gray.shape[:2]
            scale = max_size / float(max(h, w))
            gray = cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
        return gray
    raise TypeError(f"Unsupported image type: {type(image)}")


class SIFTRetriever(ImageRetriever):
    """SIFT-based image retriever supporting BoVW (with optional TF-IDF) and direct matching."""

    name = "sift"

    def __init__(
        self,
        n_keypoints: int = 500,
        vocab_size: int = 1024,
        use_tfidf: bool = True,
        mode: str = "bovw",
        similarity: str = "cosine",
        ratio_threshold: float = 0.75,
        max_image_size: Optional[int] = 1024,
        sample_size_for_kmeans: int = 150000,
        kmeans_batch_size: int = 4096,
        kmeans_max_iter: int = 100,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_keypoints = n_keypoints
        self.vocab_size = vocab_size
        self.use_tfidf = use_tfidf
        self.mode = mode
        self.similarity = similarity
        self.ratio_threshold = ratio_threshold
        self.max_image_size = max_image_size
        self.sample_size_for_kmeans = sample_size_for_kmeans
        self.kmeans_batch_size = kmeans_batch_size
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed

        self.detector = cv2.SIFT_create(
            nfeatures=n_keypoints,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )
        self.centroids: Optional[np.ndarray] = None
        self._centroid_index: Optional[faiss.IndexFlatL2] = None
        self.idf: Optional[np.ndarray] = None
        self.bovw_matrix: Optional[np.ndarray] = None
        self.vector_index: Optional[VectorIndex] = None
        self.raw_descriptors: List[np.ndarray] = []

    def extract_descriptors(self, image: Any) -> np.ndarray:
        gray = _load_gray(image, self.max_image_size)
        _, desc = self.detector.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            return np.zeros((0, 128), dtype=np.float32)
        return np.asarray(desc, dtype=np.float32)

    def _assign_words(self, descriptors: np.ndarray) -> np.ndarray:
        if len(descriptors) == 0 or self._centroid_index is None:
            return np.zeros((0,), dtype=np.int64)
        _, words = self._centroid_index.search(np.ascontiguousarray(descriptors, dtype=np.float32), 1)
        return words.ravel().astype(np.int64)

    def _descriptors_to_hist(self, descriptors: np.ndarray) -> np.ndarray:
        hist = np.zeros((self.vocab_size,), dtype=np.float32)
        if len(descriptors) > 0:
            words = self._assign_words(descriptors)
            counts = np.bincount(words, minlength=self.vocab_size).astype(np.float32)
            hist[: len(counts)] = counts[: self.vocab_size]
            if self.use_tfidf and self.idf is not None:
                total = float(hist.sum())
                if total > 0:
                    tf = hist / total
                    hist = tf * self.idf
        return l2_normalize(hist)

    def fit(self, database_images: Sequence[Any]) -> None:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            raw_descriptors = list(pool.map(self.extract_descriptors, database_images))
        feat_time = time.perf_counter() - t0
        self.fit_from_descriptors(raw_descriptors, feature_extraction_time_s=feat_time)

    def fit_from_descriptors(
        self, raw_descriptors: List[np.ndarray], feature_extraction_time_s: float = 0.0
    ) -> None:
        self.raw_descriptors = raw_descriptors
        self.feature_extraction_time_s = feature_extraction_time_s
        n_images = len(raw_descriptors)

        t1 = time.perf_counter()
        if self.mode == "direct":
            self.bovw_matrix = None
            self.vector_index = None
            self.indexing_time_s = time.perf_counter() - t1
            return

        non_empty = [d for d in self.raw_descriptors if len(d) > 0]
        if not non_empty:
            self.centroids = np.zeros((self.vocab_size, 128), dtype=np.float32)
            self.bovw_matrix = np.zeros((n_images, self.vocab_size), dtype=np.float32)
            self.vector_index = VectorIndex(metric=self.similarity)
            self.vector_index.build(self.bovw_matrix)
            self.indexing_time_s = time.perf_counter() - t1
            return

        all_desc = np.vstack(non_empty)
        rng = np.random.default_rng(self.seed)
        if len(all_desc) > self.sample_size_for_kmeans:
            idx = rng.choice(len(all_desc), size=self.sample_size_for_kmeans, replace=False)
            sample = np.ascontiguousarray(all_desc[idx], dtype=np.float32)
        else:
            sample = np.ascontiguousarray(all_desc, dtype=np.float32)

        n_clusters = min(self.vocab_size, len(sample))
        kmeans = faiss.Kmeans(128, n_clusters, niter=20, verbose=False, seed=self.seed)
        kmeans.train(sample)
        self.centroids = np.ascontiguousarray(kmeans.centroids, dtype=np.float32)
        if n_clusters < self.vocab_size:
            pad = np.zeros((self.vocab_size - n_clusters, 128), dtype=np.float32)
            self.centroids = np.vstack([self.centroids, pad])

        self._centroid_index = faiss.IndexFlatL2(128)
        self._centroid_index.add(self.centroids)

        raw_hists = np.zeros((n_images, self.vocab_size), dtype=np.float32)
        for i, desc in enumerate(self.raw_descriptors):
            if len(desc) > 0:
                words = self._assign_words(desc)
                raw_hists[i] = np.bincount(words, minlength=self.vocab_size).astype(np.float32)

        if self.use_tfidf:
            n_docs = float(n_images)
            df = np.sum(raw_hists > 0, axis=0).astype(np.float32)
            self.idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
            row_sums = np.maximum(raw_hists.sum(axis=1, keepdims=True), 1e-12)
            tf = raw_hists / row_sums
            weighted = tf * self.idf[None, :]
            self.bovw_matrix = l2_normalize(weighted)
        else:
            self.idf = None
            self.bovw_matrix = l2_normalize(raw_hists)

        self.vector_index = VectorIndex(metric=self.similarity)
        self.vector_index.build(self.bovw_matrix)
        self.indexing_time_s = time.perf_counter() - t1

    def retrieve(self, query_image: Any, top_k: int = 10) -> List[Tuple[int, float]]:
        t0 = time.perf_counter()
        q_desc = self.extract_descriptors(query_image)
        self.last_query_feature_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if self.mode == "direct":
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            scores = np.zeros((len(self.raw_descriptors),), dtype=np.float32)
            if len(q_desc) >= 2:
                for i, db_desc in enumerate(self.raw_descriptors):
                    if len(db_desc) < 2:
                        continue
                    matches = matcher.knnMatch(q_desc, db_desc, k=2)
                    good = sum(1 for m, n in matches if m.distance < self.ratio_threshold * n.distance)
                    scores[i] = float(good) / float(max(len(q_desc), 1))
            k = min(top_k, len(scores))
            order = np.argsort(-scores)[:k]
            self.last_query_search_time_s = time.perf_counter() - t1
            return [(int(i), float(scores[i])) for i in order]

        if self.vector_index is None:
            raise RuntimeError("SIFTRetriever must be fitted before calling retrieve().")
        q_hist = self._descriptors_to_hist(q_desc)
        results = self.vector_index.search(q_hist, top_k=top_k)
        self.last_query_search_time_s = time.perf_counter() - t1
        return results

    def get_representations(self) -> Any:
        if self.mode == "direct":
            return self.raw_descriptors
        return self.bovw_matrix
