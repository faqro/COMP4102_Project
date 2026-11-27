"""ORB binary feature extractor and Hamming-based image retriever."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.features.sift import _load_gray
from src.retrieval.index import ImageRetriever, VectorIndex
from src.retrieval.similarity import l2_normalize


class ORBRetriever(ImageRetriever):
    """ORB binary descriptor retriever using Hamming distance matching or binary k-medoids BoVW."""

    name = "orb"

    def __init__(
        self,
        n_keypoints: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        fast_threshold: int = 20,
        mode: str = "direct",
        ratio_threshold: float = 0.75,
        vocab_size: int = 512,
        max_image_size: Optional[int] = 1024,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_keypoints = n_keypoints
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.fast_threshold = fast_threshold
        self.mode = mode
        self.ratio_threshold = ratio_threshold
        self.vocab_size = vocab_size
        self.max_image_size = max_image_size
        self.seed = seed

        self.detector = cv2.ORB_create(
            nfeatures=n_keypoints,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            fastThreshold=fast_threshold,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.raw_descriptors: List[np.ndarray] = []
        self.medoids: Optional[np.ndarray] = None
        self.bovw_matrix: Optional[np.ndarray] = None
        self.vector_index: Optional[VectorIndex] = None

    def extract_descriptors(self, image: Any) -> np.ndarray:
        gray = _load_gray(image, self.max_image_size)
        _, desc = self.detector.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            return np.zeros((0, 32), dtype=np.uint8)
        return np.asarray(desc, dtype=np.uint8)

    def _assign_binary_words(self, descriptors: np.ndarray) -> np.ndarray:
        if len(descriptors) == 0 or self.medoids is None:
            return np.zeros((0,), dtype=np.int64)
        matches = self.matcher.match(descriptors, self.medoids)
        return np.array([m.trainIdx for m in matches], dtype=np.int64)

    def fit(self, database_images: Sequence[Any]) -> None:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.raw_descriptors = list(pool.map(self.extract_descriptors, database_images))
        self.feature_extraction_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if self.mode == "bovw_medoids":
            non_empty = [d for d in self.raw_descriptors if len(d) > 0]
            if not non_empty:
                self.medoids = np.zeros((self.vocab_size, 32), dtype=np.uint8)
                self.bovw_matrix = np.zeros((len(database_images), self.vocab_size), dtype=np.float32)
            else:
                all_desc = np.vstack(non_empty)
                rng = np.random.default_rng(self.seed)
                k = min(self.vocab_size, len(all_desc))
                idx = rng.choice(len(all_desc), size=k, replace=False)
                self.medoids = np.ascontiguousarray(all_desc[idx], dtype=np.uint8)

                hists = np.zeros((len(database_images), self.vocab_size), dtype=np.float32)
                for i, desc in enumerate(self.raw_descriptors):
                    if len(desc) > 0:
                        words = self._assign_binary_words(desc)
                        hists[i, :k] = np.bincount(words, minlength=k).astype(np.float32)
                self.bovw_matrix = l2_normalize(hists)
            self.vector_index = VectorIndex(metric="cosine")
            self.vector_index.build(self.bovw_matrix)
        else:
            self.bovw_matrix = None
            self.vector_index = None
        self.indexing_time_s = time.perf_counter() - t1

    def retrieve(self, query_image: Any, top_k: int = 10) -> List[Tuple[int, float]]:
        t0 = time.perf_counter()
        q_desc = self.extract_descriptors(query_image)
        self.last_query_feature_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if self.mode == "bovw_medoids":
            if self.vector_index is None or self.medoids is None:
                raise RuntimeError("ORBRetriever must be fitted before calling retrieve().")
            hist = np.zeros((self.vocab_size,), dtype=np.float32)
            if len(q_desc) > 0:
                words = self._assign_binary_words(q_desc)
                k = len(self.medoids)
                hist[:k] = np.bincount(words, minlength=k).astype(np.float32)
            results = self.vector_index.search(l2_normalize(hist), top_k=top_k)
            self.last_query_search_time_s = time.perf_counter() - t1
            return results

        scores = np.zeros((len(self.raw_descriptors),), dtype=np.float32)
        if len(q_desc) >= 2 and len(self.raw_descriptors) > 0:
            ratio = self.ratio_threshold
            denom = float(max(len(q_desc), 1))

            def _match_chunk(chunk: List[np.ndarray]) -> List[float]:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
                out: List[float] = []
                for db_desc in chunk:
                    if len(db_desc) < 2:
                        out.append(0.0)
                        continue
                    matches = matcher.knnMatch(q_desc, db_desc, k=2)
                    good = 0
                    for pair in matches:
                        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                            good += 1
                    out.append(float(good) / denom)
                return out

            n_db = len(self.raw_descriptors)
            if n_db > 100:
                chunk_size = max(1, (n_db + 7) // 8)
                chunks = [
                    self.raw_descriptors[i : i + chunk_size] for i in range(0, n_db, chunk_size)
                ]
                with ThreadPoolExecutor(max_workers=8) as pool:
                    chunk_scores = list(pool.map(_match_chunk, chunks))
                scores = np.array([s for sub in chunk_scores for s in sub], dtype=np.float32)
            else:
                scores = np.array(_match_chunk(self.raw_descriptors), dtype=np.float32)

        k = min(top_k, len(scores))
        order = np.argsort(-scores)[:k]
        self.last_query_search_time_s = time.perf_counter() - t1
        return [(int(i), float(scores[i])) for i in order]

    def get_representations(self) -> Any:
        if self.mode == "bovw_medoids":
            return self.bovw_matrix
        return self.raw_descriptors
