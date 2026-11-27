"""Common ImageRetriever interface and vector indexing (exact + FAISS)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from src.retrieval.similarity import compute_similarity, l2_normalize


class ImageRetriever(ABC):
    """Unified interface for all classical and deep image retrieval methods."""

    name: str = "base"

    def __init__(self) -> None:
        self.feature_extraction_time_s: float = 0.0
        self.indexing_time_s: float = 0.0
        self.last_query_feature_time_s: float = 0.0
        self.last_query_search_time_s: float = 0.0

    @abstractmethod
    def fit(self, database_images: Sequence[Any]) -> None:
        """Extract and store database representations."""

    @abstractmethod
    def retrieve(self, query_image: Any, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return the top-k most similar (gallery_index, score) pairs."""

    @abstractmethod
    def get_representations(self) -> Any:
        """Return stored image representations."""

    def memory_bytes(self) -> int:
        """Return memory footprint in bytes of stored database representations."""
        reps = self.get_representations()
        if isinstance(reps, np.ndarray):
            return int(reps.nbytes)
        if isinstance(reps, list):
            total = 0
            for item in reps:
                if isinstance(item, np.ndarray):
                    total += int(item.nbytes)
            return total
        return 0


class VectorIndex:
    """Vector index supporting exact NumPy search and FAISS exact/approximate search."""

    def __init__(
        self,
        metric: str = "cosine",
        use_faiss: bool = False,
        faiss_type: str = "flat",
        nlist: int = 64,
        nprobe: int = 8,
    ) -> None:
        self.metric = metric
        self.use_faiss = use_faiss
        self.faiss_type = faiss_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.vectors: Optional[np.ndarray] = None
        self._faiss_index: Optional[faiss.Index] = None

    def build(self, vectors: np.ndarray) -> None:
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric == "cosine":
            arr = np.ascontiguousarray(l2_normalize(arr), dtype=np.float32)
        self.vectors = arr

        if not self.use_faiss:
            self._faiss_index = None
            return

        dim = arr.shape[1]
        if self.faiss_type == "ivf" and arr.shape[0] >= self.nlist:
            quantizer = (
                faiss.IndexFlatIP(dim) if self.metric == "cosine" else faiss.IndexFlatL2(dim)
            )
            metric_type = (
                faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
            )
            index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, metric_type)
            index.train(arr)
            index.add(arr)
            index.nprobe = min(self.nprobe, self.nlist)
            self._faiss_index = index
        else:
            index = faiss.IndexFlatIP(dim) if self.metric == "cosine" else faiss.IndexFlatL2(dim)
            index.add(arr)
            self._faiss_index = index

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.vectors is None or len(self.vectors) == 0:
            return []

        k = min(top_k, len(self.vectors))
        q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
        if self.metric == "cosine":
            q = np.ascontiguousarray(l2_normalize(q), dtype=np.float32)

        if self.use_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q, k)
            out: List[Tuple[int, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                val = float(score) if self.metric == "cosine" else float(-np.sqrt(max(score, 0.0)))
                out.append((int(idx), val))
            return out

        sims = compute_similarity(q[0], self.vectors, metric=self.metric)
        if k == len(sims):
            order = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k - 1)[:k]
            order = part[np.argsort(-sims[part])]
        return [(int(i), float(sims[i])) for i in order]
