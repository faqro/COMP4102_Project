"""Pretrained DINOv2 Vision Transformer embedding extractor and retriever."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from torchvision import transforms
from transformers import AutoModel

from src.features.resnet import _resolve_device, _to_pil
from src.retrieval.index import ImageRetriever, VectorIndex
from src.retrieval.similarity import l2_normalize


class DINOv2Retriever(ImageRetriever):
    """Pretrained DINOv2 ViT-B/14 embedding retriever."""

    name = "dinov2"

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        input_size: int = 224,
        pooling: str = "cls",
        batch_size: int = 32,
        device: str = "auto",
        normalize: bool = True,
        pca_dim: Optional[int] = None,
        similarity: str = "cosine",
        use_faiss: bool = False,
        faiss_type: str = "flat",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = _resolve_device(device)
        self.normalize = normalize
        self.pca_dim = pca_dim
        self.similarity = similarity
        self.use_faiss = use_faiss
        self.faiss_type = faiss_type
        self.seed = seed

        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.pca: Optional[PCA] = None
        self.embeddings: Optional[np.ndarray] = None
        self.vector_index: Optional[VectorIndex] = None

    def _sync_if_cuda(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _pool_tokens(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        if self.pooling == "mean":
            return last_hidden_state[:, 1:].mean(dim=1)
        if self.pooling == "gem":
            patch_tokens = torch.clamp(last_hidden_state[:, 1:], min=1e-6)
            return patch_tokens.pow(3.0).mean(dim=1).pow(1.0 / 3.0)
        raise ValueError(f"Unknown pooling mode: {self.pooling}")

    @torch.no_grad()
    def _extract_batch(self, images: Sequence[Any]) -> np.ndarray:
        feats: List[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(images), self.batch_size):
                batch_imgs = images[start : start + self.batch_size]
                tensor_list = list(pool.map(lambda im: self.transform(_to_pil(im)), batch_imgs))
                tensors = torch.stack(tensor_list).to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    outputs = self.model(pixel_values=tensors, interpolate_pos_encoding=True)
                    pooled = self._pool_tokens(outputs.last_hidden_state)
                feats.append(pooled.detach().float().cpu().numpy().astype(np.float32))
        if not feats:
            return np.zeros((0, 768), dtype=np.float32)
        return np.vstack(feats)

    def fit(self, database_images: Sequence[Any]) -> None:
        self._sync_if_cuda()
        t0 = time.perf_counter()
        raw = self._extract_batch(database_images)
        self._sync_if_cuda()
        self.feature_extraction_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if self.normalize:
            raw = l2_normalize(raw)
        if self.pca_dim is not None and len(raw) >= self.pca_dim:
            self.pca = PCA(n_components=self.pca_dim, whiten=True, random_state=self.seed)
            raw = self.pca.fit_transform(raw).astype(np.float32)
            if self.normalize:
                raw = l2_normalize(raw)
        else:
            self.pca = None

        self.embeddings = np.ascontiguousarray(raw, dtype=np.float32)
        self.vector_index = VectorIndex(
            metric=self.similarity,
            use_faiss=self.use_faiss,
            faiss_type=self.faiss_type,
        )
        self.vector_index.build(self.embeddings)
        self.indexing_time_s = time.perf_counter() - t1

    @torch.no_grad()
    def retrieve(self, query_image: Any, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.vector_index is None:
            raise RuntimeError("DINOv2Retriever must be fitted before calling retrieve().")

        self._sync_if_cuda()
        t0 = time.perf_counter()
        tensor = self.transform(_to_pil(query_image)).unsqueeze(0).to(self.device)
        outputs = self.model(pixel_values=tensor, interpolate_pos_encoding=True)
        q_emb = self._pool_tokens(outputs.last_hidden_state).detach().cpu().numpy().astype(np.float32)[0]
        self._sync_if_cuda()
        self.last_query_feature_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if self.normalize:
            q_emb = l2_normalize(q_emb)
        if self.pca is not None:
            q_emb = self.pca.transform(q_emb.reshape(1, -1)).astype(np.float32)[0]
            if self.normalize:
                q_emb = l2_normalize(q_emb)
        results = self.vector_index.search(q_emb, top_k=top_k)
        self.last_query_search_time_s = time.perf_counter() - t1
        return results

    def get_representations(self) -> Any:
        return self.embeddings
