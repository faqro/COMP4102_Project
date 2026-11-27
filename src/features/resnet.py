"""Pretrained ResNet-50 CNN embedding extractor and retriever."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import models, transforms

from src.retrieval.index import ImageRetriever, VectorIndex
from src.retrieval.similarity import l2_normalize


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_pil(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


class ResNetRetriever(ImageRetriever):
    """Pretrained ResNet-50 global average-pooled embedding retriever."""

    name = "resnet"

    def __init__(
        self,
        input_size: int = 224,
        batch_size: int = 64,
        device: str = "auto",
        normalize: bool = True,
        pca_dim: Optional[int] = None,
        similarity: str = "cosine",
        use_faiss: bool = False,
        faiss_type: str = "flat",
        weights: str = "IMAGENET1K_V2",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = _resolve_device(device)
        self.normalize = normalize
        self.pca_dim = pca_dim
        self.similarity = similarity
        self.use_faiss = use_faiss
        self.faiss_type = faiss_type
        self.seed = seed

        weight_enum = (
            models.ResNet50_Weights.IMAGENET1K_V2
            if weights == "IMAGENET1K_V2"
            else models.ResNet50_Weights.IMAGENET1K_V1
        )
        backbone = models.resnet50(weights=weight_enum)
        backbone.fc = nn.Identity()
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.model = backbone.to(self.device)

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

    @torch.no_grad()
    def _extract_batch(self, images: Sequence[Any]) -> np.ndarray:
        feats: List[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(images), self.batch_size):
                batch_imgs = images[start : start + self.batch_size]
                tensor_list = list(pool.map(lambda im: self.transform(_to_pil(im)), batch_imgs))
                tensors = torch.stack(tensor_list).to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    out = self.model(tensors)
                feats.append(out.detach().float().cpu().numpy().astype(np.float32))
        if not feats:
            return np.zeros((0, 2048), dtype=np.float32)
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
            raise RuntimeError("ResNetRetriever must be fitted before calling retrieve().")

        self._sync_if_cuda()
        t0 = time.perf_counter()
        tensor = self.transform(_to_pil(query_image)).unsqueeze(0).to(self.device)
        q_emb = self.model(tensor).detach().cpu().numpy().astype(np.float32)[0]
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
