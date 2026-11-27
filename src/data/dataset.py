"""ROxford5k dataset downloader, loader, and statistics utilities."""

from __future__ import annotations

import json
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

ROXFORD_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz"
ROXFORD_GND_URL = "http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl"


@dataclass
class QueryItem:
    query_id: int
    name: str
    image_path: Path
    bbox: Tuple[float, float, float, float]
    easy: np.ndarray
    hard: np.ndarray
    junk: np.ndarray


class _DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, filename=str(dest), reporthook=t.update_to)
    return dest


def download_roxford5k(root: str | Path = "data/raw/roxford5k") -> Path:
    """Download and extract ROxford5k images and ground-truth annotations."""
    root = Path(root)
    images_dir = root / "jpg"
    gnd_path = root / "gnd_roxford5k.pkl"

    root.mkdir(parents=True, exist_ok=True)
    download_file(ROXFORD_GND_URL, gnd_path)

    if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) < 4900:
        images_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            import pyarrow.parquet as pq

            snap_dir = Path(
                snapshot_download(
                    repo_id="vikhyatk/revisitop",
                    repo_type="dataset",
                    allow_patterns="roxford5k/*.parquet",
                )
            )
            parquet_files = sorted((snap_dir / "roxford5k").glob("*.parquet"))
            for pf in tqdm(parquet_files, desc="Extracting ROxford5k from parquet"):
                table = pq.read_table(pf, columns=["image", "filename"])
                images_col = table["image"].to_pylist()
                names_col = table["filename"].to_pylist()
                for img_dict, name in zip(images_col, names_col):
                    out_file = images_dir / f"{name}.jpg"
                    if not out_file.exists():
                        out_file.write_bytes(img_dict["bytes"])
        except Exception:
            tgz_path = root / "oxbuild_images-v1.tgz"
            download_file(ROXFORD_IMAGES_URL, tgz_path)
            with tarfile.open(tgz_path, "r:gz") as tar:
                members = [
                    m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".jpg")
                ]
                for m in tqdm(members, desc="Extracting ROxford5k"):
                    m.name = Path(m.name).name
                    tar.extract(m, path=images_dir)

    return root


class ROxford5kDataset:
    """Loader for the Revisited Oxford5k (ROxford5k) landmark retrieval benchmark."""

    def __init__(
        self,
        root: str | Path = "data/raw/roxford5k",
        processed_dir: str | Path = "data/processed",
        crop_queries: bool = True,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "jpg"
        self.gnd_path = self.root / "gnd_roxford5k.pkl"
        self.processed_dir = Path(processed_dir)
        self.crop_queries = crop_queries
        self.seed = seed

        if not self.gnd_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found at {self.gnd_path}. Run download_roxford5k() first."
            )

        with open(self.gnd_path, "rb") as f:
            cfg = pickle.load(f)

        self.imlist: List[str] = list(cfg["imlist"])
        self.qimlist: List[str] = list(cfg["qimlist"])
        self.gallery_paths: List[Path] = [self._resolve_image(name) for name in self.imlist]

        self.queries: List[QueryItem] = []
        for idx, (qname, qgnd) in enumerate(zip(self.qimlist, cfg["gnd"])):
            bbx = tuple(float(v) for v in qgnd["bbx"])
            self.queries.append(
                QueryItem(
                    query_id=idx,
                    name=qname,
                    image_path=self._resolve_image(qname),
                    bbox=bbx,
                    easy=np.array(qgnd["easy"], dtype=np.int64),
                    hard=np.array(qgnd["hard"], dtype=np.int64),
                    junk=np.array(qgnd["junk"], dtype=np.int64),
                )
            )

        self.val_ids, self.test_ids = self._load_or_create_split()

    def _resolve_image(self, name: str) -> Path:
        p = self.images_dir / f"{name}.jpg"
        if not p.exists():
            raise FileNotFoundError(f"Missing image file: {p}")
        return p

    def _load_or_create_split(self, val_ratio: float = 0.3) -> Tuple[List[int], List[int]]:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        split_file = self.processed_dir / "roxford5k_split.json"
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["val_query_ids"], data["test_query_ids"]

        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(self.queries))
        rng.shuffle(indices)
        n_val = int(round(len(indices) * val_ratio))
        val_ids = sorted(indices[:n_val].tolist())
        test_ids = sorted(indices[n_val:].tolist())

        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": self.seed,
                    "n_gallery": len(self.gallery_paths),
                    "n_queries": len(self.queries),
                    "val_query_ids": val_ids,
                    "test_query_ids": test_ids,
                    "val_query_names": [self.queries[i].name for i in val_ids],
                    "test_query_names": [self.queries[i].name for i in test_ids],
                },
                f,
                indent=2,
            )
        return val_ids, test_ids

    def get_query_image(self, query_id: int) -> Image.Image:
        q = self.queries[query_id]
        img = Image.open(q.image_path).convert("RGB")
        if self.crop_queries:
            x1, y1, x2, y2 = q.bbox
            w, h = img.size
            box = (
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                min(w, int(round(x2))),
                min(h, int(round(y2))),
            )
            if box[2] > box[0] and box[3] > box[1]:
                img = img.crop(box)
        return img

    def get_gallery_image(self, idx: int) -> Image.Image:
        return Image.open(self.gallery_paths[idx]).convert("RGB")

    def compute_statistics(self, save: bool = True) -> Dict[str, object]:
        """Compute and optionally save dataset dimension and relevance statistics."""
        widths: List[int] = []
        heights: List[int] = []
        for p in self.gallery_paths:
            with Image.open(p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)

        easy_counts = [len(q.easy) for q in self.queries]
        hard_counts = [len(q.hard) for q in self.queries]
        junk_counts = [len(q.junk) for q in self.queries]

        stats: Dict[str, object] = {
            "dataset": "ROxford5k",
            "n_gallery": len(self.gallery_paths),
            "n_queries": len(self.queries),
            "n_val_queries": len(self.val_ids),
            "n_test_queries": len(self.test_ids),
            "width_min": int(np.min(widths)),
            "width_max": int(np.max(widths)),
            "width_mean": float(np.mean(widths)),
            "height_min": int(np.min(heights)),
            "height_max": int(np.max(heights)),
            "height_mean": float(np.mean(heights)),
            "easy_positives_mean": float(np.mean(easy_counts)),
            "hard_positives_mean": float(np.mean(hard_counts)),
            "junk_mean": float(np.mean(junk_counts)),
        }
        if save:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            with open(self.processed_dir / "roxford5k_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        return stats
