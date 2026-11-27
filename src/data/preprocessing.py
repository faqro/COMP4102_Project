"""Image loading, resizing, and robustness transformation utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def pil_to_gray(img: Image.Image, max_size: int | None = None) -> np.ndarray:
    if max_size is not None:
        img = resize_max_side(img, max_size)
    rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def resize_max_side(img: Image.Image, max_size: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        return img
    scale = max_size / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


def apply_rotation(img: Image.Image, angle: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    mat[0, 2] += (new_w / 2.0) - center[0]
    mat[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(
        arr, mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return Image.fromarray(rotated)


def apply_gaussian_blur(img: Image.Image, kernel_size: int) -> Image.Image:
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    arr = np.asarray(img.convert("RGB"))
    blurred = cv2.GaussianBlur(arr, (k, k), 0)
    return Image.fromarray(blurred)


def apply_gaussian_noise(img: Image.Image, sigma: float, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    noise = rng.normal(0.0, float(sigma), size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img.convert("RGB"))
    return enhancer.enhance(float(factor))


def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_center_crop(img: Image.Image, retain_area_ratio: float) -> Image.Image:
    w, h = img.size
    side_scale = float(np.sqrt(retain_area_ratio))
    crop_w = max(1, int(round(w * side_scale)))
    crop_h = max(1, int(round(h * side_scale)))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


TRANSFORMATION_SPECS: List[Tuple[str, str, float]] = [
    ("none", "0", 0.0),
    ("rotation", "15", 15.0),
    ("rotation", "30", 30.0),
    ("rotation", "45", 45.0),
    ("blur", "k3", 3.0),
    ("blur", "k5", 5.0),
    ("blur", "k9", 9.0),
    ("noise", "sigma10", 10.0),
    ("noise", "sigma25", 25.0),
    ("noise", "sigma50", 50.0),
    ("brightness", "0.50", 0.50),
    ("brightness", "0.75", 0.75),
    ("brightness", "1.25", 1.25),
    ("jpeg", "q20", 20.0),
    ("jpeg", "q50", 50.0),
    ("jpeg", "q80", 80.0),
    ("crop", "90pct", 0.90),
    ("crop", "70pct", 0.70),
    ("crop", "50pct", 0.50),
]


def transform_image(img: Image.Image, transform_type: str, param: float, seed: int = 42) -> Image.Image:
    if transform_type == "none":
        return img.convert("RGB")
    if transform_type == "rotation":
        return apply_rotation(img, param)
    if transform_type == "blur":
        return apply_gaussian_blur(img, int(param))
    if transform_type == "noise":
        return apply_gaussian_noise(img, param, seed=seed)
    if transform_type == "brightness":
        return apply_brightness(img, param)
    if transform_type == "jpeg":
        return apply_jpeg_compression(img, int(param))
    if transform_type == "crop":
        return apply_center_crop(img, param)
    raise ValueError(f"Unknown transform_type: {transform_type}")


def generate_transformed_queries(
    dataset,
    output_dir: str | Path = "data/transformed",
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """Materialize transformed query images to disk for reproducibility."""
    output_dir = Path(output_dir)
    saved: Dict[str, List[Path]] = {}
    for t_type, level, param in TRANSFORMATION_SPECS:
        if t_type == "none":
            continue
        key = f"{t_type}_{level}"
        sub_dir = output_dir / key
        sub_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for q in dataset.queries:
            out_path = sub_dir / f"{q.query_id:02d}_{q.name}.png"
            if not out_path.exists():
                q_img = dataset.get_query_image(q.query_id)
                t_img = transform_image(q_img, t_type, param, seed=seed + q.query_id)
                t_img.save(out_path)
            paths.append(out_path)
        saved[key] = paths
    return saved
