"""Synthetic Domain Shift Corruptions.
Provides deterministic, seed-fixed image transformations to simulate production domain shifts
(lighting, weather, sensor degradation, compression) without altering bounding box locations.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

CorruptionType = Literal[
    "normal",
    "dark_mild",
    "dark_severe",
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "jpeg_compression",
    "grayscale",
    "contrast_low",
    "contrast_high",
]

AVAILABLE_CORRUPTIONS: list[str] = [
    "normal",
    "dark_mild",
    "dark_severe",
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "jpeg_compression",
    "grayscale",
    "contrast_low",
    "contrast_high",
]


def apply_corruption(
    image: np.ndarray | Image.Image,
    corruption_type: str = "normal",
    seed: int = 42,
) -> np.ndarray:
    """Apply a deterministic photometric corruption to an image.

    Args:
        image: Input image as RGB numpy array (H, W, 3) or PIL Image.
        corruption_type: Name of corruption.
        seed: Random seed for deterministic reproducibility.

    Returns:
        np.ndarray: Corrupted image in RGB format uint8 (H, W, 3).
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.asarray(image).copy()

    # Ensure RGB uint8
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    rng = np.random.default_rng(seed)

    c_type = corruption_type.lower().strip()

    if c_type == "normal":
        return img_np

    elif c_type == "dark_mild":
        # Mild darkness (night twilight or underexposed sensor)
        factor = 0.6
        out = np.clip(img_np.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return out

    elif c_type == "dark_severe":
        # Severe darkness (nighttime low-light camera)
        factor = 0.25
        out = np.clip(img_np.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return out

    elif c_type == "gaussian_blur":
        # Defocus or lens fog
        ksize = 19
        out = cv2.GaussianBlur(img_np, (ksize, ksize), sigmaX=6.0)
        return out

    elif c_type == "motion_blur":
        # Moving camera or fast object vehicle motion
        size = 15
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[int((size - 1) / 2), :] = np.ones(size, dtype=np.float32)
        kernel /= size
        out = cv2.filter2D(img_np, -1, kernel)
        return out

    elif c_type == "gaussian_noise":
        # Low-light ISO sensor noise
        sigma = 35.0
        noise = rng.normal(0, sigma, img_np.shape).astype(np.float32)
        noisy = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    elif c_type == "jpeg_compression":
        # Bandwidth throttling / streaming compression artifacts
        quality = 18
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        # Convert RGB to BGR for cv2 imencode
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode(".jpg", bgr, encode_param)
        dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)

    elif c_type == "grayscale":
        # IR / monochrome night sensor
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    elif c_type == "contrast_low":
        # Fog / haze reducing dynamic range
        alpha = 0.4
        beta = 70
        out = np.clip(img_np.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return out

    elif c_type == "contrast_high":
        # Harsh glare / extreme sunny lighting
        alpha = 1.8
        beta = -50
        out = np.clip(img_np.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return out

    else:
        # Fallback to normal if unknown
        return img_np


def save_corrupted_image(
    src_path: Path | str,
    dest_path: Path | str,
    corruption_type: str = "normal",
    seed: int = 42,
) -> Path:
    """Read an image file, apply corruption deterministically, and save to destination path."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.open(src_path).convert("RGB")
    corrupted_np = apply_corruption(pil_img, corruption_type=corruption_type, seed=seed)
    Image.fromarray(corrupted_np).save(dest, format="JPEG", quality=95)
    return dest
