"""Batch Data Ingestion Validation.
Checks for corrupted files, duplicates, dimension anomalies, small batches, and overlap with source data.
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from backend.app.core.schemas import BatchValidationResult

logger = logging.getLogger(__name__)


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hexadecimal digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_phash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a fast average perceptual hash (aHash) for detecting near-identical duplicates."""
    resized = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized).flatten()
    avg = pixels.mean()
    diff = pixels > avg
    bit_string = "".join(["1" if b else "0" for b in diff])
    hex_string = f"{int(bit_string, 2):0{hash_size * hash_size // 4}x}"
    return hex_string


class BatchValidator:
    """Validates files in a candidate target batch against quality and integrity criteria."""

    def __init__(
        self,
        min_batch_size: int = 5,
        min_dim: int = 32,
        max_aspect_ratio: float = 10.0,
        source_hashes: set[str] | None = None,
    ) -> None:
        self.min_batch_size = min_batch_size
        self.min_dim = min_dim
        self.max_aspect_ratio = max_aspect_ratio
        self.source_hashes = source_hashes or set()

    def validate_file_items(
        self,
        items: list[tuple[str, bytes | Path]],
    ) -> tuple[BatchValidationResult, list[tuple[str, Image.Image, str]]]:
        """Validate a list of image items (filename, bytes_or_path).

        Returns:
            tuple[BatchValidationResult, list[tuple[str, Image.Image, str]]]:
                - Validation summary report
                - Valid list of (image_id, pil_image, sha256_hash)
        """
        warnings: list[str] = []
        errors: list[str] = []
        valid_items: list[tuple[str, Image.Image, str]] = []
        seen_exact_hashes: set[str] = set()
        seen_phashes: set[str] = set()

        total_files = len(items)
        corrupted_count = 0
        duplicates_count = 0
        dimension_anomalies_count = 0
        source_overlap_count = 0

        if total_files == 0:
            errors.append("Batch is empty. No images provided.")
            return (
                BatchValidationResult(
                    is_valid=False,
                    total_files=0,
                    valid_images=0,
                    corrupted_images=0,
                    duplicates=0,
                    dimension_anomalies=0,
                    source_overlap_count=0,
                    warnings=warnings,
                    errors=errors,
                ),
                [],
            )

        if total_files < self.min_batch_size:
            warnings.append(
                f"Batch size ({total_files}) is smaller than recommended minimum ({self.min_batch_size}). "
                "Statistical shift estimates may have higher variance."
            )

        for filename, data_or_path in items:
            image_id = Path(filename).name
            raw_bytes: bytes

            if isinstance(data_or_path, Path):
                try:
                    raw_bytes = data_or_path.read_bytes()
                except Exception as e:
                    corrupted_count += 1
                    warnings.append(f"Cannot read file {filename}: {e}")
                    continue
            else:
                raw_bytes = data_or_path

            # 1. Exact hash check
            sha = compute_sha256(raw_bytes)
            if sha in seen_exact_hashes:
                duplicates_count += 1
                warnings.append(f"Duplicate image detected: {filename}")
                continue
            seen_exact_hashes.add(sha)

            # Check overlap with source domain
            if sha in self.source_hashes:
                source_overlap_count += 1
                warnings.append(f"Target image {filename} overlaps with reference source dataset.")

            # 2. Open image with PIL to check corruption & dimensions
            try:
                img = Image.open(io.BytesIO(raw_bytes))
                img.verify()  # verify integrity
                # Reopen for actual usage after verify()
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            except Exception as e:
                corrupted_count += 1
                warnings.append(f"Corrupted or invalid image file {filename}: {e}")
                continue

            width, height = img.size

            # 3. Dimension anomaly check
            if width < self.min_dim or height < self.min_dim:
                dimension_anomalies_count += 1
                warnings.append(
                    f"Image {filename} dimension ({width}x{height}) is unusually small (< {self.min_dim}px)."
                )

            aspect_ratio = max(width / max(height, 1), height / max(width, 1))
            if aspect_ratio > self.max_aspect_ratio:
                dimension_anomalies_count += 1
                warnings.append(
                    f"Image {filename} has extreme aspect ratio ({aspect_ratio:.1f}:1)."
                )

            # 4. Perceptual duplicate check
            try:
                phash = compute_phash(img)
                if phash in seen_phashes:
                    warnings.append(f"Near-identical perceptual duplicate detected: {filename}")
                seen_phashes.add(phash)
            except Exception:
                pass

            valid_items.append((image_id, img, sha))

        valid_count = len(valid_items)
        is_valid = valid_count > 0 and len(errors) == 0

        res = BatchValidationResult(
            is_valid=is_valid,
            total_files=total_files,
            valid_images=valid_count,
            corrupted_images=corrupted_count,
            duplicates=duplicates_count,
            dimension_anomalies=dimension_anomalies_count,
            source_overlap_count=source_overlap_count,
            warnings=warnings,
            errors=errors,
        )
        return res, valid_items
