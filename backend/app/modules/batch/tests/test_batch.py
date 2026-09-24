"""Tests for Module A: Target Batch Ingestion, Corruptions, Validation, and Inference."""
import io
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.core.schemas import TargetBatchIngestRequest
from backend.app.main import app
from backend.app.modules.batch.corruptions import (
    AVAILABLE_CORRUPTIONS,
    apply_corruption,
)
from backend.app.modules.batch.service import BatchService
from backend.app.modules.batch.validation import BatchValidator
from backend.app.modules.source.model_manager import ModelManager

client = TestClient(app)


def test_batch_health():
    resp = client.get("/batch/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_corruption_determinism():
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 100
    res1 = apply_corruption(test_img, corruption_type="gaussian_noise", seed=123)
    res2 = apply_corruption(test_img, corruption_type="gaussian_noise", seed=123)
    res3 = apply_corruption(test_img, corruption_type="gaussian_noise", seed=999)

    # Identical seed must yield identical result
    np.testing.assert_array_equal(res1, res2)
    # Different seed should yield different noise
    assert not np.array_equal(res1, res3)


def test_all_corruptions_runnable():
    test_img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    for c_type in AVAILABLE_CORRUPTIONS:
        corrupted = apply_corruption(test_img, corruption_type=c_type, seed=42)
        assert corrupted.shape == (64, 64, 3)
        assert corrupted.dtype == np.uint8


def test_batch_validator_detects_duplicates_and_corruptions():
    # Valid image
    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    valid_bytes = buf.getvalue()

    corrupt_bytes = b"NOT_AN_IMAGE_FILE_DATA_CORRUPTED"

    items = [
        ("img1.jpg", valid_bytes),
        ("img2_duplicate.jpg", valid_bytes),  # exact duplicate
        ("bad.jpg", corrupt_bytes),  # corrupted
    ]

    validator = BatchValidator(min_batch_size=1)
    res, valid_items = validator.validate_file_items(items)

    assert res.total_files == 3
    assert res.valid_images == 1
    assert res.duplicates == 1
    assert res.corrupted_images == 1
    assert len(valid_items) == 1


def test_synthetic_batch_generation_and_embeddings():
    req = TargetBatchIngestRequest(
        project_id="test_batch_proj",
        batch_name="Test Dark Target Batch",
        corruption_type="dark_severe",
        sample_count=8,
    )
    res = BatchService.ingest_synthetic_batch(req)

    assert res.batch_id.startswith("batch_")
    assert res.num_images == 8
    assert res.embedding_dim == 256
    assert res.confidence_mean >= 0.0
    assert res.ready_for_module_b is True

    # Test retrieval endpoints
    retrieved = client.get(f"/batch/test_batch_proj/{res.batch_id}")
    assert retrieved.status_code == 200
    assert retrieved.json()["total_images"] == 8

    # Test Module B handoff endpoint
    embs_res = client.get(f"/batch/test_batch_proj/{res.batch_id}/embeddings")
    assert embs_res.status_code == 200
    embs_data = embs_res.json()
    assert embs_data["num_embeddings"] == 8
    assert len(embs_data["embedding_records"]) == 8
    assert len(embs_data["embedding_records"][0]["embedding"]) == 256
