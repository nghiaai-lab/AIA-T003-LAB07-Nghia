"""Tests for Module A: Project Setup, Model Management, and Source Baseline."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.artifacts import ArtifactStore
from backend.app.core.schemas import ProjectConfig, SourceRegisterRequest
from backend.app.main import app
from backend.app.modules.source.model_manager import ModelManager
from backend.app.modules.source.service import SourceService

client = TestClient(app)


def test_source_health():
    resp = client.get("/source/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_project_save_and_retrieve(tmp_path):
    store = ArtifactStore(base_dir=tmp_path)
    config_dict = {
        "project_id": "test_proj_01",
        "project_name": "Test Radar Project",
        "task_type": "object_detection",
        "yolo_checkpoint": "yolo26n.pt",
        "performance_metric": "mAP50-95",
        "risk_threshold": 0.15,
        "embedding_layer": "second-to-last",
        "classes": ["car", "truck"],
        "device": "cpu",
    }
    store.save_project_config("test_proj_01", config_dict)
    loaded = store.load_project_config("test_proj_01")
    assert loaded is not None
    assert loaded["project_name"] == "Test Radar Project"
    assert loaded["risk_threshold"] == 0.15


def test_project_endpoints():
    payload = {
        "project_id": "integration_proj",
        "project_name": "Integration Project",
        "task_type": "object_detection",
        "yolo_checkpoint": "yolo26n.pt",
        "performance_metric": "mAP50-95",
        "risk_threshold": 0.10,
        "embedding_layer": "second-to-last",
        "classes": [],
        "device": "cpu",
    }
    res_save = client.post("/source/project", json=payload)
    assert res_save.status_code == 200
    saved_data = res_save.json()
    assert saved_data["project_id"] == "integration_proj"
    assert len(saved_data["classes"]) > 0  # auto-populated from YOLO model

    res_get = client.get("/source/project/integration_proj")
    assert res_get.status_code == 200
    assert res_get.json()["project_name"] == "Integration Project"


def test_model_manager_device_resolution():
    assert ModelManager.resolve_device("cpu") == "cpu"
    # Auto resolves to cpu or cuda:0 depending on environment
    res = ModelManager.resolve_device("auto")
    assert res in ("cpu", "cuda:0")


def test_model_manager_fallback():
    # Attempting to load an invalid name should fall back to valid fallback
    model, used_name = ModelManager.load_model(
        checkpoint="non_existent_yolo_model_9999.pt",
        fallback_checkpoint="yolo26n.pt",
    )
    assert model is not None
    assert used_name == "yolo26n.pt"


def test_embedding_extraction_properties():
    model, _ = ModelManager.load_model("yolo26n.pt", device="cpu")
    dummy_imgs = [
        np.zeros((320, 320, 3), dtype=np.uint8),
        np.ones((320, 320, 3), dtype=np.uint8) * 128,
    ]
    embeddings = ModelManager.extract_embeddings(model, dummy_imgs)

    # 1. Check shape
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 256

    # 2. Check no NaN or Inf
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()

    # 3. Check L2-normalization (unit length approx 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)


def test_source_baseline_computation(tmp_path):
    # Setup test project
    proj_cfg = ProjectConfig(
        project_id="test_baseline_proj",
        project_name="Test Baseline Project",
        yolo_checkpoint="yolo26n.pt",
        device="cpu",
    )
    SourceService.save_project(proj_cfg)

    # Compute baseline for coco128
    req = SourceRegisterRequest(
        project_id="test_baseline_proj",
        source_name="COCO128_Test",
        dataset_path="coco128",
    )
    resp = SourceService.compute_source_baseline(req)

    assert resp.num_images > 0
    assert resp.embedding_dim == 256
    assert resp.confidence_mean >= 0.0
    assert resp.num_reference_batches > 0
    assert resp.mAP50_95 is not None

    # Test reference data retrieval for Module B
    ref_data = SourceService.get_source_reference_data("test_baseline_proj")
    assert ref_data is not None
    assert ref_data.total_embeddings == resp.num_images
    assert ref_data.num_bootstrap_batches == resp.num_reference_batches
