import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.modules.slice.router import router
from backend.app.modules.slice import service
from backend.app.modules.slice.schemas import SliceAnalysisRequest, TargetItem


def test_slice_health_route_registered():
    paths = [r.path for r in router.routes]
    assert "/slice/health" in paths
    assert "/slice/analyze" in paths
    assert "/slice/export-records" in paths
    assert "/slice/mock-demo" in paths


def test_mock_batch_data_generator():
    source_embs, source_confs, target_items = service.generate_mock_batch_data()
    assert len(source_embs) > 50
    assert len(target_items) > 100
    assert target_items[0].embedding is not None
    assert "time" in target_items[0].metadata


def test_slice_analysis_logic():
    source_embs, source_confs, target_items = service.generate_mock_batch_data(seed=123)
    req = SliceAnalysisRequest(
        source_embeddings=source_embs,
        source_confidences=source_confs,
        target_items=target_items,
        min_slice_size=5,
    )
    result = service.analyze_target_slices(req)
    assert result.total_target_images == len(target_items)
    assert result.total_slices > 0
    assert len(result.slices) > 0
    # Verified slices are sorted descending by shift score
    scores = [s.shift_score for s in result.slices]
    assert scores == sorted(scores, reverse=True)


def test_export_to_shift_score_records():
    source_embs, source_confs, target_items = service.generate_mock_batch_data()
    req = SliceAnalysisRequest(
        source_embeddings=source_embs,
        source_confidences=source_confs,
        target_items=target_items,
    )
    analysis = service.analyze_target_slices(req)
    records = service.export_to_shift_score_records(analysis)
    assert len(records) > 0
    assert records[0].domain_or_slice == "overall_batch"
    assert hasattr(records[0], "shift_score")
    assert hasattr(records[0], "num_images")


def test_slice_api_endpoints():
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/slice/health")
    assert res_health.status_code == 200

    # 2. Mock demo endpoint
    res_demo = client.get("/slice/mock-demo")
    assert res_demo.status_code == 200
    data = res_demo.json()
    assert data["total_slices"] > 0
    assert "overall_shift_score" in data
    assert "highest_risk_slice" in data

    # 3. Export records endpoint
    res_export = client.post("/slice/export-records", json=data)
    assert res_export.status_code == 200
    records = res_export.json()
    assert isinstance(records, list)
    assert len(records) > 0
    assert "shift_score" in records[0]
