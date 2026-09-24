import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.modules.shift.router import router
from backend.app.modules.shift import service


def test_shift_health_route_registered():
    paths = [r.path for r in router.routes]
    assert "/shift/health" in paths
    assert "/shift/calculate" in paths


def test_mmd_computation():
    rng = np.random.default_rng(42)
    # Similar distributions
    X = rng.normal(0.0, 1.0, size=(100, 16))
    Y_similar = rng.normal(0.05, 1.0, size=(100, 16))
    dist_sim = service.compute_mmd(X, Y_similar)

    # Shifted distribution
    Y_shifted = rng.normal(2.5, 1.0, size=(100, 16))
    dist_shift = service.compute_mmd(X, Y_shifted)

    assert dist_shift > dist_sim
    assert dist_sim >= 0.0


def test_frechet_computation():
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(50, 8))
    Y = rng.normal(1.5, 1.0, size=(50, 8))
    fd = service.compute_frechet_distance(X, Y)
    assert fd > 0.0


def test_confidence_shift_detection():
    # Source has high confidence, Target has low confidence
    s_conf = [0.90, 0.92, 0.88, 0.95, 0.89] * 10
    t_conf = [0.40, 0.45, 0.35, 0.50, 0.42] * 10
    score, details = service.compute_confidence_shift(s_conf, t_conf)
    assert score > 60.0
    assert details["confidence_drop"] > 0.4


def test_risk_classification():
    assert service.classify_risk_level(20.0) == "Normal"
    assert service.classify_risk_level(70.0) == "Watch"
    assert service.classify_risk_level(85.0) == "High risk"
    assert service.classify_risk_level(95.0) == "Critical"


def test_shift_calculate_api():
    client = TestClient(app)
    rng = np.random.default_rng(42)
    source = rng.normal(0.0, 1.0, size=(40, 8)).tolist()
    target = rng.normal(2.0, 1.0, size=(40, 8)).tolist()

    response = client.post(
        "/shift/calculate",
        json={
            "source_embeddings": source,
            "target_embeddings": target,
            "source_confidences": [0.9] * 40,
            "target_confidences": [0.5] * 40,
            "method": "mmd",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "overall_shift_score" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["High risk", "Critical"]
    assert data["num_source_samples"] == 40
    assert data["num_target_samples"] == 40
