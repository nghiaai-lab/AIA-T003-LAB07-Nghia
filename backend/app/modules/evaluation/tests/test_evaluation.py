from backend.app.modules.evaluation.router import router


def test_evaluation_health_route_registered():
    paths = [r.path for r in router.routes]
    assert "/evaluation/health" in paths
