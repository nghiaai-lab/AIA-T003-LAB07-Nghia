from backend.app.modules.failure.router import router


def test_failure_health_route_registered():
    paths = [r.path for r in router.routes]
    assert "/failure/health" in paths
