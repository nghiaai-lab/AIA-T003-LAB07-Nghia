from backend.app.modules.source.router import router


def test_source_health_route_registered():
    paths = [r.path for r in router.routes]
    assert "/source/health" in paths
