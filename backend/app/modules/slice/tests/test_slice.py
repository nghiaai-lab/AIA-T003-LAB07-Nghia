from backend.app.modules.slice.router import router


def test_slice_health_route_registered():
    paths = [r.path for r in router.routes]
    assert "/slice/health" in paths
