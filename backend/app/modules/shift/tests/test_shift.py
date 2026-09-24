from backend.app.modules.shift.router import router


def test_shift_health_route_registered():
    paths = [r.path for r in router.routes]
    assert "/shift/health" in paths
