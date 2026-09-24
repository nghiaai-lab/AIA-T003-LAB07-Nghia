from backend.app.modules.batch.router import router


def test_batch_health_route_registered():
    paths = [r.path for r in router.routes]
    assert "/batch/health" in paths
