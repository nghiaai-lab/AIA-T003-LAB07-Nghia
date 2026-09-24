from backend.app.modules.report.router import router


def test_report_health_route_registered():
    paths = [r.path for r in router.routes]
    assert "/report/health" in paths
