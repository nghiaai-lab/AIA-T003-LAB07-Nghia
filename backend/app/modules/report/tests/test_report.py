from datetime import datetime, timezone

from backend.app.modules.report.service import _env


def test_report_template_renders_with_empty_data():
    template = _env.get_template("report.html")

    html = template.render(
        project_id="demo",
        project_name="Demo",
        generated_at=datetime.now(timezone.utc).isoformat(),
        evaluations=[],
        correlation=None,
        failure_cases=[],
    )

    assert "Demo" in html
    assert "Chưa có kết quả evaluation" in html
