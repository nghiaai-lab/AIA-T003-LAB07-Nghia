"""Report Export (Module C, toilatrung) — gộp kết quả evaluation/correlation/failure
thành 1 báo cáo HTML. Xem docs/spec.md 2.11.
"""
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.modules.correlation import service as correlation_service
from backend.app.modules.evaluation import service as evaluation_service
from backend.app.modules.failure import service as failure_service
from backend.app.modules.report.schemas import ReportRequest, ReportResponse

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(["html"]),
)


def report_path(project_id: str) -> Path:
    return settings.data_dir / "artifacts" / f"{project_id}_report.html"


def export_report(db: Session, req: ReportRequest) -> ReportResponse:
    evaluations = evaluation_service.list_evaluations(db, req.project_id)
    correlation = correlation_service.get_latest_correlation(db, req.project_id)
    failures = failure_service.list_cases(db, req.project_id)

    template = _env.get_template("report.html")
    html = template.render(
        project_id=req.project_id,
        project_name=req.project_name or req.project_id,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        evaluations=evaluations,
        correlation=correlation,
        failure_cases=failures.cases,
    )

    out_path = report_path(req.project_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    return ReportResponse(project_id=req.project_id, file_path=str(out_path))
