"""Module C (toilatrung) — 2.11 Report Export. Xem docs/spec.md."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.core.db import get_db
from backend.app.modules.report import service
from backend.app.modules.report.schemas import ReportRequest, ReportResponse

router = APIRouter(prefix="/report", tags=["report"])


@router.post("/export", response_model=ReportResponse)
def export(req: ReportRequest, db: Session = Depends(get_db)) -> ReportResponse:
    return service.export_report(db, req)


@router.get("/{project_id}/download")
def download(project_id: str) -> FileResponse:
    path = service.report_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report chưa được export cho project này")
    return FileResponse(path, media_type="text/html", filename=path.name)
