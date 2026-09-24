"""Module C (toilatrung) — 2.10 False Alarm & Miss Explorer. Xem docs/spec.md."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.db import get_db
from backend.app.modules.failure import service
from backend.app.modules.failure.schemas import FailureAnalysisRequest, FailureAnalysisResult

router = APIRouter(prefix="/failure", tags=["failure"])


@router.post("/analyze", response_model=FailureAnalysisResult)
def analyze(req: FailureAnalysisRequest, db: Session = Depends(get_db)) -> FailureAnalysisResult:
    return service.analyze(db, req)


@router.get("/{project_id}", response_model=FailureAnalysisResult)
def list_cases(project_id: str, db: Session = Depends(get_db)) -> FailureAnalysisResult:
    return service.list_cases(db, project_id)
