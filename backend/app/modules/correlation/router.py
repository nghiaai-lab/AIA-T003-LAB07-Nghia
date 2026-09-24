"""Module C (toilatrung) — 2.9 Correlation Validator. Xem docs/spec.md."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.db import get_db
from backend.app.modules.correlation import service
from backend.app.modules.correlation.schemas import CorrelationRequest, CorrelationResult

router = APIRouter(prefix="/correlation", tags=["correlation"])


@router.post("/compute", response_model=CorrelationResult)
def compute(req: CorrelationRequest, db: Session = Depends(get_db)) -> CorrelationResult:
    try:
        return service.compute_correlation(db, req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}", response_model=CorrelationResult)
def latest(project_id: str, db: Session = Depends(get_db)) -> CorrelationResult:
    result = service.get_latest_correlation(db, project_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Chưa có correlation result cho project này")
    return result
