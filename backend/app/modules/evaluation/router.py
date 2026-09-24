"""Module C (toilatrung) — 2.8 Ground Truth & Performance Evaluation. Xem docs/spec.md."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.db import get_db
from backend.app.modules.evaluation import service
from backend.app.modules.evaluation.schemas import EvaluationRequest, EvaluationResult

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationResult)
def run(req: EvaluationRequest, db: Session = Depends(get_db)) -> EvaluationResult:
    return service.run_evaluation(db, req)


@router.get("/{project_id}", response_model=list[EvaluationResult])
def list_results(project_id: str, db: Session = Depends(get_db)) -> list[EvaluationResult]:
    return service.list_evaluations(db, project_id)
