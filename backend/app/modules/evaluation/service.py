"""Business logic cho Performance Evaluation (Module C, toilatrung). Xem docs/spec.md 2.8."""
from sqlalchemy.orm import Session

from backend.app.modules.evaluation.metrics import compute_map
from backend.app.modules.evaluation.models import EvaluationResultORM
from backend.app.modules.evaluation.schemas import EvaluationRequest, EvaluationResult


def run_evaluation(db: Session, req: EvaluationRequest) -> EvaluationResult:
    target_metric, ap_per_class = compute_map(req.predictions, req.ground_truth, req.iou_threshold)
    performance_drop = req.source_metric - target_metric

    row = EvaluationResultORM(
        project_id=req.project_id,
        domain_or_slice=req.domain_or_slice,
        target_metric=target_metric,
        source_metric=req.source_metric,
        performance_drop=performance_drop,
        ap_per_class=ap_per_class,
        num_predictions=len(req.predictions),
        num_ground_truth=len(req.ground_truth),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return _to_schema(row)


def list_evaluations(db: Session, project_id: str) -> list[EvaluationResult]:
    rows = (
        db.query(EvaluationResultORM)
        .filter(EvaluationResultORM.project_id == project_id)
        .order_by(EvaluationResultORM.created_at.desc())
        .all()
    )
    return [_to_schema(r) for r in rows]


def _to_schema(row: EvaluationResultORM) -> EvaluationResult:
    return EvaluationResult(
        project_id=row.project_id,
        domain_or_slice=row.domain_or_slice,
        target_metric=row.target_metric,
        source_metric=row.source_metric,
        performance_drop=row.performance_drop,
        ap_per_class=row.ap_per_class,
        num_predictions=row.num_predictions,
        num_ground_truth=row.num_ground_truth,
    )
