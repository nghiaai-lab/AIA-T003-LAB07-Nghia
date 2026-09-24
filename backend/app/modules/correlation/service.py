"""Business logic cho Correlation Validator (Module C, toilatrung). Xem docs/spec.md 2.9."""
from scipy import stats
from sqlalchemy.orm import Session

from backend.app.modules.correlation.models import CorrelationResultORM
from backend.app.modules.correlation.schemas import (
    MIN_RECORDS,
    CorrelationInputRecord,
    CorrelationRequest,
    CorrelationResult,
)


def _compute_spearman(records: list[CorrelationInputRecord]) -> tuple[float, float]:
    shift_scores = [r.shift_score for r in records]
    perf_drops = [r.performance_drop for r in records]
    rho, p_value = stats.spearmanr(shift_scores, perf_drops)
    return float(rho), float(p_value)


def compute_correlation(db: Session, req: CorrelationRequest) -> CorrelationResult:
    if len(req.records) < MIN_RECORDS:
        raise ValueError(f"Cần tối thiểu {MIN_RECORDS} domain/slice để tính Spearman correlation")

    rho, p_value = _compute_spearman(req.records)

    row = CorrelationResultORM(
        project_id=req.project_id,
        spearman_rho=rho,
        p_value=p_value,
        n=len(req.records),
        records=[r.model_dump() for r in req.records],
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return CorrelationResult(
        project_id=row.project_id,
        spearman_rho=row.spearman_rho,
        p_value=row.p_value,
        n=row.n,
        records=req.records,
    )


def get_latest_correlation(db: Session, project_id: str) -> CorrelationResult | None:
    row = (
        db.query(CorrelationResultORM)
        .filter(CorrelationResultORM.project_id == project_id)
        .order_by(CorrelationResultORM.created_at.desc())
        .first()
    )
    if row is None:
        return None
    return CorrelationResult(
        project_id=row.project_id,
        spearman_rho=row.spearman_rho,
        p_value=row.p_value,
        n=row.n,
        records=[CorrelationInputRecord(**r) for r in row.records],
    )
