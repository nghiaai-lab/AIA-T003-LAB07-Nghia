"""Business logic cho False Alarm & Miss Explorer (Module C, toilatrung). Xem docs/spec.md 2.10."""
from sqlalchemy.orm import Session

from backend.app.modules.failure.models import FailureCaseORM
from backend.app.modules.failure.schemas import (
    FailureAnalysisRequest,
    FailureAnalysisResult,
    FailureCase,
    FailureThresholds,
    FailureType,
)

FALSE_ALARM_NOTE = (
    "Shift score cao nhưng performance drop thấp — có thể do thay đổi không ảnh hưởng "
    "task (màu sắc, sensor...), feature phục vụ nhận diện vẫn đủ tốt."
)
MISS_NOTE = (
    "Shift score thấp nhưng performance drop cao — global embedding hiện tại có thể "
    "không nhạy với slice/nhóm object này."
)
OK_NOTE = "Shift score và performance drop nhất quán với nhau."


def _classify(shift_score: float, performance_drop: float, t: FailureThresholds) -> tuple[FailureType, str]:
    if shift_score >= t.shift_high and performance_drop <= t.drop_low:
        return FailureType.FALSE_ALARM, FALSE_ALARM_NOTE
    if shift_score <= t.shift_low and performance_drop >= t.drop_high:
        return FailureType.MISS, MISS_NOTE
    return FailureType.OK, OK_NOTE


def analyze(db: Session, req: FailureAnalysisRequest) -> FailureAnalysisResult:
    cases: list[FailureCase] = []
    for r in req.records:
        failure_type, note = _classify(r.shift_score, r.performance_drop, req.thresholds)
        cases.append(
            FailureCase(
                domain_or_slice=r.domain_or_slice,
                shift_score=r.shift_score,
                performance_drop=r.performance_drop,
                failure_type=failure_type,
                note=note,
            )
        )
        db.add(
            FailureCaseORM(
                project_id=req.project_id,
                domain_or_slice=r.domain_or_slice,
                shift_score=r.shift_score,
                performance_drop=r.performance_drop,
                failure_type=failure_type.value,
                note=note,
            )
        )
    db.commit()
    return FailureAnalysisResult(project_id=req.project_id, cases=cases)


def list_cases(db: Session, project_id: str) -> FailureAnalysisResult:
    rows = (
        db.query(FailureCaseORM)
        .filter(FailureCaseORM.project_id == project_id)
        .order_by(FailureCaseORM.id.desc())
        .all()
    )
    cases = [
        FailureCase(
            domain_or_slice=r.domain_or_slice,
            shift_score=r.shift_score,
            performance_drop=r.performance_drop,
            failure_type=FailureType(r.failure_type),
            note=r.note,
        )
        for r in rows
    ]
    return FailureAnalysisResult(project_id=project_id, cases=cases)
