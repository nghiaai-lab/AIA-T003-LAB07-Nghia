"""Schemas cho Correlation Validator. Xem docs/spec.md 2.9."""
from pydantic import BaseModel

MIN_RECORDS = 3
RECOMMENDED_RECORDS = 8


class CorrelationInputRecord(BaseModel):
    domain_or_slice: str
    shift_score: float
    performance_drop: float


class CorrelationRequest(BaseModel):
    project_id: str
    records: list[CorrelationInputRecord]


class CorrelationResult(BaseModel):
    project_id: str
    spearman_rho: float
    p_value: float
    n: int
    records: list[CorrelationInputRecord]
