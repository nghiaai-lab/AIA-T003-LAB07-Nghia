"""Schemas cho False Alarm & Miss Explorer. Xem docs/spec.md 2.10."""
from enum import Enum

from pydantic import BaseModel

from backend.app.modules.correlation.schemas import CorrelationInputRecord


class FailureType(str, Enum):
    FALSE_ALARM = "false_alarm"
    MISS = "miss"
    OK = "ok"


class FailureThresholds(BaseModel):
    shift_high: float = 70.0
    shift_low: float = 40.0
    drop_high: float = 0.15
    drop_low: float = 0.05


class FailureAnalysisRequest(BaseModel):
    project_id: str
    records: list[CorrelationInputRecord]
    thresholds: FailureThresholds = FailureThresholds()


class FailureCase(BaseModel):
    domain_or_slice: str
    shift_score: float
    performance_drop: float
    failure_type: FailureType
    note: str


class FailureAnalysisResult(BaseModel):
    project_id: str
    cases: list[FailureCase]
