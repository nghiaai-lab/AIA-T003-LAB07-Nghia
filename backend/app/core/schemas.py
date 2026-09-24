"""Schema dùng chung giữa các module. Đổi ở đây thì báo trong PR description
để cả 3 module (A/B/C) cùng review — xem CONTRIBUTING.md.
"""
from pydantic import BaseModel


class EmbeddingRecord(BaseModel):
    """Output của Module A (Embedding & Inference Engine) — input cho Module B."""

    image_id: str
    embedding: list[float]
    confidence: float
    num_detections: int


class ShiftScoreRecord(BaseModel):
    """Output của Module B (Shift Score Engine / Slice Analyzer) — input cho Module C."""

    domain_or_slice: str
    shift_score: float
    num_images: int


class PerformanceRecord(BaseModel):
    """Output của Module C (Performance Evaluation) dùng cho Correlation Validator."""

    domain_or_slice: str
    performance_drop: float
