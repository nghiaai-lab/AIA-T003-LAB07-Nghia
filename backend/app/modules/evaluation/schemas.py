"""Schemas cho Ground Truth & Performance Evaluation. Xem docs/spec.md 2.8."""
from pydantic import BaseModel


class Detection(BaseModel):
    image_id: str
    class_name: str
    bbox: list[float]  # [x, y, w, h]
    score: float | None = None  # confidence — chỉ có ở prediction, không có ở ground truth


class EvaluationRequest(BaseModel):
    project_id: str
    domain_or_slice: str
    predictions: list[Detection]
    ground_truth: list[Detection]
    source_metric: float
    iou_threshold: float = 0.5


class EvaluationResult(BaseModel):
    project_id: str
    domain_or_slice: str
    target_metric: float
    source_metric: float
    performance_drop: float
    ap_per_class: dict[str, float]
    num_predictions: int
    num_ground_truth: int
