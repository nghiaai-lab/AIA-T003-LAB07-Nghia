"""Schemas for Slice Analyzer & Risk Dashboard (Module B, Chien27803).
See docs/spec.md — 2.6 Slice Analyzer / 2.7 Risk Dashboard.
"""
from typing import Any
from pydantic import BaseModel, Field


class TargetItem(BaseModel):
    """An individual image record from the target batch."""

    image_id: str
    embedding: list[float]
    confidence: float = 0.85
    num_detections: int = 1
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Attributes such as camera, time, weather, city, lighting, etc.",
    )


class SliceAnalysisRequest(BaseModel):
    """Payload to perform slice breakdown and risk ranking on a target batch."""

    source_embeddings: list[list[float]] | None = Field(
        None, description="Source baseline embeddings. If omitted, uses default synthetic source baseline."
    )
    source_confidences: list[float] | None = None
    target_items: list[TargetItem]
    baseline_distances: list[float] | None = Field(
        None, description="Source-vs-source internal distances for ECDF scaling."
    )
    min_slice_size: int = Field(
        1, description="Minimum number of samples required to consider a slice."
    )


class SliceMetric(BaseModel):
    """Analysis result for a specific slice."""

    slice_name: str
    num_images: int
    shift_score: float
    confidence_shift_score: float
    avg_confidence: float
    risk_level: str
    top_outlier_image_ids: list[str] = Field(default_factory=list)
    recommendation: str = ""
    metadata_filters: dict[str, str] = Field(default_factory=dict)


class SliceAnalysisResponse(BaseModel):
    """Overall response containing ranked slices and summary risk."""

    total_target_images: int
    total_slices: int
    overall_shift_score: float
    overall_risk_level: str
    highest_risk_slice: str
    slices: list[SliceMetric]
