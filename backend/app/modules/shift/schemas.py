"""Schemas for Shift Score Engine (Module B, Chien27803).
See docs/spec.md — 2.5 Shift Score Engine.
"""
from typing import Any
from pydantic import BaseModel, Field


class ShiftCalculationRequest(BaseModel):
    """Payload to calculate shift score between Source and Target datasets."""

    source_embeddings: list[list[float]] = Field(
        ..., description="List of embedding vectors for the source baseline."
    )
    target_embeddings: list[list[float]] = Field(
        ..., description="List of embedding vectors for the target batch."
    )
    source_confidences: list[float] | None = Field(
        None, description="Confidence scores from source model predictions."
    )
    target_confidences: list[float] | None = Field(
        None, description="Confidence scores from target model predictions."
    )
    baseline_distances: list[float] | None = Field(
        None, description="Source-vs-source internal distances used for ECDF calibration."
    )
    method: str = Field(
        "mmd", description="Distance metric: 'mmd' or 'frechet'."
    )


class ShiftCalculationResponse(BaseModel):
    """Result of shift score calculation."""

    embedding_shift_raw: float
    embedding_shift_score: float = Field(
        ..., description="Normalized score 0-100 based on source-source distribution."
    )
    confidence_shift_score: float = Field(
        ..., description="Confidence distribution shift score 0-100."
    )
    prediction_shift_score: float = Field(
        ..., description="Prediction output shift score 0-100."
    )
    overall_shift_score: float = Field(
        ..., description="Overall weighted shift score 0-100."
    )
    risk_level: str = Field(
        ..., description="Risk classification: Normal, Watch, High risk, Critical."
    )
    num_source_samples: int
    num_target_samples: int
    details: dict[str, Any] = Field(default_factory=dict)
