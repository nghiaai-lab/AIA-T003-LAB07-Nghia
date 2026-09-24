"""Module B (Chien27803) — 2.5 Shift Score Engine. See docs/spec.md."""
from fastapi import APIRouter, HTTPException

from backend.app.modules.shift.schemas import (
    ShiftCalculationRequest,
    ShiftCalculationResponse,
)
from backend.app.modules.shift import service

router = APIRouter(prefix="/shift", tags=["shift"])


@router.get("/health")
def health() -> dict:
    return {
        "module": "shift",
        "responsible": "Chien27803",
        "features": ["2.5 Shift Score Engine"],
        "status": "ready",
    }


@router.post("/calculate", response_model=ShiftCalculationResponse)
def calculate_shift_score(payload: ShiftCalculationRequest) -> ShiftCalculationResponse:
    """Calculate domain shift between Source embeddings and Target batch embeddings.
    
    Includes Signal A (Embedding shift via MMD/Fréchet) and Signal B (Confidence shift).
    Outputs normalized 0-100 scores and risk level.
    """
    if not payload.source_embeddings or not payload.target_embeddings:
        raise HTTPException(
            status_code=400,
            detail="Both source_embeddings and target_embeddings must contain at least 1 vector.",
        )

    try:
        return service.calculate_shift(
            source_embeddings=payload.source_embeddings,
            target_embeddings=payload.target_embeddings,
            source_confidences=payload.source_confidences,
            target_confidences=payload.target_confidences,
            baseline_distances=payload.baseline_distances,
            method=payload.method,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shift calculation error: {str(e)}")
