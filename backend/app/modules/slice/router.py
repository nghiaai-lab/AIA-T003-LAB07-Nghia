"""Module B (Chien27803) — 2.6 Slice Analyzer / 2.7 Risk Dashboard. See docs/spec.md."""
from fastapi import APIRouter, HTTPException

from backend.app.core.schemas import ShiftScoreRecord
from backend.app.modules.slice.schemas import (
    SliceAnalysisRequest,
    SliceAnalysisResponse,
)
from backend.app.modules.slice import service

router = APIRouter(prefix="/slice", tags=["slice"])


@router.get("/health")
def health() -> dict:
    return {
        "module": "slice",
        "responsible": "Chien27803",
        "features": ["2.6 Slice Analyzer", "2.7 Risk Dashboard"],
        "status": "ready",
    }


@router.post("/analyze", response_model=SliceAnalysisResponse)
def analyze_slices(request: SliceAnalysisRequest) -> SliceAnalysisResponse:
    """Analyze target batch metadata slices, compute shift score & risk level for each slice,
    and rank them descending by risk.
    """
    if not request.target_items:
        raise HTTPException(status_code=400, detail="Target items list cannot be empty.")
    try:
        return service.analyze_target_slices(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Slice analysis failed: {str(e)}")


@router.post("/export-records", response_model=list[ShiftScoreRecord])
def export_records_for_module_c(response: SliceAnalysisResponse) -> list[ShiftScoreRecord]:
    """Export slice shift score records directly formatted for Module C (Correlation Validator).
    
    Returns list of ShiftScoreRecord(domain_or_slice, shift_score, num_images).
    """
    try:
        return service.export_to_shift_score_records(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export records: {str(e)}")


@router.get("/mock-demo", response_model=SliceAnalysisResponse)
def get_mock_demo_analysis() -> SliceAnalysisResponse:
    """Run an end-to-end slice analysis on simulated benchmark batch data.
    
    Includes simulated Day/Sunny, Night/Clear, Day/Rain, Night/Rain, Fog slices with calibrated shifts.
    """
    source_embs, source_confs, target_items = service.generate_mock_batch_data()
    req = SliceAnalysisRequest(
        source_embeddings=source_embs,
        source_confidences=source_confs,
        target_items=target_items,
        min_slice_size=2,
    )
    return service.analyze_target_slices(req)
