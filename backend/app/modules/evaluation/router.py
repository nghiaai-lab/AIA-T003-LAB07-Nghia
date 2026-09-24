"""Module C (toilatrung) — 2.8 Ground Truth & Performance Evaluation. See docs/spec.md."""
from fastapi import APIRouter

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/health")
def health() -> dict:
    return {"module": "evaluation", "status": "stub"}

# TODO(toilatrung): implement endpoints for 2.8 Ground Truth & Performance Evaluation
