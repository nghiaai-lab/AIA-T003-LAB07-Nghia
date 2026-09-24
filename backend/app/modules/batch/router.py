"""Module A (BanhKhuc04) — 2.3 Target Batch Ingestion / 2.4 Embedding & Inference API Router.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.app.core.schemas import (
    BatchAnalysisResponse,
    TargetBatchIngestRequest,
)
from backend.app.modules.batch.service import batch_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"module": "batch", "status": "ok"}


@router.post("/demo", response_model=BatchAnalysisResponse)
def generate_synthetic_batch(req: TargetBatchIngestRequest) -> BatchAnalysisResponse:
    """Generate a target batch using deterministic photometric corruptions on source images (FR 2.3)."""
    try:
        return batch_service.ingest_synthetic_batch(req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to generate synthetic batch for project %s", req.project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate synthetic target batch: {e}",
        )


@router.post("/upload", response_model=BatchAnalysisResponse)
async def upload_batch(
    project_id: str = Form("default_project"),
    batch_name: str = Form(...),
    camera: str = Form("camera_target"),
    city: str = Form("unknown"),
    weather: str = Form("unknown"),
    time_of_day: str = Form("day"),
    sensor: str = Form("rgb_sensor"),
    notes: str = Form(""),
    files: list[UploadFile] = File(...),
) -> BatchAnalysisResponse:
    """Ingest uploaded target images, validate, run YOLO inference and extract task-aware embeddings (FR 2.3/2.4)."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded. At least one image file is required.",
        )

    file_items: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        file_items.append((f.filename or "unnamed.jpg", content))

    metadata = {
        "camera": camera,
        "city": city,
        "weather": weather,
        "time_of_day": time_of_day,
        "sensor": sensor,
        "notes": notes,
    }

    try:
        return batch_service.ingest_uploaded_batch(
            project_id=project_id,
            batch_name=batch_name,
            uploaded_files=file_items,
            metadata_dict=metadata,
        )
    except Exception as e:
        logger.exception("Batch upload failed for project %s", project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {e}",
        )


@router.get("/{project_id}/{batch_id}")
def get_batch_details(project_id: str, batch_id: str) -> dict[str, Any]:
    """Retrieve full batch analysis details, predictions, and metadata."""
    batch_data = batch_service.get_batch(project_id, batch_id)
    if not batch_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found in project '{project_id}'.",
        )
    # Exclude raw numpy array from json response
    meta = batch_data.get("metadata", {})
    preds = batch_data.get("predictions", [])
    records = batch_data.get("embedding_records", [])
    return {
        "metadata": meta,
        "total_images": len(preds),
        "predictions": preds,
        "embedding_records_count": len(records),
    }


@router.get("/{project_id}/{batch_id}/embeddings")
def get_batch_embeddings(project_id: str, batch_id: str) -> dict[str, Any]:
    """Handoff endpoint for Module B: provides target embeddings and per-image records."""
    embs_data = batch_service.get_batch_embeddings(project_id, batch_id)
    if not embs_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch embeddings not found for '{batch_id}' in project '{project_id}'.",
        )
    return embs_data


@router.get("/{project_id}")
def list_batches(project_id: str) -> list[dict[str, Any]]:
    """List all target batches analyzed for the given project."""
    return batch_service.list_batches(project_id)
