"""Module A (BanhKhuc04) — 2.1 Project Setup / 2.2 Source Domain Registry API Router.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app.core.schemas import (
    ProjectConfig,
    SourceBaselineResponse,
    SourceReferenceDataResponse,
    SourceRegisterRequest,
)
from backend.app.modules.source.service import source_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/source", tags=["source"])


class ModelTestRequest(BaseModel):
    checkpoint: str = "yolo26n.pt"
    device: str = "auto"


@router.get("/health")
def health() -> dict[str, str]:
    return {"module": "source", "status": "ok"}


@router.post("/project", response_model=ProjectConfig)
def save_project(config: ProjectConfig) -> ProjectConfig:
    """Save or update project configuration (FR 2.1 Project Setup)."""
    try:
        return source_service.save_project(config)
    except Exception as e:
        logger.exception("Error saving project %s", config.project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save project: {e}",
        )


@router.get("/project/{project_id}", response_model=ProjectConfig)
def get_project(project_id: str) -> ProjectConfig:
    """Retrieve an existing project configuration."""
    proj = source_service.get_project(project_id)
    if not proj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )
    return proj


@router.get("/projects", response_model=list[ProjectConfig])
def list_projects() -> list[ProjectConfig]:
    """List all registered projects."""
    return source_service.list_projects()


@router.post("/model/load")
def load_and_test_model(req: ModelTestRequest) -> dict[str, Any]:
    """Test loading a YOLO checkpoint and verify its classes and device placement."""
    try:
        return source_service.test_model(checkpoint=req.checkpoint, device=req.device)
    except Exception as e:
        logger.exception("Model loading test failed for checkpoint %s", req.checkpoint)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model loading failed: {e}",
        )


@router.post("/baseline", response_model=SourceBaselineResponse)
def compute_source_baseline(req: SourceRegisterRequest) -> SourceBaselineResponse:
    """Register source domain and compute baseline artifacts (FR 2.2)."""
    try:
        return source_service.compute_source_baseline(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to compute source baseline for project %s", req.project_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute source baseline: {e}",
        )


@router.get("/baseline/{project_id}", response_model=SourceBaselineResponse)
def get_source_baseline(project_id: str) -> SourceBaselineResponse:
    """Retrieve computed source baseline for a project."""
    res = source_service.get_source_baseline(project_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source baseline for project '{project_id}' not found. Please run baseline first.",
        )
    return res


@router.get("/reference-data/{project_id}", response_model=SourceReferenceDataResponse)
def get_source_reference_data(project_id: str) -> SourceReferenceDataResponse:
    """Endpoint for Module B to fetch reference embeddings and bootstrap subsets for normal-variation distribution."""
    data = source_service.get_source_reference_data(project_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference data for project '{project_id}' not found.",
        )
    return data
