"""Batch Ingestion & Analysis Service.
Handles Target Batch Ingestion (FR 2.3), Synthetic Domain Shift Generation,
Batch Validation, Task-Aware Embedding Extraction (FR 2.4), and Artifact Handoff for Module B.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any
import uuid

import cv2
import numpy as np
from PIL import Image

from backend.app.core.artifacts import artifact_store
from backend.app.core.config import settings
from backend.app.core.schemas import (
    BatchAnalysisResponse,
    BatchValidationResult,
    TargetBatchIngestRequest,
)
from backend.app.modules.batch.corruptions import apply_corruption
from backend.app.modules.batch.inference_engine import InferenceEngine
from backend.app.modules.batch.validation import BatchValidator
from backend.app.modules.source.model_manager import ModelManager
from backend.app.modules.source.service import source_service

logger = logging.getLogger(__name__)


def annotate_image_boxes(image: Image.Image, boxes: list[list[float]], class_names: list[str]) -> Image.Image:
    """Draw bounding boxes and class labels onto a PIL image for preview display."""
    img_np = np.array(image.convert("RGB"))
    # Convert RGB to BGR for OpenCV drawing
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for box, cls_name in zip(boxes, class_names):
        x1, y1, x2, y2, conf, _ = box
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(bgr, p1, p2, (0, 255, 0), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(
            bgr,
            label,
            (int(x1), max(15, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class BatchService:
    """Orchestrates ingestion, corruption, validation, inference, and persistence of target batches."""

    @classmethod
    def ingest_synthetic_batch(
        cls,
        req: TargetBatchIngestRequest,
    ) -> BatchAnalysisResponse:
        """Generate a synthetic target batch from the source domain using deterministic corruptions,
        then run validation, inference, and embedding extraction."""
        # 1. Resolve source images to draw from
        source_paths, _, _ = source_service.resolve_source_images("coco128")
        if not source_paths:
            raise ValueError("Source reference dataset must be available to generate synthetic batch.")

        # Limit sample count to requested number
        count = min(req.sample_count, len(source_paths))
        selected_paths = source_paths[:count]

        # 2. Apply deterministic corruption to each selected image
        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        target_dir = settings.target_data_dir / req.project_id / batch_id
        target_dir.mkdir(parents=True, exist_ok=True)

        items_to_validate: list[tuple[str, bytes]] = []

        for idx, src_p in enumerate(selected_paths):
            pil_img = Image.open(src_p).convert("RGB")
            # Apply deterministic corruption with fixed index seed
            corrupted_np = apply_corruption(pil_img, corruption_type=req.corruption_type, seed=42 + idx)
            corrupted_pil = Image.fromarray(corrupted_np)

            # Save corrupted image into target data dir
            out_filename = f"{req.corruption_type}_{src_p.name}"
            out_path = target_dir / out_filename
            corrupted_pil.save(out_path, format="JPEG", quality=95)

            items_to_validate.append((out_filename, out_path.read_bytes()))

        return cls._process_ingested_items(
            project_id=req.project_id,
            batch_id=batch_id,
            batch_name=req.batch_name,
            items=items_to_validate,
            metadata={
                "project_id": req.project_id,
                "batch_id": batch_id,
                "batch_name": req.batch_name,
                "camera": req.camera,
                "city": req.city,
                "weather": req.weather,
                "time_of_day": req.time_of_day,
                "sensor": req.sensor,
                "notes": req.notes,
                "corruption_type": req.corruption_type,
                "synthetic": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    @classmethod
    def ingest_uploaded_batch(
        cls,
        project_id: str,
        batch_name: str,
        uploaded_files: list[tuple[str, bytes]],
        metadata_dict: dict[str, Any] | None = None,
    ) -> BatchAnalysisResponse:
        """Ingest user-uploaded image files, run validation, inference, and embedding extraction."""
        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        target_dir = settings.target_data_dir / project_id / batch_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save files to target directory
        for filename, data in uploaded_files:
            file_path = target_dir / Path(filename).name
            file_path.write_bytes(data)

        meta = {
            "project_id": project_id,
            "batch_id": batch_id,
            "batch_name": batch_name,
            "synthetic": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata_dict:
            meta.update(metadata_dict)

        return cls._process_ingested_items(
            project_id=project_id,
            batch_id=batch_id,
            batch_name=batch_name,
            items=uploaded_files,
            metadata=meta,
        )

    @classmethod
    def _process_ingested_items(
        cls,
        project_id: str,
        batch_id: str,
        batch_name: str,
        items: list[tuple[str, bytes]],
        metadata: dict[str, Any],
    ) -> BatchAnalysisResponse:
        """Common pipeline: Validate -> Inference + Embeddings -> Persist Artifacts."""
        # 1. Validation
        validator = BatchValidator(min_batch_size=5)
        val_result, valid_items = validator.validate_file_items(items)

        if not val_result.is_valid:
            logger.warning("Target batch %s validation failed: %s", batch_id, val_result.errors)

        # 2. Load model
        proj = source_service.get_project(project_id)
        checkpoint = proj.yolo_checkpoint if proj else settings.default_model
        req_device = proj.device if proj else settings.device

        model, resolved_ckpt = ModelManager.load_model(checkpoint, device=req_device)
        engine = InferenceEngine(model=model, conf_threshold=0.25)

        # 3. Batch Inference and Feature Embedding Extraction
        img_tuples = [(item[0], item[1]) for item in valid_items]
        pred_records, emb_records, embeddings_matrix, stats = engine.process_batch(
            img_tuples,
            batch_size=16,
        )

        num_images = len(pred_records)
        embedding_dim = int(embeddings_matrix.shape[1]) if embeddings_matrix.shape[0] > 0 else 256

        # 4. Generate annotated preview thumbnails
        thumb_dir = artifact_store.get_batch_dir(project_id, batch_id) / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        for p_rec, (_, pil_img, _) in zip(pred_records[:10], valid_items[:10]):
            try:
                annotated = annotate_image_boxes(pil_img, p_rec.boxes, p_rec.class_names)
                annotated.thumbnail((480, 480))
                annotated.save(thumb_dir / f"annotated_{p_rec.image_id}.jpg", "JPEG", quality=85)
            except Exception as e:
                logger.debug("Thumbnail generation error for %s: %s", p_rec.image_id, e)

        # 5. Persist Batch Artifacts
        metadata.update({
            "model_name": resolved_ckpt,
            "device": ModelManager.resolve_device(req_device),
            "num_images": num_images,
            "embedding_dim": embedding_dim,
            "validation_summary": val_result.model_dump(),
            "prediction_statistics": stats,
        })

        saved_paths = artifact_store.save_batch_artifacts(
            project_id=project_id,
            batch_id=batch_id,
            metadata=metadata,
            embeddings=embeddings_matrix,
            embedding_records=[r.model_dump() for r in emb_records],
            predictions=[r.model_dump() for r in pred_records],
        )

        return BatchAnalysisResponse(
            project_id=project_id,
            batch_id=batch_id,
            batch_name=batch_name,
            num_images=num_images,
            embedding_dim=embedding_dim,
            confidence_mean=stats["confidence_mean"],
            confidence_std=stats["confidence_std"],
            detections_per_image=stats["detections_per_image"],
            empty_detection_rate=stats["empty_detection_rate"],
            class_distribution=stats["class_distribution"],
            bbox_size_distribution=stats["bbox_size_distribution"],
            validation_result=val_result,
            artifact_paths=saved_paths,
            ready_for_module_b=True,
        )

    @classmethod
    def get_batch(cls, project_id: str, batch_id: str) -> dict[str, Any] | None:
        """Get stored batch artifacts and metadata."""
        return artifact_store.load_batch_artifacts(project_id, batch_id)

    @classmethod
    def list_batches(cls, project_id: str) -> list[dict[str, Any]]:
        """List all target batches analyzed for a project."""
        return artifact_store.list_batches(project_id)

    @classmethod
    def get_batch_embeddings(cls, project_id: str, batch_id: str) -> dict[str, Any] | None:
        """Get embeddings and embedding records for downstream Module B."""
        artifacts = artifact_store.load_batch_artifacts(project_id, batch_id)
        if not artifacts:
            return None
        return {
            "project_id": project_id,
            "batch_id": batch_id,
            "num_embeddings": len(artifacts["embedding_records"]),
            "embedding_records": artifacts["embedding_records"],
            "embeddings_npy_path": str(artifact_store.get_batch_dir(project_id, batch_id) / "embeddings.npy"),
        }


batch_service = BatchService()
