"""Source Baseline Service.
Handles Project Setup (FR 2.1), Model Management, Source Domain Registration (FR 2.2),
Task-Aware Embedding Extraction, Performance Evaluation, and Bootstrap Reference Subsets generation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ultralytics.data.utils import check_det_dataset

from backend.app.core.artifacts import artifact_store
from backend.app.core.config import settings
from backend.app.core.schemas import (
    ProjectConfig,
    SourceBaselineResponse,
    SourceReferenceDataResponse,
    SourceRegisterRequest,
)
from backend.app.modules.batch.inference_engine import InferenceEngine
from backend.app.modules.source.model_manager import ModelManager

logger = logging.getLogger(__name__)


class SourceService:
    """Orchestrates Project Setup, Model Verification, and Source Baseline calculation."""

    @classmethod
    def save_project(cls, config: ProjectConfig) -> ProjectConfig:
        """Create or update a project configuration."""
        now = datetime.now(timezone.utc).isoformat()
        if not config.created_at:
            config.created_at = now
        config.updated_at = now

        # Ensure model is checked to populate classes if empty
        if not config.classes:
            try:
                model, _ = ModelManager.load_model(config.yolo_checkpoint, device=config.device)
                config.classes = list(getattr(model, "names", {}).values())
            except Exception as e:
                logger.warning("Could not auto-populate classes for project %s: %s", config.project_id, e)

        artifact_store.save_project_config(config.project_id, config.model_dump())
        return config

    @classmethod
    def get_project(cls, project_id: str) -> ProjectConfig | None:
        """Retrieve an existing project configuration."""
        data = artifact_store.load_project_config(project_id)
        if not data:
            return None
        return ProjectConfig(**data)

    @classmethod
    def list_projects(cls) -> list[ProjectConfig]:
        """List all registered projects."""
        p_ids = artifact_store.list_projects()
        results: list[ProjectConfig] = []
        for pid in p_ids:
            p = cls.get_project(pid)
            if p:
                results.append(p)
        return results

    @classmethod
    def test_model(cls, checkpoint: str, device: str = "auto") -> dict[str, Any]:
        """Test model availability and return diagnostic info."""
        model, resolved_ckpt = ModelManager.load_model(checkpoint=checkpoint, device=device)
        return ModelManager.get_model_info(model, checkpoint=resolved_ckpt, device=device)

    @classmethod
    def resolve_source_images(cls, dataset_spec: str) -> tuple[list[Path], str, Path | None]:
        """Resolve image file paths for source dataset.
        Supports 'coco128' (auto-downloads via Ultralytics) or a local filesystem folder.

        Returns:
            tuple[list[Path], str, Path | None]: (image_paths, resolved_dataset_name, dataset_yaml_path)
        """
        spec = dataset_spec.strip().lower()

        if spec in ("coco128", "coco128.yaml", "default"):
            data_dict = check_det_dataset("coco128.yaml")
            val_path = Path(data_dict["val"])
            yaml_path = Path(data_dict.get("yaml_file", "coco128.yaml"))

            if val_path.is_dir():
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                img_paths = sorted([p for p in val_path.iterdir() if p.suffix.lower() in valid_exts])
                return img_paths, "COCO128", yaml_path
            elif val_path.is_file():
                # Text file containing list of images
                lines = val_path.read_text(encoding="utf-8").splitlines()
                img_paths = [Path(l.strip()) for l in lines if l.strip() and Path(l.strip()).exists()]
                return img_paths, "COCO128", yaml_path

        # Custom local folder path
        local_dir = Path(dataset_spec)
        if local_dir.exists() and local_dir.is_dir():
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            img_paths = sorted([p for p in local_dir.iterdir() if p.suffix.lower() in valid_exts])
            return img_paths, local_dir.name, None

        raise FileNotFoundError(f"Could not locate dataset or image folder at: '{dataset_spec}'")

    @classmethod
    def compute_source_baseline(
        cls,
        req: SourceRegisterRequest,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> SourceBaselineResponse:
        """Run complete source baseline workflow:
        1. Resolve source dataset (COCO128 or custom).
        2. Load YOLO model.
        3. Run validation if annotations are available.
        4. Run batched inference & extract embeddings.
        5. Partition source into bootstrap mini-batches for Module B normal-variation calibration.
        6. Persist all artifacts.
        """
        if progress_callback:
            progress_callback(0.05, "Resolving source dataset...")

        img_paths, dataset_label, yaml_path = cls.resolve_source_images(req.dataset_path)
        total_images = len(img_paths)
        if total_images == 0:
            raise ValueError(f"No valid image files found in dataset '{req.dataset_path}'.")

        # Get project config to determine model & device
        proj = cls.get_project(req.project_id)
        checkpoint = proj.yolo_checkpoint if proj else settings.default_model
        req_device = proj.device if proj else settings.device

        if progress_callback:
            progress_callback(0.15, f"Loading model {checkpoint}...")

        model, resolved_ckpt = ModelManager.load_model(checkpoint, device=req_device)
        resolved_device = ModelManager.resolve_device(req_device)

        # 3. Optional Ground Truth validation (e.g. COCO128)
        map50_95: float | None = None
        map50: float | None = None
        precision: float | None = None
        recall: float | None = None

        if yaml_path is not None:
            if progress_callback:
                progress_callback(0.25, "Running YOLO validation on source ground truth...")
            try:
                val_res = model.val(data=str(yaml_path), split="val", imgsz=640, batch=16, verbose=False)
                if hasattr(val_res, "box"):
                    map50_95 = round(float(val_res.box.map), 4)
                    map50 = round(float(val_res.box.map50), 4)
                    precision = round(float(val_res.box.mp), 4)
                    recall = round(float(val_res.box.mr), 4)
            except Exception as e:
                logger.warning("Could not run YOLO validation on %s: %s", yaml_path, e)

        # 4. Inference & Embedding Extraction
        if progress_callback:
            progress_callback(0.40, f"Extracting embeddings and predictions for {total_images} source images...")

        engine = InferenceEngine(model=model, conf_threshold=0.25)
        image_items = [(p.name, p) for p in img_paths]

        pred_records, emb_records, embeddings_matrix, stats = engine.process_batch(
            image_items,
            batch_size=16,
        )

        embedding_dim = int(embeddings_matrix.shape[1]) if embeddings_matrix.shape[0] > 0 else 256

        # 5. Partition source into bootstrap reference subsets (for Module B normal-variation calibration)
        if progress_callback:
            progress_callback(0.80, "Constructing source-vs-source reference subsets...")

        num_subsets = max(4, min(12, total_images // 10))
        rng = np.random.default_rng(42)
        shuffled_indices = rng.permutation(total_images)
        split_indices = np.array_split(shuffled_indices, num_subsets)

        source_batches_info: list[dict[str, Any]] = []
        for s_idx, idx_group in enumerate(split_indices):
            sub_embs = embeddings_matrix[idx_group]
            sub_records = [pred_records[i] for i in idx_group]
            sub_stats = engine._compute_aggregate_statistics(sub_records)

            source_batches_info.append({
                "subset_id": f"source_subset_{s_idx:02d}",
                "num_images": len(idx_group),
                "indices": idx_group.tolist(),
                "embedding_mean": sub_embs.mean(axis=0).tolist(),
                "prediction_statistics": sub_stats,
            })

        # 6. Save Artifacts
        if progress_callback:
            progress_callback(0.90, "Saving baseline artifacts to disk...")

        metadata = {
            "project_id": req.project_id,
            "source_name": req.source_name,
            "dataset_label": dataset_label,
            "dataset_path": str(req.dataset_path),
            "camera": req.camera,
            "city": req.city,
            "weather": req.weather,
            "time_of_day": req.time_of_day,
            "sensor": req.sensor,
            "model_name": resolved_ckpt,
            "device": resolved_device,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        baseline_summary = {
            "project_id": req.project_id,
            "source_name": req.source_name,
            "num_images": total_images,
            "model_name": resolved_ckpt,
            "device": resolved_device,
            "embedding_dim": embedding_dim,
            "confidence_mean": stats["confidence_mean"],
            "confidence_std": stats["confidence_std"],
            "detections_per_image": stats["detections_per_image"],
            "empty_detection_rate": stats["empty_detection_rate"],
            "class_distribution": stats["class_distribution"],
            "bbox_size_distribution": stats["bbox_size_distribution"],
            "mAP50_95": map50_95,
            "mAP50": map50,
            "precision": precision,
            "recall": recall,
            "num_reference_batches": len(source_batches_info),
        }

        saved_paths = artifact_store.save_source_artifacts(
            project_id=req.project_id,
            metadata=metadata,
            baseline=baseline_summary,
            embeddings=embeddings_matrix,
            embedding_records=[r.model_dump() for r in emb_records],
            predictions=[r.model_dump() for r in pred_records],
            source_batches=source_batches_info,
        )

        if progress_callback:
            progress_callback(1.0, "Source baseline completed successfully!")

        return SourceBaselineResponse(
            **baseline_summary,
            baseline_artifact_path=saved_paths["baseline"],
        )

    @classmethod
    def get_source_baseline(cls, project_id: str) -> SourceBaselineResponse | None:
        """Get the cached source baseline result for a project."""
        artifacts = artifact_store.load_source_artifacts(project_id)
        if not artifacts or not artifacts.get("baseline"):
            return None
        base = artifacts["baseline"]
        base_path = str(artifact_store.get_source_dir(project_id) / "baseline.json")
        return SourceBaselineResponse(**base, baseline_artifact_path=base_path)

    @classmethod
    def get_source_reference_data(cls, project_id: str) -> SourceReferenceDataResponse | None:
        """Get the reference embedding data and bootstrap subsets for Module B."""
        artifacts = artifact_store.load_source_artifacts(project_id)
        if not artifacts:
            return None

        baseline = artifacts.get("baseline", {})
        embeddings = artifacts.get("embeddings")
        source_batches = artifacts.get("source_batches", [])

        if embeddings is None:
            return None

        total_embs, emb_dim = embeddings.shape
        batch_sizes = [b["num_images"] for b in source_batches]

        return SourceReferenceDataResponse(
            project_id=project_id,
            total_embeddings=total_embs,
            embedding_dim=emb_dim,
            num_bootstrap_batches=len(source_batches),
            bootstrap_batch_sizes=batch_sizes,
            prediction_statistics={
                "confidence_mean": baseline.get("confidence_mean", 0.0),
                "confidence_std": baseline.get("confidence_std", 0.0),
                "detections_per_image": baseline.get("detections_per_image", 0.0),
                "empty_detection_rate": baseline.get("empty_detection_rate", 0.0),
                "class_distribution": baseline.get("class_distribution", {}),
                "bbox_size_distribution": baseline.get("bbox_size_distribution", {}),
            },
            artifacts_dir=str(artifact_store.get_source_dir(project_id)),
        )


source_service = SourceService()
