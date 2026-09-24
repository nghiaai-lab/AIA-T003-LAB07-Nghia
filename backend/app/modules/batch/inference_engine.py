"""YOLO Inference & Embedding Engine.
Executes batch inference and task-aware feature embedding extraction,
computing detailed prediction statistics, confidence distributions, and bounding box metrics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from backend.app.core.schemas import EmbeddingRecord, PredictionRecord
from backend.app.modules.source.model_manager import ModelManager

logger = logging.getLogger(__name__)


def classify_bbox_size(w: float, h: float) -> str:
    """Classify bounding box into small, medium, or large based on COCO pixel area boundaries."""
    area = w * h
    if area < 32 * 32:
        return "small"
    elif area <= 96 * 96:
        return "medium"
    else:
        return "large"


class InferenceEngine:
    """Handles batched inference, prediction record parsing, and embedding generation."""

    def __init__(self, model: YOLO, conf_threshold: float = 0.25) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.class_names: dict[int, str] = getattr(model, "names", {})

    def process_batch(
        self,
        images: list[tuple[str, Image.Image | np.ndarray | Path | str]],
        batch_size: int = 16,
    ) -> tuple[list[PredictionRecord], list[EmbeddingRecord], np.ndarray, dict[str, Any]]:
        """Run batched inference and feature embedding extraction.

        Args:
            images: List of (image_id, image_content_or_path).
            batch_size: Mini-batch size for GPU/CPU memory efficiency.

        Returns:
            tuple:
                - List of PredictionRecord
                - List of EmbeddingRecord
                - np.ndarray of all embeddings (N, D)
                - Dictionary of aggregate prediction statistics
        """
        all_pred_records: list[PredictionRecord] = []
        all_emb_records: list[EmbeddingRecord] = []
        all_embeddings_list: list[np.ndarray] = []

        total_images = len(images)
        if total_images == 0:
            empty_stats = {
                "confidence_mean": 0.0,
                "confidence_std": 0.0,
                "confidence_max": 0.0,
                "detections_per_image": 0.0,
                "empty_detection_rate": 0.0,
                "class_distribution": {},
                "bbox_size_distribution": {"small": 0, "medium": 0, "large": 0},
            }
            return [], [], np.empty((0, 256), dtype=np.float32), empty_stats

        for start_idx in range(0, total_images, batch_size):
            chunk = images[start_idx : start_idx + batch_size]
            chunk_ids = [item[0] for item in chunk]
            chunk_imgs = [item[1] for item in chunk]

            # 1. Extract task-aware embeddings for this chunk
            chunk_embs = ModelManager.extract_embeddings(self.model, chunk_imgs)
            all_embeddings_list.append(chunk_embs)

            # 2. Run object detection inference
            preds = self.model.predict(
                chunk_imgs,
                conf=self.conf_threshold,
                verbose=False,
            )

            # 3. Parse prediction results
            for idx, (img_id, result) in enumerate(zip(chunk_ids, preds)):
                boxes_data = result.boxes
                emb_vector = chunk_embs[idx].tolist()

                if boxes_data is None or len(boxes_data) == 0:
                    pred_rec = PredictionRecord(
                        image_id=img_id,
                        confidence_mean=0.0,
                        confidence_max=0.0,
                        num_detections=0,
                        classes=[],
                        class_names=[],
                        boxes=[],
                        empty_detection=True,
                        bbox_sizes={"small": 0, "medium": 0, "large": 0},
                    )
                    emb_rec = EmbeddingRecord(
                        image_id=img_id,
                        embedding=emb_vector,
                        confidence=0.0,
                        num_detections=0,
                    )
                else:
                    confs = boxes_data.conf.cpu().numpy().tolist()
                    clss = boxes_data.cls.cpu().numpy().astype(int).tolist()
                    xyxy = boxes_data.xyxy.cpu().numpy().tolist()

                    c_mean = float(np.mean(confs))
                    c_max = float(np.max(confs))
                    num_det = len(confs)

                    box_sizes = {"small": 0, "medium": 0, "large": 0}
                    formatted_boxes = []

                    for box, c, cl in zip(xyxy, confs, clss):
                        x1, y1, x2, y2 = box
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        sz = classify_bbox_size(w, h)
                        box_sizes[sz] += 1
                        formatted_boxes.append([float(x1), float(y1), float(x2), float(y2), float(c), int(cl)])

                    names = [self.class_names.get(c_id, f"class_{c_id}") for c_id in clss]

                    pred_rec = PredictionRecord(
                        image_id=img_id,
                        confidence_mean=c_mean,
                        confidence_max=c_max,
                        num_detections=num_det,
                        classes=clss,
                        class_names=names,
                        boxes=formatted_boxes,
                        empty_detection=False,
                        bbox_sizes=box_sizes,
                    )
                    emb_rec = EmbeddingRecord(
                        image_id=img_id,
                        embedding=emb_vector,
                        confidence=c_mean,
                        num_detections=num_det,
                    )

                all_pred_records.append(pred_rec)
                all_emb_records.append(emb_rec)

        full_embeddings = np.vstack(all_embeddings_list)

        # 4. Compute aggregate statistics across the entire batch
        stats = self._compute_aggregate_statistics(all_pred_records)

        return all_pred_records, all_emb_records, full_embeddings, stats

    def _compute_aggregate_statistics(self, records: list[PredictionRecord]) -> dict[str, Any]:
        """Compute aggregate summary metrics across prediction records."""
        total = len(records)
        if total == 0:
            return {
                "confidence_mean": 0.0,
                "confidence_std": 0.0,
                "confidence_max": 0.0,
                "detections_per_image": 0.0,
                "empty_detection_rate": 0.0,
                "class_distribution": {},
                "bbox_size_distribution": {"small": 0, "medium": 0, "large": 0},
            }

        confs = [r.confidence_mean for r in records if not r.empty_detection]
        total_detections = sum(r.num_detections for r in records)
        empty_count = sum(1 for r in records if r.empty_detection)

        class_dist: dict[str, int] = {}
        bbox_dist = {"small": 0, "medium": 0, "large": 0}

        for r in records:
            for c_name in r.class_names:
                class_dist[c_name] = class_dist.get(c_name, 0) + 1
            for k in bbox_dist:
                bbox_dist[k] += r.bbox_sizes.get(k, 0)

        conf_mean = float(np.mean(confs)) if confs else 0.0
        conf_std = float(np.std(confs)) if confs else 0.0
        conf_max = float(np.max(confs)) if confs else 0.0

        return {
            "confidence_mean": round(conf_mean, 4),
            "confidence_std": round(conf_std, 4),
            "confidence_max": round(conf_max, 4),
            "detections_per_image": round(total_detections / total, 2),
            "empty_detection_rate": round(empty_count / total, 4),
            "total_detections": total_detections,
            "class_distribution": dict(sorted(class_dist.items(), key=lambda x: x[1], reverse=True)),
            "bbox_size_distribution": bbox_dist,
        }
