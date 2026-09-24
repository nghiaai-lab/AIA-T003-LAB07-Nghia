"""Model Manager for YOLO detection models.
Provides cached loading, device resolution, embedding extraction, and fallback handling.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages YOLO model instances, cached loading, device placement, and task-aware embedding."""

    _models: dict[str, YOLO] = {}

    @classmethod
    def resolve_device(cls, requested: str = "auto") -> str:
        """Resolve requested device string ('auto', 'cpu', 'cuda', 'cuda:0') to an available PyTorch device."""
        if not requested or requested.lower() == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.lower().startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return requested

    @classmethod
    def load_model(
        cls,
        checkpoint: str = "yolo26n.pt",
        device: str = "auto",
        fallback_checkpoint: str = "yolo11n.pt",
    ) -> tuple[YOLO, str]:
        """Load and cache a YOLO model.
        Falls back to fallback_checkpoint or yolov8n.pt if primary checkpoint fails.

        Returns:
            tuple[YOLO, str]: (loaded_model, resolved_checkpoint_name)
        """
        resolved_dev = cls.resolve_device(device)
        cache_key = f"{checkpoint}:{resolved_dev}"

        if cache_key in cls._models:
            return cls._models[cache_key], checkpoint

        candidates = [checkpoint, fallback_checkpoint, "yolov8n.pt"]
        last_error: Exception | None = None

        for cand in candidates:
            if not cand:
                continue
            try:
                logger.info("Attempting to load YOLO model: %s on device: %s", cand, resolved_dev)
                model = YOLO(cand)
                # Move to device if supported
                if hasattr(model, "to"):
                    try:
                        model.to(resolved_dev)
                    except Exception as dev_err:
                        logger.warning("Could not explicitly move model to %s: %s", resolved_dev, dev_err)
                cls._models[cache_key] = model
                return model, cand
            except Exception as e:
                logger.warning("Failed to load checkpoint '%s': %s", cand, e)
                last_error = e

        raise RuntimeError(
            f"Failed to load any YOLO model from candidates {candidates}. Last error: {last_error}"
        )

    @classmethod
    def extract_embeddings(
        cls,
        model: YOLO,
        images: list[Any] | list[str] | list[Path] | np.ndarray,
        layer_index: int | None = None,
    ) -> np.ndarray:
        """Extract task-aware embeddings from YOLO model.

        Returns:
            np.ndarray: shape (N, D), float32, normalized, with no NaN or Inf.
        """
        if len(images) == 0:
            return np.empty((0, 256), dtype=np.float32)

        kwargs: dict[str, Any] = {}
        if layer_index is not None:
            kwargs["embed"] = [layer_index]

        raw_embeddings = model.embed(images, **kwargs)

        # Handle list of tensors or single tensor
        if isinstance(raw_embeddings, list):
            tensors = [
                e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else np.asarray(e)
                for e in raw_embeddings
            ]
            emb_matrix = np.stack(tensors, axis=0)
        elif isinstance(raw_embeddings, torch.Tensor):
            emb_matrix = raw_embeddings.detach().cpu().numpy()
        else:
            emb_matrix = np.asarray(raw_embeddings)

        # Flatten if 3D or 4D (e.g. from intermediate feature map pooling)
        if emb_matrix.ndim > 2:
            emb_matrix = emb_matrix.reshape(emb_matrix.shape[0], -1)

        emb_matrix = emb_matrix.astype(np.float32)

        # Ensure no NaN or Inf
        emb_matrix = np.nan_to_num(emb_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        # L2-normalize each embedding vector for stable downstream distance metrics (MMD/Fréchet)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = emb_matrix / norms

        return normalized

    @classmethod
    def get_model_info(cls, model: YOLO, checkpoint: str, device: str) -> dict[str, Any]:
        """Return diagnostic and structural info about the loaded model."""
        class_names = list(model.names.values()) if hasattr(model, "names") else []
        num_classes = len(class_names)
        num_params = 0
        if hasattr(model, "model") and model.model is not None:
            try:
                num_params = sum(p.numel() for p in model.model.parameters())
            except Exception:
                num_params = 0

        return {
            "checkpoint": checkpoint,
            "device": cls.resolve_device(device),
            "num_classes": num_classes,
            "class_names": class_names,
            "parameter_count": num_params,
            "status": "ready",
        }
