"""Artifact Store for Domain Shift Radar.
Manages disk persistence for project configurations, source baselines, and target batch artifacts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Manages reading and writing structured artifacts for projects, sources, and batches."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or settings.artifacts_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_project_dir(self, project_id: str) -> Path:
        p = self.base_dir / project_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_source_dir(self, project_id: str) -> Path:
        p = self.get_project_dir(project_id) / "source"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_batch_dir(self, project_id: str, batch_id: str) -> Path:
        p = self.get_project_dir(project_id) / "batches" / batch_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    # --- Project Config Persistence ---

    def save_project_config(self, project_id: str, data: dict[str, Any]) -> Path:
        out_path = self.get_project_dir(project_id) / "project_config.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path

    def load_project_config(self, project_id: str) -> dict[str, Any] | None:
        path = self.get_project_dir(project_id) / "project_config.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_projects(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return [
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "project_config.json").exists()
        ]

    # --- Source Artifact Persistence ---

    def save_source_artifacts(
        self,
        project_id: str,
        metadata: dict[str, Any],
        baseline: dict[str, Any],
        embeddings: np.ndarray,
        embedding_records: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        source_batches: list[dict[str, Any]],
    ) -> dict[str, str]:
        sdir = self.get_source_dir(project_id)

        meta_path = sdir / "metadata.json"
        base_path = sdir / "baseline.json"
        emb_npy_path = sdir / "embeddings.npy"
        emb_rec_path = sdir / "embedding_records.json"
        pred_path = sdir / "predictions.json"
        batches_path = sdir / "source_batches.json"

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        with open(base_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        np.save(emb_npy_path, embeddings.astype(np.float32))
        with open(emb_rec_path, "w", encoding="utf-8") as f:
            json.dump(embedding_records, f, indent=2, ensure_ascii=False)
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        with open(batches_path, "w", encoding="utf-8") as f:
            json.dump(source_batches, f, indent=2, ensure_ascii=False)

        return {
            "metadata": str(meta_path),
            "baseline": str(base_path),
            "embeddings": str(emb_npy_path),
            "embedding_records": str(emb_rec_path),
            "predictions": str(pred_path),
            "source_batches": str(batches_path),
        }

    def load_source_artifacts(self, project_id: str) -> dict[str, Any] | None:
        sdir = self.get_source_dir(project_id)
        base_path = sdir / "baseline.json"
        if not base_path.exists():
            return None

        with open(base_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        meta_path = sdir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        emb_npy_path = sdir / "embeddings.npy"
        embeddings = np.load(emb_npy_path) if emb_npy_path.exists() else None

        batches_path = sdir / "source_batches.json"
        source_batches = []
        if batches_path.exists():
            with open(batches_path, "r", encoding="utf-8") as f:
                source_batches = json.load(f)

        return {
            "metadata": metadata,
            "baseline": baseline,
            "embeddings": embeddings,
            "source_batches": source_batches,
        }

    # --- Batch Artifact Persistence ---

    def save_batch_artifacts(
        self,
        project_id: str,
        batch_id: str,
        metadata: dict[str, Any],
        embeddings: np.ndarray,
        embedding_records: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> dict[str, str]:
        bdir = self.get_batch_dir(project_id, batch_id)

        meta_path = bdir / "metadata.json"
        emb_npy_path = bdir / "embeddings.npy"
        emb_rec_path = bdir / "embedding_records.json"
        pred_path = bdir / "predictions.json"

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        np.save(emb_npy_path, embeddings.astype(np.float32))
        with open(emb_rec_path, "w", encoding="utf-8") as f:
            json.dump(embedding_records, f, indent=2, ensure_ascii=False)
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        return {
            "metadata": str(meta_path),
            "embeddings": str(emb_npy_path),
            "embedding_records": str(emb_rec_path),
            "predictions": str(pred_path),
        }

    def load_batch_artifacts(self, project_id: str, batch_id: str) -> dict[str, Any] | None:
        bdir = self.get_batch_dir(project_id, batch_id)
        meta_path = bdir / "metadata.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        emb_npy_path = bdir / "embeddings.npy"
        embeddings = np.load(emb_npy_path) if emb_npy_path.exists() else None

        emb_rec_path = bdir / "embedding_records.json"
        embedding_records = []
        if emb_rec_path.exists():
            with open(emb_rec_path, "r", encoding="utf-8") as f:
                embedding_records = json.load(f)

        pred_path = bdir / "predictions.json"
        predictions = []
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)

        return {
            "metadata": metadata,
            "embeddings": embeddings,
            "embedding_records": embedding_records,
            "predictions": predictions,
        }

    def list_batches(self, project_id: str) -> list[dict[str, Any]]:
        batches_dir = self.get_project_dir(project_id) / "batches"
        if not batches_dir.exists():
            return []
        results = []
        for d in batches_dir.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                with open(d / "metadata.json", "r", encoding="utf-8") as f:
                    try:
                        meta = json.load(f)
                        results.append(meta)
                    except Exception:
                        pass
        return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)


artifact_store = ArtifactStore()
