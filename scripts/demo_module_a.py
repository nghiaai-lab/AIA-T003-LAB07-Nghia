"""End-to-End Demo Script for Module A (BanhKhuc04) — Domain Shift Radar.
Demonstrates complete source baseline extraction and target batch ingestion pipeline
without requiring a running frontend or manual user intervention.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.core.artifacts import artifact_store
from backend.app.core.schemas import ProjectConfig, SourceRegisterRequest, TargetBatchIngestRequest
from backend.app.modules.batch.service import batch_service
from backend.app.modules.source.model_manager import ModelManager
from backend.app.modules.source.service import source_service


def main() -> None:
    print("=" * 60)
    print("DOMAIN SHIFT RADAR — MODULE A END-TO-END DEMO")
    print("=" * 60)

    project_id = "demo_highway_radar"
    checkpoint = "yolo26n.pt"

    # Step 1: Initialize Project Configuration
    print("\n[1/5] Initializing Project Configuration...")
    proj_cfg = ProjectConfig(
        project_id=project_id,
        project_name="Highway Vehicle Shift Radar",
        task_type="object_detection",
        yolo_checkpoint=checkpoint,
        performance_metric="mAP50-95",
        risk_threshold=0.10,
        embedding_layer="second-to-last",
        device="auto",
    )
    source_service.save_project(proj_cfg)
    device_info = ModelManager.resolve_device(proj_cfg.device)
    print(f"  Project ID: {project_id}")
    print(f"  Model Checkpoint: {checkpoint}")
    print(f"  Resolved Device: {device_info}")

    # Step 2: Ensure Dataset & Register Source Domain
    print("\n[2/5] Registering Source Domain & Computing Baseline...")
    source_req = SourceRegisterRequest(
        project_id=project_id,
        source_name="COCO128_Highway_Baseline",
        dataset_path="coco128",
        camera="cam_tollway_north",
        city="Hanoi",
        weather="clear",
        time_of_day="day",
        sensor="Sony_RGB_1080p",
    )

    baseline_resp = source_service.compute_source_baseline(
        source_req,
        progress_callback=lambda p, msg: print(f"  [{int(p*100):02d}%] {msg}"),
    )

    print("\n[3/5] Source Baseline Computed Successfully:")
    print(f"  Source Images: {baseline_resp.num_images}")
    map_str = f"{baseline_resp.mAP50_95:.4f}" if baseline_resp.mAP50_95 is not None else "N/A"
    print(f"  Source mAP50-95: {map_str}")
    print(f"  Source Embeddings: ({baseline_resp.num_images}, {baseline_resp.embedding_dim})")
    print(f"  Source Mean Confidence: {baseline_resp.confidence_mean:.4f}")
    print(f"  Source Detections/Image: {baseline_resp.detections_per_image:.2f}")
    print(f"  Source Reference Bootstrap Batches: {baseline_resp.num_reference_batches}")

    # Step 3: Ingest Target Batch (Synthetic Dark Severe Domain Shift)
    print("\n[4/5] Ingesting & Analyzing Target Batch (Domain Shift: dark_severe)...")
    target_req = TargetBatchIngestRequest(
        project_id=project_id,
        batch_name="batch_night_rain_severe",
        camera="cam_tollway_north",
        city="Hanoi",
        weather="rain",
        time_of_day="night",
        sensor="Sony_RGB_1080p",
        notes="Severe nighttime low-light condition with rain",
        corruption_type="dark_severe",
        sample_count=30,
    )

    batch_resp = batch_service.ingest_synthetic_batch(target_req)

    print("\n[5/5] Target Batch Analysis Completed:")
    print(f"  Target Batch ID: {batch_resp.batch_id}")
    print(f"  Target Domain: dark_severe")
    print(f"  Target Images: {batch_resp.num_images}")
    print(f"  Target Embeddings: ({batch_resp.num_images}, {batch_resp.embedding_dim})")
    print(f"  Target Mean Confidence: {batch_resp.confidence_mean:.4f} (Source: {baseline_resp.confidence_mean:.4f})")
    print(f"  Target Detections/Image: {batch_resp.detections_per_image:.2f} (Source: {baseline_resp.detections_per_image:.2f})")
    print(f"  Target Empty Rate: {batch_resp.empty_detection_rate*100:.1f}% (Source: {baseline_resp.empty_detection_rate*100:.1f}%)")

    # Step 4: Summary of Saved Artifacts for Module B
    s_dir = artifact_store.get_source_dir(project_id)
    b_dir = artifact_store.get_batch_dir(project_id, batch_resp.batch_id)

    print("\n" + "=" * 60)
    print("ARTIFACTS SAVED TO DISK:")
    print(f"  Source Baseline: {s_dir}")
    print(f"  Target Batch:    {b_dir}")
    print("=" * 60)
    print("\nREADY FOR MODULE B (Shift Score Engine / Slice Analyzer)")
    print("Module B endpoints:")
    print(f"  GET /source/reference-data/{project_id}")
    print(f"  GET /batch/{project_id}/{batch_resp.batch_id}/embeddings")
    print("=" * 60)


if __name__ == "__main__":
    main()
