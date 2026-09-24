"""Business logic for Slice Analyzer & Risk Dashboard (Module B, Chien27803).
See docs/spec.md — 2.6 Slice Analyzer / 2.7 Risk Dashboard.
"""
from collections import defaultdict
from typing import Any
import numpy as np

from backend.app.core.schemas import ShiftScoreRecord
from backend.app.modules.shift import service as shift_service
from backend.app.modules.slice.schemas import (
    SliceAnalysisRequest,
    SliceAnalysisResponse,
    SliceMetric,
    TargetItem,
)


def extract_slices_from_items(
    items: list[TargetItem],
    min_slice_size: int = 1,
) -> dict[str, tuple[dict[str, str], list[TargetItem]]]:
    """Group target items into metadata-based slices (single and composite)."""
    slice_groups: dict[str, list[TargetItem]] = defaultdict(list)
    slice_filters: dict[str, dict[str, str]] = {}

    for item in items:
        meta = item.metadata or {}
        # 1. Single attribute slices (e.g. weather=rain, time=night)
        for k, v in meta.items():
            if not v:
                continue
            slice_name = f"{k}: {v}"
            slice_groups[slice_name].append(item)
            if slice_name not in slice_filters:
                slice_filters[slice_name] = {k: v}

        # 2. Key composite slices (e.g. time: night + weather: rain)
        if "time" in meta and "weather" in meta and meta["time"] and meta["weather"]:
            composite_name = f"time={meta['time']} + weather={meta['weather']}"
            slice_groups[composite_name].append(item)
            if composite_name not in slice_filters:
                slice_filters[composite_name] = {
                    "time": meta["time"],
                    "weather": meta["weather"],
                }

    # Filter out slices smaller than min_slice_size
    valid_slices = {}
    for name, group in slice_groups.items():
        if len(group) >= min_slice_size:
            valid_slices[name] = (slice_filters.get(name, {}), group)

    return valid_slices


def find_top_outliers(
    slice_items: list[TargetItem],
    source_mean: np.ndarray,
    top_k: int = 5,
) -> list[str]:
    """Find image_ids within a slice that have the largest Euclidean distance from the source mean."""
    if not slice_items:
        return []
    embeddings = np.array([item.embedding for item in slice_items], dtype=np.float64)
    distances = np.linalg.norm(embeddings - source_mean, axis=1)
    top_indices = np.argsort(distances)[::-1][:top_k]
    return [slice_items[i].image_id for i in top_indices]


def make_slice_recommendation(risk_level: str, slice_name: str, avg_conf: float) -> str:
    """Generate human-readable actionable recommendation."""
    if risk_level == "Critical":
        return (
            f"⚠️ Rủi ro nghiêm trọng: Slice '{slice_name}' có độ dịch chuyển rất lớn và độ tự tin sụt giảm (conf: {avg_conf:.2f}). "
            f"Ưu tiên cao nhất: gán nhãn đánh giá mAP và bổ sung vào tập retraining."
        )
    if risk_level == "High risk":
        return (
            f"🟠 Rủi ro cao: Slice '{slice_name}' dịch chuyển rõ rệt so với source baseline. "
            f"Khuyến nghị rà soát mẫu ảnh outlier và theo dõi tỷ lệ miss detection."
        )
    if risk_level == "Watch":
        return (
            f"🟡 Cần chú ý: Slice '{slice_name}' có dấu hiệu biến động vượt nhẹ mức thông thường. "
            f"Tiếp tục quan sát trong các batch kế tiếp."
        )
    return f"🟢 Bình thường: Slice '{slice_name}' nằm trong biên độ dao động thông thường của source domain."


def analyze_target_slices(request: SliceAnalysisRequest) -> SliceAnalysisResponse:
    """Run slice analysis and risk ranking across all discovered metadata slices."""
    target_items = request.target_items
    if not target_items:
        return SliceAnalysisResponse(
            total_target_images=0,
            total_slices=0,
            overall_shift_score=0.0,
            overall_risk_level="Normal",
            highest_risk_slice="None",
            slices=[],
        )

    dim = len(target_items[0].embedding)

    # 1. Source baseline vectors
    if request.source_embeddings and len(request.source_embeddings) > 0:
        source_embs = np.asarray(request.source_embeddings, dtype=np.float64)
    else:
        # Default standard baseline if not provided
        np.random.seed(42)
        source_embs = np.random.normal(loc=0.0, scale=1.0, size=(100, dim))

    source_mean = np.mean(source_embs, axis=0)
    source_confs = request.source_confidences or [0.85] * len(source_embs)

    # 2. Overall target shift
    all_target_embs = [item.embedding for item in target_items]
    all_target_confs = [item.confidence for item in target_items]
    overall_res = shift_service.calculate_shift(
        source_embeddings=source_embs.tolist(),
        target_embeddings=all_target_embs,
        source_confidences=source_confs,
        target_confidences=all_target_confs,
        baseline_distances=request.baseline_distances,
    )

    # 3. Discover slices
    slice_groups = extract_slices_from_items(target_items, min_slice_size=request.min_slice_size)

    slices_metrics: list[SliceMetric] = []

    for slice_name, (filters, group_items) in slice_groups.items():
        slice_embs = [item.embedding for item in group_items]
        slice_confs = [item.confidence for item in group_items]

        shift_res = shift_service.calculate_shift(
            source_embeddings=source_embs.tolist(),
            target_embeddings=slice_embs,
            source_confidences=source_confs,
            target_confidences=slice_confs,
            baseline_distances=request.baseline_distances,
        )

        avg_conf = float(np.mean(slice_confs)) if slice_confs else 0.0
        outliers = find_top_outliers(group_items, source_mean=source_mean, top_k=4)
        rec = make_slice_recommendation(shift_res.risk_level, slice_name, avg_conf)

        slices_metrics.append(
            SliceMetric(
                slice_name=slice_name,
                num_images=len(group_items),
                shift_score=shift_res.overall_shift_score,
                confidence_shift_score=shift_res.confidence_shift_score,
                avg_confidence=round(avg_conf, 3),
                risk_level=shift_res.risk_level,
                top_outlier_image_ids=outliers,
                recommendation=rec,
                metadata_filters=filters,
            )
        )

    # Sort slices descending by shift_score (highest risk first)
    slices_metrics.sort(key=lambda x: x.shift_score, reverse=True)

    highest_slice = slices_metrics[0].slice_name if slices_metrics else "None"

    return SliceAnalysisResponse(
        total_target_images=len(target_items),
        total_slices=len(slices_metrics),
        overall_shift_score=overall_res.overall_shift_score,
        overall_risk_level=overall_res.risk_level,
        highest_risk_slice=highest_slice,
        slices=slices_metrics,
    )


def export_to_shift_score_records(response: SliceAnalysisResponse) -> list[ShiftScoreRecord]:
    """Convert slice results into ShiftScoreRecord schema for Module C (toilatrung).
    
    See docs/spec.md (2.9 Correlation Validator).
    """
    records: list[ShiftScoreRecord] = []
    # Add overall batch as primary record
    records.append(
        ShiftScoreRecord(
            domain_or_slice="overall_batch",
            shift_score=response.overall_shift_score,
            num_images=response.total_target_images,
            risk_level=response.overall_risk_level,
        )
    )
    # Add each individual slice
    for s in response.slices:
        records.append(
            ShiftScoreRecord(
                domain_or_slice=s.slice_name,
                shift_score=s.shift_score,
                num_images=s.num_images,
                avg_confidence=s.avg_confidence,
                risk_level=s.risk_level,
            )
        )
    return records


def generate_mock_batch_data(seed: int = 42) -> tuple[list[list[float]], list[float], list[TargetItem]]:
    """Generate realistic synthetic source baseline and target batch with domain shifts for testing and demo."""
    rng = np.random.default_rng(seed)
    dim = 64  # Feature embedding dimension

    # 1. Source domain (normal day sunny conditions)
    n_source = 150
    source_embs = rng.normal(loc=0.0, scale=1.0, size=(n_source, dim))
    source_confs = rng.uniform(0.80, 0.95, size=n_source).tolist()

    # 2. Target domain with varying slices
    slices_specs = [
        # (name, time, weather, camera, city, count, shift_mean, conf_range)
        ("day_sunny", "day", "sunny", "Cam_A", "Hanoi", 45, 0.1, (0.80, 0.92)),
        ("day_cloudy", "day", "cloudy", "Cam_B", "Danang", 30, 0.4, (0.75, 0.88)),
        ("night_clear", "night", "clear", "Cam_A", "Hanoi", 35, 1.2, (0.65, 0.78)),
        ("rain_day", "day", "rain", "Cam_C", "Hanoi", 25, 1.5, (0.55, 0.72)),
        ("night_rain", "night", "rain", "Cam_C", "Saigon", 25, 2.6, (0.35, 0.55)),
        ("fog_morning", "day", "fog", "Cam_D", "Sapa", 15, 2.1, (0.40, 0.60)),
    ]

    target_items: list[TargetItem] = []
    item_counter = 1

    for spec_name, t, w, cam, city, count, shift_delta, (conf_min, conf_max) in slices_specs:
        shift_vector = rng.normal(loc=shift_delta, scale=1.0, size=(count, dim))
        confs = rng.uniform(conf_min, conf_max, size=count)
        for i in range(count):
            img_id = f"img_{item_counter:04d}_{spec_name}"
            target_items.append(
                TargetItem(
                    image_id=img_id,
                    embedding=shift_vector[i].tolist(),
                    confidence=float(round(confs[i], 3)),
                    num_detections=int(rng.integers(1, 8)),
                    metadata={
                        "time": t,
                        "weather": w,
                        "camera": cam,
                        "city": city,
                    },
                )
            )
            item_counter += 1

    return source_embs.tolist(), source_confs, target_items
