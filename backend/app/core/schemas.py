"""Schema dùng chung giữa các module. Đổi ở đây thì báo trong PR description
để cả 3 module (A/B/C) cùng review — xem CONTRIBUTING.md.
"""
from pydantic import BaseModel


class EmbeddingRecord(BaseModel):
    """Output của Module A (Embedding & Inference Engine) — input cho Module B."""

    image_id: str
    embedding: list[float]
    confidence: float
    num_detections: int


class ShiftScoreRecord(BaseModel):
    """Output của Module B (Shift Score Engine / Slice Analyzer) — input cho Module C."""

    domain_or_slice: str
    shift_score: float
    num_images: int


class PerformanceRecord(BaseModel):
    """Output của Module C (Performance Evaluation) dùng cho Correlation Validator."""

    domain_or_slice: str
    performance_drop: float


# ==========================================
# Module A Schemas — Project, Source & Batch
# ==========================================


class ProjectConfig(BaseModel):
    """Cấu hình phiên làm việc (FR 2.1 Project Setup)."""

    project_id: str
    project_name: str
    task_type: str = "object_detection"
    yolo_checkpoint: str = "yolo26n.pt"
    performance_metric: str = "mAP50-95"
    risk_threshold: float = 0.10
    embedding_layer: str = "second-to-last"
    classes: list[str] = []
    device: str = "auto"
    created_at: str = ""
    updated_at: str = ""


class PredictionRecord(BaseModel):
    """Chi tiết kết quả inference cho một ảnh (FR 2.4)."""

    image_id: str
    confidence_mean: float = 0.0
    confidence_max: float = 0.0
    num_detections: int = 0
    classes: list[int] = []
    class_names: list[str] = []
    boxes: list[list[float]] = []  # [[x1, y1, x2, y2, conf, cls], ...]
    empty_detection: bool = False
    bbox_sizes: dict[str, int] = {"small": 0, "medium": 0, "large": 0}


class SourceRegisterRequest(BaseModel):
    """Đăng ký miền tham chiếu source (FR 2.2)."""

    project_id: str = "default_project"
    source_name: str = "COCO128_Reference"
    dataset_path: str = "coco128"
    camera: str = "camera_main"
    city: str = "reference_city"
    weather: str = "clear"
    time_of_day: str = "day"
    sensor: str = "rgb_sensor"


class SourceBaselineResponse(BaseModel):
    """Kết quả tính toán source baseline artifacts (FR 2.2 & 2.4)."""

    project_id: str
    source_name: str
    num_images: int
    model_name: str
    device: str
    embedding_dim: int
    confidence_mean: float
    confidence_std: float
    detections_per_image: float
    empty_detection_rate: float
    class_distribution: dict[str, int]
    bbox_size_distribution: dict[str, int]
    mAP50_95: float | None = None
    mAP50: float | None = None
    precision: float | None = None
    recall: float | None = None
    num_reference_batches: int = 0
    baseline_artifact_path: str = ""


class SourceReferenceDataResponse(BaseModel):
    """Dữ liệu phục vụ Module B xây dựng normal-variation distribution."""

    project_id: str
    total_embeddings: int
    embedding_dim: int
    num_bootstrap_batches: int
    bootstrap_batch_sizes: list[int]
    prediction_statistics: dict
    artifacts_dir: str


class BatchValidationResult(BaseModel):
    """Kết quả kiểm tra chất lượng target batch (FR 2.3)."""

    is_valid: bool = True
    total_files: int = 0
    valid_images: int = 0
    corrupted_images: int = 0
    duplicates: int = 0
    dimension_anomalies: int = 0
    source_overlap_count: int = 0
    warnings: list[str] = []
    errors: list[str] = []


class TargetBatchIngestRequest(BaseModel):
    """Yêu cầu nạp và phân tích batch target (FR 2.3)."""

    project_id: str = "default_project"
    batch_name: str
    camera: str = "camera_target"
    city: str = "production_city"
    weather: str = "unknown"
    time_of_day: str = "day"
    sensor: str = "rgb_sensor"
    notes: str = ""
    corruption_type: str = "none"
    sample_count: int = 30


class BatchAnalysisResponse(BaseModel):
    """Kết quả phân tích inference và trích xuất embedding cho batch target (FR 2.3/2.4)."""

    project_id: str
    batch_id: str
    batch_name: str
    num_images: int
    embedding_dim: int
    confidence_mean: float
    confidence_std: float
    detections_per_image: float
    empty_detection_rate: float
    class_distribution: dict[str, int]
    bbox_size_distribution: dict[str, int]
    validation_result: BatchValidationResult
    artifact_paths: dict[str, str]
    ready_for_module_b: bool = True
