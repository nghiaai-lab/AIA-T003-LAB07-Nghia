"""Business logic for Shift Score Engine (Module B, Chien27803).
See docs/spec.md — 2.5 Shift Score Engine.
"""
from typing import Any
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

from backend.app.modules.shift.schemas import ShiftCalculationResponse


def compute_median_gamma(X: np.ndarray, Y: np.ndarray | None = None) -> float:
    """Compute gamma for RBF kernel using the median heuristic."""
    data = X if Y is None else np.vstack([X, Y])
    # Subsample if large to keep computation fast and responsive
    if len(data) > 500:
        indices = np.random.choice(len(data), size=500, replace=False)
        data = data[indices]
    dists = cdist(data, data, metric="sqeuclidean")
    median_dist = np.median(dists[dists > 0])
    if median_dist <= 0:
        return 1.0 / X.shape[1]
    return float(1.0 / (2.0 * median_dist))


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """Compute Maximum Mean Discrepancy (MMD) with RBF kernel between X and Y.
    
    X: shape (n_samples, n_features) - Source embeddings
    Y: shape (m_samples, n_features) - Target embeddings
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n = len(X)
    m = len(Y)
    if n == 0 or m == 0:
        return 0.0

    if gamma is None:
        gamma = compute_median_gamma(X, Y)

    K_xx = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
    K_yy = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
    K_xy = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))

    # Unbiased estimate or standard U-statistic
    if n > 1 and m > 1:
        np.fill_diagonal(K_xx, 0.0)
        np.fill_diagonal(K_yy, 0.0)
        mmd_sq = (
            np.sum(K_xx) / (n * (n - 1))
            + np.sum(K_yy) / (m * (m - 1))
            - 2.0 * np.sum(K_xy) / (n * m)
        )
    else:
        mmd_sq = np.mean(K_xx) + np.mean(K_yy) - 2.0 * np.mean(K_xy)

    return float(np.sqrt(max(0.0, float(mmd_sq))))


def compute_frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Fréchet distance (Wasserstein-2 between Gaussians)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)

    diff = mu_x - mu_y
    mean_dist_sq = float(np.dot(diff, diff))

    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    if cov_x.ndim == 0:
        cov_x = np.array([[cov_x]])
        cov_y = np.array([[cov_y]])

    try:
        from scipy.linalg import sqrtm
        cov_mean = sqrtm(cov_x.dot(cov_y))
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        trace_term = np.trace(cov_x + cov_y - 2.0 * cov_mean)
        fd = mean_dist_sq + float(trace_term)
        return float(np.sqrt(max(0.0, fd)))
    except Exception:
        # Fallback to mean squared difference if covariance matrix is singular
        return float(np.sqrt(mean_dist_sq))


def normalize_to_score(
    raw_distance: float,
    baseline_distances: list[float] | np.ndarray | None = None,
) -> float:
    """Map raw distance to 0-100 Shift Score.
    
    Formula from docs/spec.md (2.5):
    ShiftScore = ECDF_{source-source}(D_{source, target}) * 100
    """
    if baseline_distances is not None and len(baseline_distances) > 0:
        base = np.asarray(baseline_distances, dtype=np.float64)
        score = float(stats.percentileofscore(base, raw_distance, kind="rank"))
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    # Fallback calibration curve when explicit baseline distances are omitted
    # Uses continuous sigmoid/exponential mapping calibrated to standard embedding MMD ranges
    k = 3.5
    calibrated = 100.0 * (1.0 - np.exp(-k * raw_distance))
    return round(float(np.clip(calibrated, 0.0, 100.0)), 2)


def compute_confidence_shift(
    source_conf: list[float] | np.ndarray | None,
    target_conf: list[float] | np.ndarray | None,
) -> tuple[float, dict[str, Any]]:
    """Compute confidence distribution shift score (0-100) using KS-test and Wasserstein."""
    if source_conf is None or target_conf is None or len(source_conf) == 0 or len(target_conf) == 0:
        return 0.0, {"status": "no_confidence_data"}

    s_conf = np.asarray(source_conf, dtype=np.float64)
    t_conf = np.asarray(target_conf, dtype=np.float64)

    ks_stat, p_val = stats.ks_2samp(s_conf, t_conf)
    wass_dist = stats.wasserstein_distance(s_conf, t_conf)
    mean_drop = float(np.mean(s_conf) - np.mean(t_conf))

    # KS statistic is bounded in [0, 1]
    conf_score = float(np.clip(ks_stat * 100.0, 0.0, 100.0))

    details = {
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_val), 6),
        "wasserstein_distance": round(float(wass_dist), 4),
        "mean_source_conf": round(float(np.mean(s_conf)), 3),
        "mean_target_conf": round(float(np.mean(t_conf)), 3),
        "confidence_drop": round(mean_drop, 3),
    }
    return round(conf_score, 2), details


def classify_risk_level(overall_score: float) -> str:
    """Classify risk level according to docs/spec.md (2.7 Risk Dashboard):
    - Normal: Nằm trong biến động thông thường của source (< 60)
    - Watch: Có shift nhưng chưa đủ bằng chứng về performance risk (60 - 79)
    - High risk: Shift lớn và signal từng liên hệ với performance drop (80 - 91)
    - Critical: Dự đoán performance drop vượt ngưỡng cho phép (>= 92)
    """
    if overall_score < 60.0:
        return "Normal"
    if overall_score < 80.0:
        return "Watch"
    if overall_score < 92.0:
        return "High risk"
    return "Critical"


def calculate_shift(
    source_embeddings: list[list[float]],
    target_embeddings: list[list[float]],
    source_confidences: list[float] | None = None,
    target_confidences: list[float] | None = None,
    baseline_distances: list[float] | None = None,
    method: str = "mmd",
) -> ShiftCalculationResponse:
    """Main calculation routine for Shift Score Engine (FR 2.5)."""
    X = np.asarray(source_embeddings, dtype=np.float64)
    Y = np.asarray(target_embeddings, dtype=np.float64)

    # 1. Signal A: Embedding Shift
    if method.lower() == "frechet":
        raw_emb_dist = compute_frechet_distance(X, Y)
    else:
        raw_emb_dist = compute_mmd(X, Y)

    emb_score = normalize_to_score(raw_emb_dist, baseline_distances)

    # 2. Signal B: Confidence / Prediction Shift
    conf_score, conf_details = compute_confidence_shift(source_confidences, target_confidences)

    pred_score = conf_score

    # 3. Overall Shift Score
    if source_confidences and target_confidences:
        overall_score = round(0.65 * emb_score + 0.35 * conf_score, 2)
    else:
        overall_score = emb_score

    risk = classify_risk_level(overall_score)

    details = {
        "method": method,
        "embedding_raw_distance": round(raw_emb_dist, 6),
        "confidence_shift_details": conf_details,
    }

    return ShiftCalculationResponse(
        embedding_shift_raw=round(raw_emb_dist, 6),
        embedding_shift_score=emb_score,
        confidence_shift_score=conf_score,
        prediction_shift_score=pred_score,
        overall_shift_score=overall_score,
        risk_level=risk,
        num_source_samples=len(X),
        num_target_samples=len(Y),
        details=details,
    )
