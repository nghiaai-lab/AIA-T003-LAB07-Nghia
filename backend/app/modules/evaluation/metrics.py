"""Tính mAP (mean Average Precision) cho object detection. Xem docs/spec.md 2.8.

Thuật toán: match prediction với ground truth theo IoU threshold trong cùng
image + cùng class (greedy theo confidence giảm dần), rồi tính AP theo cách
nội suy toàn điểm (all-point interpolation, giống COCO).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backend.app.modules.evaluation.schemas import Detection


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map(
    predictions: list[Detection],
    ground_truths: list[Detection],
    iou_threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Trả về (mAP, {class_name: AP}) trên các class có mặt trong ground_truths."""
    classes = sorted({gt.class_name for gt in ground_truths})
    ap_per_class: dict[str, float] = {}

    for cls in classes:
        gts_by_image: dict[str, list[Detection]] = defaultdict(list)
        for gt in ground_truths:
            if gt.class_name == cls:
                gts_by_image[gt.image_id].append(gt)
        matched: dict[str, list[bool]] = {img: [False] * len(dets) for img, dets in gts_by_image.items()}
        n_gt = sum(len(dets) for dets in gts_by_image.values())
        if n_gt == 0:
            continue

        preds = sorted(
            (p for p in predictions if p.class_name == cls),
            key=lambda p: p.score if p.score is not None else 0.0,
            reverse=True,
        )

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for i, pred in enumerate(preds):
            candidates = gts_by_image.get(pred.image_id, [])
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(candidates):
                if matched[pred.image_id][j]:
                    continue
                iou = _iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0:
                tp[i] = 1
                matched[pred.image_id][best_j] = True
            else:
                fp[i] = 1

        if len(preds) == 0:
            ap_per_class[cls] = 0.0
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)
        ap_per_class[cls] = _average_precision(recall, precision)

    mean_ap = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return mean_ap, ap_per_class
