from backend.app.modules.evaluation.metrics import compute_map
from backend.app.modules.evaluation.schemas import Detection


def test_compute_map_perfect_match():
    gt = [Detection(image_id="1", class_name="car", bbox=[0, 0, 10, 10])]
    pred = [Detection(image_id="1", class_name="car", bbox=[0, 0, 10, 10], score=0.9)]

    mean_ap, ap_per_class = compute_map(pred, gt, iou_threshold=0.5)

    assert ap_per_class["car"] == 1.0
    assert mean_ap == 1.0


def test_compute_map_no_overlap_is_false_positive():
    gt = [Detection(image_id="1", class_name="car", bbox=[0, 0, 10, 10])]
    pred = [Detection(image_id="1", class_name="car", bbox=[100, 100, 10, 10], score=0.9)]

    mean_ap, ap_per_class = compute_map(pred, gt, iou_threshold=0.5)

    assert ap_per_class["car"] == 0.0
    assert mean_ap == 0.0


def test_compute_map_ignores_other_classes_in_ground_truth():
    gt = [Detection(image_id="1", class_name="car", bbox=[0, 0, 10, 10])]
    pred = [Detection(image_id="1", class_name="truck", bbox=[0, 0, 10, 10], score=0.9)]

    _, ap_per_class = compute_map(pred, gt, iou_threshold=0.5)

    assert ap_per_class["car"] == 0.0
    assert "truck" not in ap_per_class
