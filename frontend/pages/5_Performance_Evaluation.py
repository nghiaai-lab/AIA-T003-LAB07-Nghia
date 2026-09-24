"""Performance Evaluation — phụ trách: Trịnh Quang Trung. Xem docs/spec.md (2.8)."""
import json

import streamlit as st

from frontend.api_client import get, post

st.set_page_config(page_title="Performance Evaluation", layout="wide")
st.title("Performance Evaluation")

st.markdown(
    "Upload ground truth và prediction (JSON list of "
    "`{image_id, class_name, bbox: [x, y, w, h], score?}`) để tính mAP và performance "
    "drop so với source baseline."
)

with st.form("evaluation_form"):
    project_id = st.text_input("Project ID", value="demo-project")
    domain_or_slice = st.text_input("Domain / Slice", value="camera_C_night_rain")
    source_metric = st.number_input("Source metric (baseline)", value=0.85, min_value=0.0, max_value=1.0)
    iou_threshold = st.slider("IoU threshold", 0.1, 0.9, 0.5)
    gt_file = st.file_uploader("Ground truth JSON", type="json")
    pred_file = st.file_uploader("Predictions JSON", type="json")
    submitted = st.form_submit_button("Run evaluation")

if submitted:
    if not gt_file or not pred_file:
        st.error("Cần upload cả ground truth và predictions.")
    else:
        try:
            result = post(
                "/evaluation/run",
                {
                    "project_id": project_id,
                    "domain_or_slice": domain_or_slice,
                    "predictions": json.load(pred_file),
                    "ground_truth": json.load(gt_file),
                    "source_metric": source_metric,
                    "iou_threshold": iou_threshold,
                },
            )
            st.success(
                f"Target metric: {result['target_metric']:.3f} — "
                f"performance drop: {result['performance_drop']:.3f}"
            )
            st.json(result["ap_per_class"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Lỗi khi gọi backend: {exc}")

st.divider()
st.subheader("Lịch sử evaluation của project")
if st.button("Tải lịch sử"):
    try:
        st.table(get(f"/evaluation/{project_id}"))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Lỗi khi gọi backend: {exc}")
