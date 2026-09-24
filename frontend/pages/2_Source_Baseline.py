"""Source Baseline — FR 2.2 Source Domain Registry (Module A: BanhKhuc04).
Allows users to register a source reference dataset, run YOLO inference and validation,
extract task-aware embeddings, construct normal-variation reference subsets, and view baseline statistics.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st

from backend.app.core.artifacts import artifact_store
from backend.app.core.schemas import SourceRegisterRequest
from backend.app.modules.batch.service import annotate_image_boxes
from backend.app.modules.source.service import source_service

st.set_page_config(page_title="Source Baseline — Domain Shift Radar", layout="wide", page_icon="📊")

st.title("📊 Source Domain Registry & Baseline")
st.caption("Module A (BanhKhuc04) — FR 2.2: Đăng ký reference dataset, tính baseline metric và xuất reference subsets cho Module B.")

# 1. Project Selector
projects = source_service.list_projects()
if not projects:
    st.warning("Chưa có project nào được tạo. Vui lòng vào trang **1_Projects** để thiết lập project trước.")
    st.stop()

project_ids = [p.project_id for p in projects]
selected_proj_id = st.selectbox("Select Active Project", options=project_ids, index=0)
active_project = source_service.get_project(selected_proj_id)

st.divider()

col_form, col_status = st.columns([2, 1])

with col_form:
    st.subheader("1. Source Registration")
    with st.form("source_registry_form"):
        source_name = st.text_input("Source Domain Name", value="COCO128_Reference")
        dataset_choice = st.radio(
            "Dataset Source",
            options=["Demo: COCO128 (Tự động tải & cache)", "Custom Folder Path"],
            index=0,
        )

        custom_path = ""
        if dataset_choice == "Custom Folder Path":
            custom_path = st.text_input("Local Folder Path", value="data/source")

        c1, c2 = st.columns(2)
        with c1:
            camera = st.text_input("Camera ID", value="cam_highway_01")
            city = st.text_input("City / Location", value="Hanoi")
            weather = st.selectbox("Weather", options=["clear", "sunny", "rain", "fog", "cloudy"], index=0)
        with c2:
            time_of_day = st.selectbox("Time of Day", options=["day", "twilight", "night"], index=0)
            sensor = st.text_input("Sensor Type", value="Sony_IMX_RGB")

        submit_baseline = st.form_submit_button("🚀 Run Source Baseline Pipeline", use_container_width=True)

with col_status:
    st.subheader("Project Info")
    if active_project:
        st.write(f"**Model:** `{active_project.yolo_checkpoint}`")
        st.write(f"**Device:** `{active_project.device}`")
        st.write(f"**Metric:** `{active_project.performance_metric}`")
        st.write(f"**Risk Threshold:** `{active_project.risk_threshold:.2f}`")

    # Check if baseline already exists
    existing_baseline = source_service.get_source_baseline(selected_proj_id)
    if existing_baseline:
        st.success(f"Baseline Artifact Found! ({existing_baseline.num_images} images)")
        st.caption(f"Path: `{existing_baseline.baseline_artifact_path}`")
    else:
        st.info("Chưa có baseline nào cho project này. Nhấn 'Run Source Baseline' để khởi chạy.")

if submit_baseline:
    dataset_spec = "coco128" if dataset_choice.startswith("Demo") else custom_path
    req = SourceRegisterRequest(
        project_id=selected_proj_id,
        source_name=source_name,
        dataset_path=dataset_spec,
        camera=camera,
        city=city,
        weather=weather,
        time_of_day=time_of_day,
        sensor=sensor,
    )

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(pct: float, msg: str):
        progress_bar.progress(pct)
        status_text.text(msg)

    with st.spinner("Processing source baseline..."):
        try:
            baseline_res = source_service.compute_source_baseline(req, progress_callback=update_progress)
            st.success("Source baseline computation completed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Computation failed: {e}")

# Display Baseline Results if Available
current_baseline = source_service.get_source_baseline(selected_proj_id)
if current_baseline:
    st.divider()
    st.header(f"📈 Baseline Results — {current_baseline.source_name}")

    # Metrics Summary Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Images", current_baseline.num_images)
    m1_val = f"{current_baseline.mAP50_95:.4f}" if current_baseline.mAP50_95 is not None else "N/A"
    m2.metric("mAP50-95", m1_val)
    m2_val = f"{current_baseline.mAP50:.4f}" if current_baseline.mAP50 is not None else "N/A"
    m3.metric("mAP50", m2_val)
    m4.metric("Mean Conf", f"{current_baseline.confidence_mean:.3f}")
    m5.metric("Detections/Img", f"{current_baseline.detections_per_image:.1f}")
    m6.metric("Empty Rate", f"{current_baseline.empty_detection_rate * 100:.1f}%")

    tab_charts, tab_gallery, tab_subsets, tab_artifacts = st.tabs([
        "📊 Distribution Charts",
        "🖼️ Sample Predictions",
        "🔀 Normal-Variation Reference Subsets",
        "📁 Artifacts & Contracts",
    ])

    with tab_charts:
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.subheader("Top-10 Detected Classes")
            class_dist = current_baseline.class_distribution
            if class_dist:
                top_classes = dict(list(class_dist.items())[:10])
                df_cls = pd.DataFrame({"Class": list(top_classes.keys()), "Count": list(top_classes.values())})
                fig_cls = px.bar(df_cls, x="Class", y="Count", color="Count", color_continuous_scale="Blues")
                fig_cls.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_cls, use_container_width=True)
            else:
                st.write("No class distributions available.")

        with c_chart2:
            st.subheader("Bounding Box Size Breakdown")
            bbox_dist = current_baseline.bbox_size_distribution
            if bbox_dist:
                df_bbox = pd.DataFrame({"Size Category": list(bbox_dist.keys()), "Boxes": list(bbox_dist.values())})
                fig_bbox = px.pie(df_bbox, names="Size Category", values="Boxes", color_discrete_sequence=px.colors.sequential.Teal)
                st.plotly_chart(fig_bbox, use_container_width=True)

    with tab_gallery:
        st.subheader("Predictions with Overlaid Bounding Boxes (First 8 Samples)")
        artifacts = artifact_store.load_source_artifacts(selected_proj_id)
        if artifacts:
            preds_file = artifact_store.get_source_dir(selected_proj_id) / "predictions.json"
            if preds_file.exists():
                import json
                with open(preds_file, "r", encoding="utf-8") as f:
                    preds = json.load(f)

                # Get dataset images path
                img_paths, _, _ = source_service.resolve_source_images("coco128")
                img_map = {p.name: p for p in img_paths}

                cols = st.columns(4)
                for idx, p_rec in enumerate(preds[:8]):
                    col = cols[idx % 4]
                    img_id = p_rec["image_id"]
                    if img_id in img_map:
                        pil_img = Image.open(img_map[img_id]).convert("RGB")
                        annotated = annotate_image_boxes(pil_img, p_rec["boxes"], p_rec["class_names"])
                        col.image(annotated, caption=f"{img_id} (Dets: {p_rec['num_detections']})", use_container_width=True)

    with tab_subsets:
        st.subheader("Source-vs-Source Reference Subsets for Module B")
        st.markdown(
            "Để Module B tính toán **normal-variation distribution** (phân phối biến động bình thường), "
            "source dataset được phân hoạch thành các mini-batches tham chiếu độc lập:"
        )
        artifacts = artifact_store.load_source_artifacts(selected_proj_id)
        if artifacts and artifacts.get("source_batches"):
            sub_rows = []
            for b in artifacts["source_batches"]:
                stats = b["prediction_statistics"]
                sub_rows.append({
                    "Subset ID": b["subset_id"],
                    "Image Count": b["num_images"],
                    "Mean Conf": stats["confidence_mean"],
                    "Detections/Img": stats["detections_per_image"],
                    "Empty Rate": f"{stats['empty_detection_rate']*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(sub_rows), use_container_width=True)

    with tab_artifacts:
        st.subheader("Persisted Artifacts for Downstream Consumption")
        sdir = artifact_store.get_source_dir(selected_proj_id)
        st.code(
            f"{sdir}/\n"
            f"├── metadata.json\n"
            f"├── baseline.json\n"
            f"├── embeddings.npy          # Matrix shape: ({current_baseline.num_images}, {current_baseline.embedding_dim})\n"
            f"├── embedding_records.json   # List of EmbeddingRecord\n"
            f"├── predictions.json         # List of PredictionRecord with boxes & classes\n"
            f"└── source_batches.json      # {current_baseline.num_reference_batches} reference subsets\n",
            language="text",
        )
        st.info("Module B có thể truy xuất API `/source/reference-data/{project_id}` hoặc đọc trực tiếp các artifacts trên disk.")
