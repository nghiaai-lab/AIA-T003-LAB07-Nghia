"""Batch Analysis — FR 2.3/2.4 Target Batch Ingestion & Analysis (Module A: BanhKhuc04).
Allows users to upload target images or generate synthetic domain shift batches,
runs data validation, YOLO inference, and embedding extraction, and formats artifacts for Module B.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.app.core.artifacts import artifact_store
from backend.app.core.schemas import TargetBatchIngestRequest
from backend.app.modules.batch.corruptions import AVAILABLE_CORRUPTIONS
from backend.app.modules.batch.service import batch_service
from backend.app.modules.source.service import source_service

st.set_page_config(page_title="Batch Analysis — Domain Shift Radar", layout="wide", page_icon="🎯")

st.title("🎯 Target Batch Ingestion & Analysis")
st.caption("Module A (BanhKhuc04) — FR 2.3 & 2.4: Nạp batch production, kiểm tra chất lượng, trích xuất embedding và dự đoán.")

# 1. Project Selector
projects = source_service.list_projects()
if not projects:
    st.warning("Chưa có project nào được tạo. Vui lòng vào trang **1_Projects** để thiết lập project trước.")
    st.stop()

project_ids = [p.project_id for p in projects]
selected_proj_id = st.selectbox("Select Active Project", options=project_ids, index=0)
source_baseline = source_service.get_source_baseline(selected_proj_id)

if not source_baseline:
    st.warning(
        f"Project '{selected_proj_id}' chưa có Source Baseline. "
        "Vui lòng vào trang **2_Source_Baseline** để chạy baseline trước khi phân tích target batch."
    )

st.divider()

tab_synthetic, tab_upload, tab_history = st.tabs([
    "🧪 1. Simulated Domain Shift (Demo)",
    "📤 2. Upload Real Target Images",
    "📚 3. Batch Analysis History",
])

# ----------------------------------------------------
# TAB 1: Synthetic Domain Shift Generation
# ----------------------------------------------------
with tab_synthetic:
    st.subheader("Generate Synthetic Target Batch from Source")
    st.markdown(
        "Tạo batch target nhân tạo từ source domain bằng các phép biến đổi trắc quang (photometric corruption) "
        "để mô phỏng các hiện tượng domain shift trong thực tế (sương mù, bóng tối, rung lắc, nén dữ liệu)."
    )

    with st.form("synthetic_batch_form"):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            corruption_type = st.selectbox(
                "Domain Shift Corruption",
                options=AVAILABLE_CORRUPTIONS,
                index=AVAILABLE_CORRUPTIONS.index("dark_severe"),
                format_func=lambda x: {
                    "normal": "Normal (No Shift — Reference Copy)",
                    "dark_mild": "Dark Mild (Hoàng hôn / Thiếu sáng nhẹ)",
                    "dark_severe": "Dark Severe (Đêm tối / Low-light sensor)",
                    "gaussian_blur": "Gaussian Blur (Mờ nét / Thấu kính mờ)",
                    "motion_blur": "Motion Blur (Camera di chuyển / Xe chạy nhanh)",
                    "gaussian_noise": "Gaussian Noise (Nhiễu hạt cảm biến ISO cao)",
                    "jpeg_compression": "JPEG Compression (Nén băng thông thấp)",
                    "grayscale": "Grayscale (Cảm biến đơn sắc / Hồng ngoại)",
                    "contrast_low": "Low Contrast (Sương mù / Bụi bặm)",
                    "contrast_high": "High Contrast (Chói lóa mặt trời)",
                }.get(x, x),
            )
            sample_count = st.slider("Target Batch Size (Number of Images)", min_value=10, max_value=60, value=30, step=5)
            batch_name = st.text_input("Target Batch Name", value=f"batch_{corruption_type}_prod")

        with col_s2:
            camera = st.text_input("Camera ID", value="cam_target_tollway")
            city = st.text_input("City / Region", value="Highway_Zone_B")
            weather = st.selectbox("Weather", options=["rain", "clear", "fog", "cloudy", "haze"], index=0)
            time_of_day = st.selectbox("Time of Day", options=["night", "day", "twilight"], index=0)
            notes = st.text_area("Notes", value=f"Simulated {corruption_type} target domain shift.")

        btn_run_synthetic = st.form_submit_button("⚡ Generate & Run Target Batch Analysis", use_container_width=True)

    if btn_run_synthetic:
        req = TargetBatchIngestRequest(
            project_id=selected_proj_id,
            batch_name=batch_name,
            camera=camera,
            city=city,
            weather=weather,
            time_of_day=time_of_day,
            notes=notes,
            corruption_type=corruption_type,
            sample_count=sample_count,
        )
        with st.spinner("Generating corrupted target domain and extracting embeddings..."):
            try:
                res = batch_service.ingest_synthetic_batch(req)
                st.session_state["latest_batch_id"] = res.batch_id
                st.success(f"Batch '{res.batch_id}' successfully analyzed and stored!")
            except Exception as e:
                st.error(f"Failed to generate synthetic batch: {e}")

# ----------------------------------------------------
# TAB 2: Upload Target Images
# ----------------------------------------------------
with tab_upload:
    st.subheader("Upload Real Production Images")
    st.markdown("Tải lên danh sách ảnh chụp từ camera thực tế để kiểm tra và phân tích.")

    with st.form("upload_batch_form"):
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            up_batch_name = st.text_input("Batch Name", value="batch_camera_real_01")
            up_camera = st.text_input("Camera ID", value="cam_intersection_04")
            up_city = st.text_input("City", value="Danang")
        with u_col2:
            up_weather = st.selectbox("Weather", options=["sunny", "rain", "fog", "cloudy"], index=0)
            up_time_of_day = st.selectbox("Time of Day", options=["day", "night"], index=0)
            up_notes = st.text_input("Notes", value="Real production upload")

        uploaded_files = st.file_uploader(
            "Select Target Images (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        btn_run_upload = st.form_submit_button("⚡ Ingest & Analyze Uploaded Batch", use_container_width=True)

    if btn_run_upload:
        if not uploaded_files:
            st.error("Vui lòng chọn ít nhất một tệp ảnh để tải lên.")
        else:
            file_items = [(f.name, f.read()) for f in uploaded_files]
            with st.spinner(f"Ingesting and analyzing {len(file_items)} images..."):
                try:
                    res = batch_service.ingest_uploaded_batch(
                        project_id=selected_proj_id,
                        batch_name=up_batch_name,
                        uploaded_files=file_items,
                        metadata_dict={
                            "camera": up_camera,
                            "city": up_city,
                            "weather": up_weather,
                            "time_of_day": up_time_of_day,
                            "notes": up_notes,
                        },
                    )
                    st.session_state["latest_batch_id"] = res.batch_id
                    st.success(f"Batch '{res.batch_id}' successfully analyzed and stored!")
                except Exception as e:
                    st.error(f"Failed to ingest uploaded batch: {e}")

# ----------------------------------------------------
# TAB 3: Batch History
# ----------------------------------------------------
with tab_history:
    st.subheader("Analyzed Target Batches")
    all_batches = batch_service.list_batches(selected_proj_id)
    if not all_batches:
        st.info("Chưa có target batch nào được phân tích trong project này.")
    else:
        history_rows = []
        for b in all_batches:
            stats = b.get("prediction_statistics", {})
            history_rows.append({
                "Batch ID": b.get("batch_id"),
                "Batch Name": b.get("batch_name"),
                "Images": b.get("num_images"),
                "Corruption": b.get("corruption_type", "real"),
                "Mean Conf": stats.get("confidence_mean", 0.0),
                "Dets/Img": stats.get("detections_per_image", 0.0),
                "Created At": b.get("created_at", "")[:19],
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True)


# ----------------------------------------------------
# DISPLAY ACTIVE BATCH RESULTS
# ----------------------------------------------------
latest_id = st.session_state.get("latest_batch_id")
all_batches = batch_service.list_batches(selected_proj_id)

if all_batches:
    st.divider()
    batch_options = [b["batch_id"] for b in all_batches]
    default_idx = batch_options.index(latest_id) if latest_id in batch_options else 0
    active_batch_id = st.selectbox("Inspect Target Batch", options=batch_options, index=default_idx)

    batch_data = batch_service.get_batch(selected_proj_id, active_batch_id)
    if batch_data:
        meta = batch_data.get("metadata", {})
        preds = batch_data.get("predictions", [])
        stats = meta.get("prediction_statistics", {})
        val_summary = meta.get("validation_summary", {})

        st.header(f"📊 Results: {meta.get('batch_name', active_batch_id)}")

        # Validation Quality Banner
        if val_summary:
            v_col1, v_col2, v_col3, v_col4 = st.columns(4)
            v_col1.metric("Valid Images", val_summary.get("valid_images", len(preds)))
            v_col2.metric("Duplicates Filtered", val_summary.get("duplicates", 0))
            v_col3.metric("Corrupted Files", val_summary.get("corrupted_images", 0))
            v_col4.metric("Dimension Anomalies", val_summary.get("dimension_anomalies", 0))

            if val_summary.get("warnings"):
                with st.expander("⚠️ Validation Warnings"):
                    for w in val_summary["warnings"]:
                        st.write(f"- {w}")

        st.subheader("Target Batch vs Source Baseline Statistics")
        s_conf = source_baseline.confidence_mean if source_baseline else 0.0
        s_dets = source_baseline.detections_per_image if source_baseline else 0.0
        s_empty = source_baseline.empty_detection_rate if source_baseline else 0.0

        t_conf = stats.get("confidence_mean", 0.0)
        t_dets = stats.get("detections_per_image", 0.0)
        t_empty = stats.get("empty_detection_rate", 0.0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target Images", meta.get("num_images", len(preds)))
        c2.metric("Mean Confidence", f"{t_conf:.3f}", delta=f"{t_conf - s_conf:+.3f} vs Source")
        c3.metric("Detections/Image", f"{t_dets:.1f}", delta=f"{t_dets - s_dets:+.1f} vs Source")
        c4.metric("Empty Detection Rate", f"{t_empty * 100:.1f}%", delta=f"{(t_empty - s_empty)*100:+.1f}% vs Source", delta_color="inverse")

        # Visual Charts Comparison
        st.subheader("Prediction Distribution Comparison")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Class distribution comparison
            t_classes = stats.get("class_distribution", {})
            s_classes = source_baseline.class_distribution if source_baseline else {}
            all_c_keys = list(set(list(t_classes.keys())[:10] + list(s_classes.keys())[:10]))

            if all_c_keys:
                fig_comp = go.Figure(data=[
                    go.Bar(name="Source Baseline", x=all_c_keys, y=[s_classes.get(k, 0) for k in all_c_keys], marker_color="royalblue"),
                    go.Bar(name="Target Batch", x=all_c_keys, y=[t_classes.get(k, 0) for k in all_c_keys], marker_color="crimson"),
                ])
                fig_comp.update_layout(barmode="group", title="Class Frequency (Top Classes)", xaxis_tickangle=-45)
                st.plotly_chart(fig_comp, use_container_width=True)

        with chart_col2:
            # Bbox size distribution comparison
            t_bbox = stats.get("bbox_size_distribution", {})
            s_bbox = source_baseline.bbox_size_distribution if source_baseline else {}
            sz_keys = ["small", "medium", "large"]

            fig_sz = go.Figure(data=[
                go.Bar(name="Source Baseline", x=sz_keys, y=[s_bbox.get(k, 0) for k in sz_keys], marker_color="teal"),
                go.Bar(name="Target Batch", x=sz_keys, y=[t_bbox.get(k, 0) for k in sz_keys], marker_color="orange"),
            ])
            fig_sz.update_layout(barmode="group", title="Bounding Box Size Distribution")
            st.plotly_chart(fig_sz, use_container_width=True)

        # Annotated Thumbnail Gallery
        st.subheader("Detection Previews on Target Batch")
        thumb_dir = artifact_store.get_batch_dir(selected_proj_id, active_batch_id) / "thumbnails"
        if thumb_dir.exists():
            thumb_files = sorted(list(thumb_dir.glob("*.jpg")))[:8]
            if thumb_files:
                cols = st.columns(4)
                for idx, t_file in enumerate(thumb_files):
                    col = cols[idx % 4]
                    col.image(Image.open(t_file), caption=t_file.stem.replace("annotated_", ""), use_container_width=True)

        # Downstream Handoff Notice
        st.divider()
        st.info(
            "📌 **Shift Score will be produced by Module B (Chien27803)**\n\n"
            f"Module A đã tạo và lưu trữ đầy đủ artifacts:\n"
            f"- **Target Embeddings**: `{artifact_store.get_batch_dir(selected_proj_id, active_batch_id) / 'embeddings.npy'}`\n"
            f"- **Prediction Records**: `{artifact_store.get_batch_dir(selected_proj_id, active_batch_id) / 'predictions.json'}`\n"
            f"- **Batch API Endpoint**: `GET /batch/{selected_proj_id}/{active_batch_id}/embeddings`\n\n"
            "Dữ liệu sẵn sàng cho Module B chạy MMD / Fréchet distance và ECDF calibration."
        )
