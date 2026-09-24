"""Projects — FR 2.1 Project Setup (Module A: BanhKhuc04).
Allows users to configure radar evaluation sessions, select and test YOLO models,
configure metrics and thresholds, and manage project configurations.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from backend.app.core.schemas import ProjectConfig
from backend.app.modules.source.model_manager import ModelManager
from backend.app.modules.source.service import source_service

st.set_page_config(page_title="Projects — Domain Shift Radar", layout="wide", page_icon="🎯")

st.title("🎯 Project Setup & Configuration")
st.caption("Module A (BanhKhuc04) — FR 2.1: Quản lý phiên đánh giá, YOLO checkpoint và ngưỡng rủi ro.")

# 1. Project Selection or Creation
existing_projects = source_service.list_projects()
project_options = ["+ Create New Project"] + [p.project_id for p in existing_projects]

selected_option = st.selectbox(
    "Select Existing Project or Create New",
    options=project_options,
    index=0 if len(project_options) == 1 else 1,
)

current_cfg: ProjectConfig | None = None
if selected_option != "+ Create New Project":
    current_cfg = source_service.get_project(selected_option)

col1, col2 = st.columns([2, 1])

with col1:
    with st.form("project_form"):
        st.subheader("1. General Configuration")
        c1, c2 = st.columns(2)
        with c1:
            project_id = st.text_input(
                "Project ID",
                value=current_cfg.project_id if current_cfg else "vehicle_shift_radar",
                help="Unique identifier for the project artifacts directory.",
            )
            project_name = st.text_input(
                "Project Name",
                value=current_cfg.project_name if current_cfg else "Vehicle Detection Shift Radar",
            )
        with c2:
            task_type = st.selectbox(
                "Task Type",
                options=["object_detection"],
                index=0,
                help="Ultralytics object detection pipeline.",
            )
            device = st.selectbox(
                "Device",
                options=["auto", "cpu", "cuda:0"],
                index=["auto", "cpu", "cuda:0"].index(current_cfg.device) if current_cfg and current_cfg.device in ["auto", "cpu", "cuda:0"] else 0,
                help="Execution device for YOLO inference and embedding extraction.",
            )

        st.subheader("2. Model & Checkpoint")
        m1, m2 = st.columns(2)
        with m1:
            yolo_checkpoint = st.text_input(
                "YOLO Checkpoint",
                value=current_cfg.yolo_checkpoint if current_cfg else "yolo26n.pt",
                help="Ultralytics checkpoint name (e.g. yolo26n.pt, yolo11n.pt, yolov8n.pt).",
            )
        with m2:
            embedding_layer = st.selectbox(
                "Embedding Extraction Layer",
                options=["second-to-last (backbone/neck pooled)", "sppf"],
                index=0,
            )

        st.subheader("3. Monitoring Metrics & Risk Thresholds")
        r1, r2 = st.columns(2)
        with r1:
            performance_metric = st.selectbox(
                "Primary Evaluation Metric",
                options=["mAP50-95", "mAP50"],
                index=0 if not current_cfg or current_cfg.performance_metric == "mAP50-95" else 1,
            )
        with r2:
            risk_threshold = st.slider(
                "Risk / Performance-Drop Threshold",
                min_value=0.01,
                max_value=0.30,
                value=float(current_cfg.risk_threshold) if current_cfg else 0.10,
                step=0.01,
                format="%.2f",
                help="Ngưỡng sụt giảm performance tối đa chấp nhận được trước khi cảnh báo High/Critical Risk.",
            )

        submit_save = st.form_submit_button("💾 Save Project Configuration", use_container_width=True)

        if submit_save:
            if not project_id.strip():
                st.error("Project ID cannot be empty.")
            else:
                new_cfg = ProjectConfig(
                    project_id=project_id.strip(),
                    project_name=project_name.strip(),
                    task_type=task_type,
                    yolo_checkpoint=yolo_checkpoint.strip(),
                    performance_metric=performance_metric,
                    risk_threshold=risk_threshold,
                    embedding_layer="second-to-last",
                    device=device,
                )
                saved = source_service.save_project(new_cfg)
                st.success(f"Project '{saved.project_id}' successfully saved!")
                st.rerun()

with col2:
    st.subheader("Model Diagnostics")
    st.info(
        f"Active Device: **{ModelManager.resolve_device('auto')}**\n\n"
        f"Selected Checkpoint: `{yolo_checkpoint if 'yolo_checkpoint' in locals() else 'yolo26n.pt'}`"
    )

    if st.button("🔍 Test Model Loading & Classes", use_container_width=True):
        with st.spinner("Testing model loading..."):
            try:
                diag = source_service.test_model(
                    checkpoint=yolo_checkpoint,
                    device=device,
                )
                st.success(f"Model loaded successfully on {diag['device']}!")
                st.metric("Total Parameters", f"{diag['parameter_count']:,}")
                st.metric("Total Classes", diag["num_classes"])

                with st.expander("View Supported Classes"):
                    st.write(", ".join(diag["class_names"]))
            except Exception as e:
                st.error(f"Model test failed: {e}")

    if current_cfg:
        st.divider()
        st.subheader("Active Configuration")
        st.json(current_cfg.model_dump())
