"""Slice Explorer & Risk Dashboard — Phụ trách: Chien27803 (Module B).
Xem docs/spec.md (2.6 Slice Analyzer / 2.7 Risk Dashboard).
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(
    page_title="Slice Explorer & Risk Dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-watch {
        background-color: #eab308;
        color: black;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-normal {
        background-color: #22c55e;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .recommendation-box {
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-top: 10px;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🧭 Radar Module B")
st.sidebar.caption("Chủ quản: **Chien27803** | Branch: `chien`")
st.sidebar.info(
    "**Phạm vi:**\n"
    "- 2.5: Shift Score Engine (MMD & KS-test)\n"
    "- 2.6: Slice Analyzer (Metadata Breakdown)\n"
    "- 2.7: Risk Dashboard (Alerts & Outliers)"
)

backend_api_url = st.sidebar.text_input("Backend API URL", value="http://127.0.0.1:8000")

# Header
st.title("🧭 Slice Explorer & Risk Dashboard")
st.markdown(
    "Đo lường mức độ **Domain Shift** không cần nhãn (unlabeled production batch), "
    "phát hiện các lát cắt dữ liệu rủi ro cao (**Slice Risk Ranking**) và khoanh vùng ảnh outlier."
)

# Mode Selector
col_mode1, col_mode2 = st.columns([3, 1])
with col_mode1:
    data_source_mode = st.radio(
        "Nguồn dữ liệu phân tích:",
        [
            "🧪 Chạy Benchmark Demo (Tự động giả lập batch có domain shift: Day/Night/Rain/Fog)",
            "🌐 Gọi API Backend trực tiếp (/slice/mock-demo hoặc /slice/analyze)",
        ],
        horizontal=True,
    )

with col_mode2:
    min_slice_count = st.number_input("Cỡ slice tối thiểu (samples)", min_value=1, max_value=50, value=2)

# Execution logic
analysis_data = None

if "Chạy Benchmark Demo" in data_source_mode:
    # Use internal service directly for ultra-fast, standalone execution
    from backend.app.modules.slice import service as slice_service
    from backend.app.modules.slice.schemas import SliceAnalysisRequest

    source_embs, source_confs, target_items = slice_service.generate_mock_batch_data()
    req = SliceAnalysisRequest(
        source_embeddings=source_embs,
        source_confidences=source_confs,
        target_items=target_items,
        min_slice_size=min_slice_count,
    )
    res = slice_service.analyze_target_slices(req)
    analysis_data = res.model_dump()
else:
    # Call live FastAPI backend
    try:
        resp = requests.get(f"{backend_api_url}/slice/mock-demo", timeout=5)
        if resp.status_code == 200:
            analysis_data = resp.json()
            st.success("✅ Đã kết nối thành công tới FastAPI backend!")
        else:
            st.error(f"❌ Backend trả về lỗi: {resp.status_code}")
    except Exception as ex:
        st.warning(f"⚠️ Chưa khởi động FastAPI backend tại {backend_api_url}. Đang chuyển sang chế độ tự động tính toán...")
        from backend.app.modules.slice import service as slice_service
        from backend.app.modules.slice.schemas import SliceAnalysisRequest

        source_embs, source_confs, target_items = slice_service.generate_mock_batch_data()
        req = SliceAnalysisRequest(
            source_embeddings=source_embs,
            source_confidences=source_confs,
            target_items=target_items,
            min_slice_size=min_slice_count,
        )
        res = slice_service.analyze_target_slices(req)
        analysis_data = res.model_dump()

if analysis_data:
    overall_score = analysis_data["overall_shift_score"]
    overall_risk = analysis_data["overall_risk_level"]
    total_imgs = analysis_data["total_target_images"]
    highest_slice = analysis_data["highest_risk_slice"]
    slices = analysis_data["slices"]

    # 1. Top KPI Cards
    st.markdown("### 📊 Tổng Quan Đợt Đánh Giá (Target Batch Overview)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    risk_colors = {
        "Critical": "🔴",
        "High risk": "🟠",
        "Watch": "🟡",
        "Normal": "🟢",
    }

    with kpi1:
        st.metric(
            label="Overall Shift Score",
            value=f"{overall_score:.1f} / 100",
            delta=f"{overall_risk}",
            delta_color="inverse" if overall_score > 60 else "normal",
        )
    with kpi2:
        st.metric(
            label="Trạng thái rủi ro",
            value=f"{risk_colors.get(overall_risk, '⚪')} {overall_risk}",
        )
    with kpi3:
        st.metric(
            label="Slice rủi ro cao nhất",
            value=highest_slice,
        )
    with kpi4:
        st.metric(
            label="Tổng ảnh / Slices",
            value=f"{total_imgs} ảnh / {len(slices)} slices",
        )

    st.markdown("---")

    # 2. Main Tabs
    tab_rank, tab_chart, tab_drill, tab_export = st.tabs([
        "🏆 Bảng Xếp Hạng Slice",
        "📈 Biểu Đồ & Tín Hiệu Shift",
        "🔍 Chi Tiết Slice & Ảnh Outlier",
        "📤 Xuất Cho Module C (toilatrung)",
    ])

    # Tab 1: Ranking
    with tab_rank:
        st.subheader("Bảng Xếp Hạng Độ Rủi Ro Theo Lát Cắt (Risk Ranking)")
        st.caption("Các slice được sắp xếp theo mức độ Domain Shift giảm dần. Tín hiệu shift cao cảnh báo nguy cơ sụt giảm performance của model.")

        table_rows = []
        for s in slices:
            table_rows.append({
                "Tên Slice": s["slice_name"],
                "Số Lượng Ảnh": s["num_images"],
                "Shift Score (0-100)": s["shift_score"],
                "Độ Tự Tin TB (Conf)": s["avg_confidence"],
                "Mức Độ Rủi Ro": s["risk_level"],
                "Khuyến Nghị Hành Động": s["recommendation"],
            })

        df_slices = pd.DataFrame(table_rows)

        # Style configuration
        def highlight_risk(val):
            if val == "Critical":
                return "background-color: rgba(239, 68, 68, 0.25); color: #ef4444; font-weight: bold;"
            elif val == "High risk":
                return "background-color: rgba(249, 115, 22, 0.25); color: #f97316; font-weight: bold;"
            elif val == "Watch":
                return "background-color: rgba(234, 179, 8, 0.25); color: #eab308; font-weight: bold;"
            elif val == "Normal":
                return "background-color: rgba(34, 197, 94, 0.25); color: #22c55e; font-weight: bold;"
            return ""

        styled_df = df_slices.style.map(highlight_risk, subset=["Mức Độ Rủi Ro"]).format({
            "Shift Score (0-100)": "{:.1f}",
            "Độ Tự Tin TB (Conf)": "{:.3f}",
        })

        st.dataframe(styled_df, use_container_width=True, height=350)

    # Tab 2: Visualizations
    with tab_chart:
        st.subheader("Trực Quan Hóa Tín Hiệu Shift")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Xếp hạng Shift Score theo Slice")
            chart_df = pd.DataFrame({
                "Slice": [s["slice_name"] for s in slices],
                "Shift Score": [s["shift_score"] for s in slices],
            }).set_index("Slice")
            st.bar_chart(chart_df, color="#3b82f6", horizontal=True)

        with c2:
            st.markdown("#### Độ tự tin trung bình của Model trên từng Slice")
            conf_df = pd.DataFrame({
                "Slice": [s["slice_name"] for s in slices],
                "Avg Confidence": [s["avg_confidence"] for s in slices],
            }).set_index("Slice")
            st.bar_chart(conf_df, color="#10b981", horizontal=True)

        st.markdown("#### Phân bố quan hệ giữa Shift Score và Độ sụt giảm Confidence")
        scatter_data = pd.DataFrame({
            "Slice": [s["slice_name"] for s in slices],
            "Shift Score": [s["shift_score"] for s in slices],
            "Avg Confidence": [s["avg_confidence"] for s in slices],
            "Số ảnh": [s["num_images"] for s in slices],
            "Rủi ro": [s["risk_level"] for s in slices],
        })
        st.scatter_chart(scatter_data, x="Shift Score", y="Avg Confidence", color="Rủi ro", size="Số ảnh")

    # Tab 3: Drill-down & Outliers
    with tab_drill:
        st.subheader("Khoanh Vùng & Phân Tích Chi Tiết Từng Lát Cắt")
        selected_slice_name = st.selectbox(
            "Chọn Slice cần đào sâu kiểm tra:",
            [s["slice_name"] for s in slices],
        )

        selected_slice = next((s for s in slices if s["slice_name"] == selected_slice_name), None)

        if selected_slice:
            d_col1, d_col2 = st.columns([2, 1])
            with d_col1:
                st.markdown(f"### Chi tiết: `{selected_slice['slice_name']}`")
                st.markdown(
                    f"- **Shift Score:** `{selected_slice['shift_score']:.1f} / 100`\n"
                    f"- **Mức độ rủi ro:** `{selected_slice['risk_level']}`\n"
                    f"- **Số lượng mẫu ảnh:** `{selected_slice['num_images']}`\n"
                    f"- **Độ tự tin trung bình:** `{selected_slice['avg_confidence']:.3f}`"
                )
                st.markdown(
                    f"<div class='recommendation-box'>{selected_slice['recommendation']}</div>",
                    unsafe_allow_html=True,
                )

            with d_col2:
                st.markdown("#### 🎯 Danh sách Ảnh Outlier (Dịch chuyển lớn nhất)")
                outlier_ids = selected_slice.get("top_outlier_image_ids", [])
                if outlier_ids:
                    for idx, oid in enumerate(outlier_ids, 1):
                        st.markdown(f"**{idx}.** `{oid}`")
                else:
                    st.write("Không có ảnh outlier đáng kể.")

            st.markdown("#### 🖼️ Thẻ Quan Sát Mẫu Ảnh Bất Thường")
            gallery_cols = st.columns(min(len(outlier_ids), 4) or 1)
            for i, oid in enumerate(outlier_ids[:4]):
                with gallery_cols[i]:
                    st.image(
                        "https://placehold.co/300x200/1e293b/ffffff.png?text=" + oid,
                        caption=f"Outlier: {oid}",
                        use_container_width=True,
                    )

    # Tab 4: Export to Module C
    with tab_export:
        st.subheader("Cung Cấp Dữ Liệu Cho Module C (toilatrung)")
        st.markdown(
            "Theo đặc tả tại [CONTRIBUTING.md](CONTRIBUTING.md) và `schemas.py`, "
            "Module C (**Correlation Validator & Performance Evaluation**) sẽ đọc bảng `ShiftScoreRecord` "
            "từ Module B để tính tương quan Spearman $\\rho$ giữa **Shift Score** và **Performance Drop (mAP Drop)**."
        )

        # Build schema records
        from backend.app.modules.slice import service as slice_service
        from backend.app.modules.slice.schemas import SliceAnalysisResponse

        mock_resp = SliceAnalysisResponse(**analysis_data)
        score_records = slice_service.export_to_shift_score_records(mock_resp)

        export_data = [r.model_dump() for r in score_records]
        export_df = pd.DataFrame(export_data)

        st.dataframe(export_df, use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Tải xuống JSON (`shift_scores.json`)",
                data=json.dumps(export_data, indent=2),
                file_name="shift_scores_module_b.json",
                mime="application/json",
            )
        with col_dl2:
            st.download_button(
                label="📥 Tải xuống CSV (`shift_scores.csv`)",
                data=export_df.to_csv(index=False),
                file_name="shift_scores_module_b.csv",
                mime="text/csv",
            )

        st.info("💡 **Gợi ý kiểm chứng liên Module:** Bạn có thể chuyển bảng này sang Module C (trang `6_Correlation_Report.py`) để kiểm tra mối tương quan với độ sụt giảm mAP!")
