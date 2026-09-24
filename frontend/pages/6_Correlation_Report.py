"""Correlation Report — phụ trách: Trịnh Quang Trung. Xem docs/spec.md (2.9)."""
import pandas as pd
import streamlit as st

from frontend.api_client import post

st.set_page_config(page_title="Correlation Report", layout="wide")
st.title("Correlation Report")

st.markdown(
    "Nhập bảng domain/slice kèm shift score (từ Module B) và performance drop "
    "(từ Performance Evaluation) để tính Spearman ρ. Khuyến nghị 8–12 dòng để kết quả "
    "có ý nghĩa thống kê — xem docs/spec.md mục 2.9."
)

project_id = st.text_input("Project ID", value="demo-project")

default_df = pd.DataFrame(
    [
        {"domain_or_slice": "camera_B", "shift_score": 32, "performance_drop": 0.04},
        {"domain_or_slice": "rain", "shift_score": 68, "performance_drop": 0.18},
        {"domain_or_slice": "night", "shift_score": 91, "performance_drop": 0.36},
        {"domain_or_slice": "night_rain", "shift_score": 97, "performance_drop": 0.48},
    ]
)
edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)

if st.button("Tính Spearman correlation"):
    try:
        result = post(
            "/correlation/compute",
            {"project_id": project_id, "records": edited_df.to_dict(orient="records")},
        )
        st.success(
            f"Spearman ρ = {result['spearman_rho']:.3f}, "
            f"p-value = {result['p_value']:.4f}, n = {result['n']}"
        )
        st.scatter_chart(pd.DataFrame(result["records"]), x="shift_score", y="performance_drop")
        if result["n"] < 8:
            st.warning("Chỉ có ít hơn 8 domain/slice — kết quả Spearman sẽ yếu, nên bổ sung thêm.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Lỗi khi gọi backend: {exc}")
