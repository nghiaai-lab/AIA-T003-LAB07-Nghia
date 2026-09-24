"""Export Report — phụ trách: Trịnh Quang Trung. Xem docs/spec.md (2.11)."""
import streamlit as st

from frontend.api_client import API_BASE_URL, post

st.set_page_config(page_title="Export Report", layout="wide")
st.title("Export Report")

st.markdown(
    "Gộp kết quả Performance Evaluation, Correlation Report và Failure Analysis của "
    "project thành 1 báo cáo HTML — dùng để nộp bài hoặc demo."
)

project_id = st.text_input("Project ID", value="demo-project")
project_name = st.text_input("Tên project (hiển thị trong report)", value="")

if st.button("Xuất report HTML"):
    try:
        result = post("/report/export", {"project_id": project_id, "project_name": project_name})
        download_url = f"{API_BASE_URL}/report/{project_id}/download"
        st.success(f"Đã xuất report: {result['file_path']}")
        st.markdown(f"[Tải report tại đây]({download_url})")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Lỗi khi gọi backend: {exc}")
