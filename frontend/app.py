"""Streamlit entrypoint. Các trang thật nằm trong frontend/pages/ (multipage app)."""
import streamlit as st

st.set_page_config(page_title="Domain Shift Radar", layout="wide")

st.title("Domain Shift Radar")
st.caption("MVP — xem docs/spec.md cho đề bài đầy đủ.")

st.markdown(
    """
    Dùng menu bên trái để mở từng màn hình:

    - **Projects / Source Baseline / Batch Analysis** — Module A (BanhKhuc04)
    - **Slice Explorer** — Module B (Chien27803)
    - **Performance Evaluation / Correlation Report / Failure Analysis / Export Report** — Module C (toilatrung)
    """
)
