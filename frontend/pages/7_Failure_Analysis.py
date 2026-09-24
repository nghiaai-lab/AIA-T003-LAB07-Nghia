"""Failure Analysis — phụ trách: Trịnh Quang Trung. Xem docs/spec.md (2.10)."""
import pandas as pd
import streamlit as st

from frontend.api_client import post

st.set_page_config(page_title="Failure Analysis", layout="wide")
st.title("False Alarm / Miss Explorer")

project_id = st.text_input("Project ID", value="demo-project")

default_df = pd.DataFrame(
    [
        {"domain_or_slice": "camera_D", "shift_score": 92, "performance_drop": 0.02},
        {"domain_or_slice": "small_vehicles", "shift_score": 35, "performance_drop": 0.28},
        {"domain_or_slice": "night_rain", "shift_score": 97, "performance_drop": 0.48},
    ]
)
edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
shift_high = col1.number_input("Shift high", value=70.0)
shift_low = col2.number_input("Shift low", value=40.0)
drop_high = col3.number_input("Drop high", value=0.15)
drop_low = col4.number_input("Drop low", value=0.05)

if st.button("Phân tích"):
    try:
        result = post(
            "/failure/analyze",
            {
                "project_id": project_id,
                "records": edited_df.to_dict(orient="records"),
                "thresholds": {
                    "shift_high": shift_high,
                    "shift_low": shift_low,
                    "drop_high": drop_high,
                    "drop_low": drop_low,
                },
            },
        )
        cases_df = pd.DataFrame(result["cases"])
        st.dataframe(cases_df, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("False alarms", int((cases_df["failure_type"] == "false_alarm").sum()))
        c2.metric("Misses", int((cases_df["failure_type"] == "miss").sum()))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Lỗi khi gọi backend: {exc}")
