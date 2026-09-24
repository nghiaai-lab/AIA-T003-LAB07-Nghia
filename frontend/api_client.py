"""HTTP client dùng chung cho các trang Streamlit gọi FastAPI backend."""
import os

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def post(path: str, json: dict) -> dict:
    resp = requests.post(f"{API_BASE_URL}{path}", json=json, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get(path: str) -> dict | list:
    resp = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()
