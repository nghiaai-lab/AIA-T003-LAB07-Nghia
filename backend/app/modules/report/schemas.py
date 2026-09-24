"""Schemas cho Report Export. Xem docs/spec.md 2.11."""
from pydantic import BaseModel


class ReportRequest(BaseModel):
    project_id: str
    project_name: str = ""


class ReportResponse(BaseModel):
    project_id: str
    file_path: str
