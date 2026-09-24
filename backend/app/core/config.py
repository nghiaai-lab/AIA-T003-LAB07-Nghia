"""Cấu hình dùng chung cho backend. Xem docs/spec.md mục 2.1 Project Setup."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Domain Shift Radar"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("data/artifacts")
    source_dir: Path = Path("data/source")
    target_dir: Path = Path("data/target")
    source_data_dir: Path = Path("data/source")
    target_data_dir: Path = Path("data/target")
    db_url: str = "sqlite:///./shift_radar.db"
    default_model: str = "yolo26n.pt"
    fallback_model: str = "yolo11n.pt"
    device: str = "auto"


settings = Settings()
