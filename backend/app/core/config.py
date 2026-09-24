"""Cấu hình dùng chung cho backend. Xem docs/spec.md mục 2.1 Project Setup."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "Domain Shift Radar"
    data_dir: Path = Path("data")
    db_url: str = "sqlite:///./shift_radar.db"


settings = Settings()
