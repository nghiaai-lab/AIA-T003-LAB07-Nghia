from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, func

from backend.app.core.db import Base


class CorrelationResultORM(Base):
    __tablename__ = "correlation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, index=True, nullable=False)
    spearman_rho = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    n = Column(Integer, nullable=False)
    records = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
