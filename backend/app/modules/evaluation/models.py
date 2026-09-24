from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, func

from backend.app.core.db import Base


class EvaluationResultORM(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, index=True, nullable=False)
    domain_or_slice = Column(String, index=True, nullable=False)
    target_metric = Column(Float, nullable=False)
    source_metric = Column(Float, nullable=False)
    performance_drop = Column(Float, nullable=False)
    ap_per_class = Column(JSON, nullable=False, default=dict)
    num_predictions = Column(Integer, nullable=False)
    num_ground_truth = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
