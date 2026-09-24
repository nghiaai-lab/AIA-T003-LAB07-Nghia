from sqlalchemy import Column, DateTime, Float, Integer, String, func

from backend.app.core.db import Base


class FailureCaseORM(Base):
    __tablename__ = "failure_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, index=True, nullable=False)
    domain_or_slice = Column(String, nullable=False)
    shift_score = Column(Float, nullable=False)
    performance_drop = Column(Float, nullable=False)
    failure_type = Column(String, nullable=False)
    note = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
