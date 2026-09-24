"""FastAPI entrypoint — gắn router của từng module. Xem CONTRIBUTING.md cho ranh giới module."""
from fastapi import FastAPI

from backend.app.core.config import settings
from backend.app.modules.batch.router import router as batch_router
from backend.app.modules.correlation.router import router as correlation_router
from backend.app.modules.evaluation.router import router as evaluation_router
from backend.app.modules.failure.router import router as failure_router
from backend.app.modules.report.router import router as report_router
from backend.app.modules.shift.router import router as shift_router
from backend.app.modules.slice.router import router as slice_router
from backend.app.modules.source.router import router as source_router

app = FastAPI(title=settings.app_name)

# Module A — BanhKhuc04
app.include_router(source_router)
app.include_router(batch_router)

# Module B — Chien27803
app.include_router(shift_router)
app.include_router(slice_router)

# Module C — toilatrung
app.include_router(evaluation_router)
app.include_router(correlation_router)
app.include_router(failure_router)
app.include_router(report_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}
