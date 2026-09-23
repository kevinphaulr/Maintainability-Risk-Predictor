from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.utils.config import settings, REPORTS_DIR
from backend.utils.logger import get_logger
from backend.database.session import init_db
from backend.routers import (
    repository_router,
    analysis_router,
    metrics_router,
    graph_router,
    ml_router,
    dashboard_router,
    history_router,
    report_router,
)

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize DB tables and configure runtime."""
    logger.info("Initializing SQLite database tables...")
    init_db()
    logger.info("Database initialized successfully.")
    yield
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Predicting Software Maintainability Risk Using Repository Evolution Graph Intelligence and Explainable AI",
    lifespan=lifespan,
)

# CORS Middleware for React frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers under Root for direct endpoint matching (e.g., /clone, /analyze, /metrics, /graph, etc.)
app.include_router(repository_router)
app.include_router(analysis_router)
app.include_router(metrics_router)
app.include_router(graph_router)
app.include_router(ml_router)
app.include_router(dashboard_router)
app.include_router(history_router)
app.include_router(report_router)

# Also mount Routers under /api/v1 for standard RESTful versioning
app.include_router(repository_router, prefix=settings.API_V1_STR)
app.include_router(analysis_router, prefix=settings.API_V1_STR)
app.include_router(metrics_router, prefix=settings.API_V1_STR)
app.include_router(graph_router, prefix=settings.API_V1_STR)
app.include_router(ml_router, prefix=settings.API_V1_STR)
app.include_router(dashboard_router, prefix=settings.API_V1_STR)
app.include_router(history_router, prefix=settings.API_V1_STR)
app.include_router(report_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    """Root metadata endpoint."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "status": "online",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
