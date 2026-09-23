from .repository_router import router as repository_router
from .analysis_router import router as analysis_router
from .metrics_router import router as metrics_router
from .graph_router import router as graph_router
from .ml_router import router as ml_router
from .dashboard_router import router as dashboard_router
from .history_router import router as history_router
from .report_router import router as report_router

__all__ = [
    "repository_router",
    "analysis_router",
    "metrics_router",
    "graph_router",
    "ml_router",
    "dashboard_router",
    "history_router",
    "report_router",
]
