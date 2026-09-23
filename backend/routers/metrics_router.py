from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.database_models import AnalysisResult, MetricRecord, Repository
from backend.schemas.pydantic_schemas import MetricItemSchema, MetricsSummarySchema
from backend.utils.logger import get_logger

logger = get_logger("metrics_router")

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_model=List[MetricItemSchema])
def get_metrics(
    analysis_id: Optional[int] = Query(None, description="Optional analysis ID (defaults to latest)"),
    db: Session = Depends(get_db),
):
    """Retrieves all 13+ software metrics per file for an analysis run."""
    if not analysis_id:
        latest = db.query(AnalysisResult).order_by(AnalysisResult.id.desc()).first()
        if not latest:
            return []
        analysis_id = latest.id

    records = db.query(MetricRecord).filter(MetricRecord.analysis_id == analysis_id).all()
    return records


@router.get("/metrics/{analysis_id}", response_model=List[MetricItemSchema])
def get_metrics_by_id(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieves file-level metrics for a specific analysis ID."""
    records = db.query(MetricRecord).filter(MetricRecord.analysis_id == analysis_id).all()
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metrics records found for analysis ID {analysis_id}.",
        )
    return records


@router.get("/metrics/{analysis_id}/summary", response_model=MetricsSummarySchema)
def get_metrics_summary(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieves aggregated metrics, statistics, and top risk files for a repository."""
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AnalysisResult with ID {analysis_id} not found.",
        )

    repo = db.query(Repository).filter(Repository.id == analysis.repository_id).first()
    records = db.query(MetricRecord).filter(MetricRecord.analysis_id == analysis_id).all()

    high_risk = [r for r in records if r.risk_class == "High"]
    med_risk = [r for r in records if r.risk_class == "Medium"]
    low_risk = [r for r in records if r.risk_class == "Low"]

    top_complex = sorted(records, key=lambda x: x.cyclomatic_complexity, reverse=True)[:5]
    top_churned = sorted(records, key=lambda x: x.code_churn, reverse=True)[:5]

    return MetricsSummarySchema(
        analysis_id=analysis.id,
        repository_name=repo.name if repo else "Unknown",
        total_files=analysis.total_files,
        total_loc=analysis.total_loc,
        avg_complexity=analysis.avg_complexity,
        avg_cohesion=analysis.avg_cohesion,
        avg_instability=analysis.avg_coupling,
        maintainability_index=analysis.maintainability_index,
        high_risk_files_count=len(high_risk),
        medium_risk_files_count=len(med_risk),
        low_risk_files_count=len(low_risk),
        top_complex_files=top_complex,
        top_churned_files=top_churned,
    )
