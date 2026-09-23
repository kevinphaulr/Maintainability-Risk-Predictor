import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.database_models import (
    Repository,
    AnalysisResult,
    MetricRecord,
    Prediction,
)
from backend.schemas.pydantic_schemas import (
    DashboardResponse,
    RepositoryResponse,
    AnalysisStatusResponse,
    MetricsSummarySchema,
    CommitTrendItem,
    DeveloperActivityItem,
    PredictionResponse,
    RiskProbabilitySchema,
    RiskExplanationFactor,
)
from backend.services.git_service import GitService
from backend.services.xai_service import ExplainableAIService
from backend.utils.logger import get_logger

logger = get_logger("dashboard_router")

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard(
    analysis_id: Optional[int] = Query(None, description="Analysis ID (defaults to latest)"),
    db: Session = Depends(get_db),
):
    """
    Returns full dashboard data package for frontend visualization:
    - Repository health & status
    - Risk Gauge (score, classification, thresholds)
    - Maintainability Index & aggregated software metrics
    - Git commit trends & developer impact activity
    - Explainable AI prediction & actionable recommendations
    """
    if not analysis_id:
        latest = db.query(AnalysisResult).order_by(AnalysisResult.id.desc()).first()
        if not latest:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No repository analysis exists yet. Please clone and analyze a repository first.",
            )
        analysis_id = latest.id

    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found.",
        )

    repo = db.query(Repository).filter(Repository.id == analysis.repository_id).first()
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated repository record not found.",
        )

    metrics_records = db.query(MetricRecord).filter(MetricRecord.analysis_id == analysis_id).all()
    prediction_record = db.query(Prediction).filter(Prediction.analysis_id == analysis_id).first()

    # If prediction record hasn't been generated yet, compute it now
    if not prediction_record:
        ExplainableAIService.generate_explanation_and_recommendations(analysis_id, db)
        prediction_record = db.query(Prediction).filter(Prediction.analysis_id == analysis_id).first()

    # Top complex and churned files
    high_risk = [r for r in metrics_records if r.risk_class == "High"]
    med_risk = [r for r in metrics_records if r.risk_class == "Medium"]
    low_risk = [r for r in metrics_records if r.risk_class == "Low"]

    top_complex = sorted(metrics_records, key=lambda x: x.cyclomatic_complexity, reverse=True)[:5]
    top_churned = sorted(metrics_records, key=lambda x: x.code_churn, reverse=True)[:5]

    metrics_summary = MetricsSummarySchema(
        analysis_id=analysis.id,
        repository_name=repo.name,
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

    # Git trends and developer activity
    git_history = GitService.analyze_git_history(Path(repo.local_path))
    commit_trends = [CommitTrendItem(**t) for t in git_history.get("commit_trends", [])]
    developer_activity = [DeveloperActivityItem(**d) for d in git_history.get("developer_activity", [])]

    # Prediction payload
    pred_response = None
    if prediction_record:
        pred_response = PredictionResponse(
            analysis_id=analysis.id,
            risk_score=prediction_record.risk_score,
            risk_class=prediction_record.risk_class,
            confidence=prediction_record.confidence,
            probabilities=RiskProbabilitySchema(
                low=prediction_record.probability_low,
                medium=prediction_record.probability_medium,
                high=prediction_record.probability_high,
            ),
            explanations=[
                RiskExplanationFactor(**exp)
                for exp in json.loads(prediction_record.explanations_json)
            ],
            recommendations=json.loads(prediction_record.recommendations_json),
        )

    quick_recommendations = (
        pred_response.recommendations if pred_response else ["Maintain current modular code structure."]
    )

    maintainability_gauge = {
        "score": analysis.maintainability_index,
        "risk_score": analysis.overall_risk_score,
        "risk_class": analysis.risk_class,
        "color": "#16a34a" if analysis.risk_class == "Low" else ("#d97706" if analysis.risk_class == "Medium" else "#dc2626"),
        "label": f"{analysis.risk_class} Risk",
    }

    return DashboardResponse(
        repository=RepositoryResponse.model_validate(repo),
        analysis=AnalysisStatusResponse.model_validate(analysis),
        maintainability_gauge=maintainability_gauge,
        metrics_summary=metrics_summary,
        commit_trends=commit_trends,
        developer_activity=developer_activity,
        prediction=pred_response,
        quick_recommendations=quick_recommendations[:4],
    )


@router.get("/dashboard/{analysis_id}", response_model=DashboardResponse)
def get_dashboard_by_id(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieves full dashboard view for a specific analysis ID."""
    return get_dashboard(analysis_id=analysis_id, db=db)


@router.get("/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "Maintainability Risk Predictor",
        "version": "1.0.0",
    }
