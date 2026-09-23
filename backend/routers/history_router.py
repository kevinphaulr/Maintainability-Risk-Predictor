from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.database_models import AnalysisResult, Repository
from backend.schemas.pydantic_schemas import HistoryItemResponse
from backend.utils.logger import get_logger

logger = get_logger("history_router")

router = APIRouter(tags=["History"])


@router.get("/history", response_model=List[HistoryItemResponse])
def get_analysis_history(db: Session = Depends(get_db)):
    """Retrieves all past repository analyses in reverse chronological order."""
    results = db.query(AnalysisResult).order_by(AnalysisResult.id.desc()).all()
    history_items: List[HistoryItemResponse] = []

    for item in results:
        repo = db.query(Repository).filter(Repository.id == item.repository_id).first()
        history_items.append(
            HistoryItemResponse(
                id=item.id,
                repository_id=item.repository_id,
                repository_name=repo.name if repo else "Unknown",
                repository_url=repo.url if repo else None,
                analyzed_at=item.analyzed_at,
                total_files=item.total_files,
                total_loc=item.total_loc,
                overall_risk_score=item.overall_risk_score,
                risk_class=item.risk_class,
                maintainability_index=item.maintainability_index,
            )
        )

    return history_items


@router.get("/history/{analysis_id}", response_model=HistoryItemResponse)
def get_history_detail(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieves metadata of a single historical analysis run."""
    item = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found.",
        )

    repo = db.query(Repository).filter(Repository.id == item.repository_id).first()
    return HistoryItemResponse(
        id=item.id,
        repository_id=item.repository_id,
        repository_name=repo.name if repo else "Unknown",
        repository_url=repo.url if repo else None,
        analyzed_at=item.analyzed_at,
        total_files=item.total_files,
        total_loc=item.total_loc,
        overall_risk_score=item.overall_risk_score,
        risk_class=item.risk_class,
        maintainability_index=item.maintainability_index,
    )
