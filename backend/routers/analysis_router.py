from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.database_models import AnalysisResult, Repository
from backend.schemas.pydantic_schemas import AnalysisTriggerRequest, AnalysisStatusResponse
from backend.services.evolution_graph_service import EvolutionGraphService
from backend.services.xai_service import ExplainableAIService
from backend.utils.logger import get_logger

logger = get_logger("analysis_router")

router = APIRouter(tags=["Analysis"])


@router.post("/analyze", response_model=AnalysisStatusResponse, status_code=status.HTTP_200_OK)
def trigger_analysis(payload: AnalysisTriggerRequest, db: Session = Depends(get_db)):
    """
    Triggers end-to-end repository analysis:
    Git mining -> AST parsing -> metrics calculation -> Evolution Graph construction -> PyG tensor export -> XAI generation.
    """
    repo = db.query(Repository).filter(Repository.id == payload.repository_id).first()
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository with ID {payload.repository_id} not found.",
        )

    try:
        # Run analysis pipeline
        result = EvolutionGraphService.analyze_repository(repo.id, db)
        analysis_id = result["analysis_id"]

        # Automatically generate Explainable AI explanations and baseline recommendations
        ExplainableAIService.generate_explanation_and_recommendations(analysis_id, db)

        analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
        return analysis
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline failed: {str(e)}",
        )


@router.get("/analysis/{analysis_id}/status", response_model=AnalysisStatusResponse)
def get_analysis_status(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieves status and high-level outcomes of a specific analysis run."""
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AnalysisResult with ID {analysis_id} not found.",
        )
    return analysis


@router.get("/analysis/{analysis_id}", response_model=AnalysisStatusResponse)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieves full analysis summary record."""
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AnalysisResult with ID {analysis_id} not found.",
        )
    return analysis
