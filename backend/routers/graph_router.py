import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
import torch

from backend.database.session import get_db
from backend.models.database_models import AnalysisResult
from backend.schemas.pydantic_schemas import GraphDataResponse, PyGDataSummaryResponse
from backend.graph.pyg_converter import PyGConverter, FEATURE_NAMES
from backend.utils.logger import get_logger

logger = get_logger("graph_router")

router = APIRouter(tags=["Graph"])


@router.get("/graph", response_model=GraphDataResponse)
def get_graph(
    analysis_id: Optional[int] = Query(None, description="Analysis ID (defaults to latest)"),
    db: Session = Depends(get_db),
):
    """Retrieves Repository Evolution Graph in React Flow compatible format."""
    if not analysis_id:
        latest = db.query(AnalysisResult).order_by(AnalysisResult.id.desc()).first()
        if not latest:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No analyzed repositories found.",
            )
        analysis_id = latest.id

    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis or not analysis.graph_json_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph data not found for analysis ID {analysis_id}.",
        )

    json_path = Path(analysis.graph_json_path)
    if not json_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Graph file on disk is missing.",
        )

    with open(json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    return GraphDataResponse(
        analysis_id=analysis.id,
        node_count=graph_data.get("node_count", 0),
        edge_count=graph_data.get("edge_count", 0),
        nodes=graph_data.get("nodes", []),
        edges=graph_data.get("edges", []),
        summary={
            "maintainability_index": analysis.maintainability_index,
            "overall_risk_score": analysis.overall_risk_score,
            "risk_class": analysis.risk_class,
        },
    )


@router.get("/graph/{analysis_id}", response_model=GraphDataResponse)
def get_graph_by_id(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieves graph for a specific analysis ID."""
    return get_graph(analysis_id=analysis_id, db=db)


@router.get("/graph/{analysis_id}/pyg-summary", response_model=PyGDataSummaryResponse)
def get_pyg_summary(analysis_id: int, db: Session = Depends(get_db)):
    """Inspects the PyTorch Geometric Data tensor structure generated for GAT training."""
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis or not analysis.pyg_data_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PyG data not found for analysis ID {analysis_id}.",
        )

    pt_path = Path(analysis.pyg_data_path)
    if not pt_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PyG .pt file is missing on disk.",
        )

    try:
        pyg_data = PyGConverter.load_pyg_data(pt_path)
        num_nodes = pyg_data.num_nodes
        num_edges = pyg_data.num_edges
        node_dim = pyg_data.x.shape[1] if pyg_data.x is not None else 0
        edge_dim = pyg_data.edge_attr.shape[1] if pyg_data.edge_attr is not None else 0

        # Check isolated nodes
        edge_index = pyg_data.edge_index
        connected_nodes = set(edge_index[0].tolist() + edge_index[1].tolist()) if num_edges > 0 else set()
        has_isolated = len(connected_nodes) < num_nodes

        return PyGDataSummaryResponse(
            analysis_id=analysis.id,
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_feature_dim=node_dim,
            edge_feature_dim=edge_dim,
            feature_names=FEATURE_NAMES,
            has_isolated_nodes=has_isolated,
            is_directed=True,
        )
    except Exception as e:
        logger.error(f"Error reading PyG data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to inspect PyG tensor: {str(e)}",
        )
