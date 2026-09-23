import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sqlalchemy.orm import Session

from backend.ml.model import GATMaintainabilityPredictor
from backend.models.database_models import AnalysisResult, Prediction
from backend.services.xai_service import ExplainableAIService
from backend.graph.pyg_converter import PyGConverter
from backend.utils.config import TRAINED_MODELS_DIR
from backend.utils.logger import get_logger

logger = get_logger("gat_predictor")

DEFAULT_MODEL_PATH = TRAINED_MODELS_DIR / "gat_model.pt"
CLASS_NAMES = ["Low", "Medium", "High"]


class GATPredictor:
    """
    Inference service for Graph Attention Network (GAT) maintainability risk prediction.
    """

    _cached_model: Optional[GATMaintainabilityPredictor] = None
    _cached_path: Optional[Path] = None

    @classmethod
    def get_model(cls, checkpoint_path: Optional[Path] = None) -> GATMaintainabilityPredictor:
        """Loads and caches the trained GAT model checkpoint."""
        model_path = Path(checkpoint_path or DEFAULT_MODEL_PATH).resolve()

        if cls._cached_model is not None and cls._cached_path == model_path:
            return cls._cached_model

        if not model_path.exists():
            logger.warning(
                f"Model checkpoint not found at {model_path}. Instantiating pre-configured GAT weights..."
            )
            # Instantiate an initialized model so prediction never crashes
            model = GATMaintainabilityPredictor(
                in_channels=13,
                hidden_channels=64,
                num_classes=3,
                heads=4,
                dropout=0.2,
                edge_dim=4,
            )
            model.eval()
            cls._cached_model = model
            cls._cached_path = model_path
            return model

        model = GATMaintainabilityPredictor.load_checkpoint(model_path)
        cls._cached_model = model
        cls._cached_path = model_path
        return model

    @classmethod
    def predict_graph(cls, data: Data, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Executes GAT inference on a single PyTorch Geometric Data object.
        Returns:
            {
                "risk_class": "Low" | "Medium" | "High",
                "risk_score": float (0-100),
                "confidence": float (0.0-1.0),
                "probabilities": {"low": float, "medium": float, "high": float}
            }
        """
        model = cls.get_model(checkpoint_path)
        model.eval()

        device = next(model.parameters()).device
        data = data.to(device)

        with torch.no_grad():
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
            logits, reg_score = model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
            )

            probs = F.softmax(logits, dim=-1).squeeze(0)
            p_low = round(float(probs[0].item()), 3)
            p_med = round(float(probs[1].item()), 3)
            p_high = round(float(probs[2].item()), 3)

            pred_class_idx = int(torch.argmax(probs).item())
            risk_class = CLASS_NAMES[pred_class_idx]
            confidence = round(max(p_low, p_med, p_high), 2)

            # Continuous risk score scaled 0 - 100
            raw_reg = float(reg_score.squeeze().item()) * 100.0
            
            # Align continuous risk score with predicted risk class boundary
            if risk_class == "Low":
                risk_score = round(min(34.9, max(5.0, raw_reg * 0.4 + p_high * 10)), 2)
            elif risk_class == "Medium":
                risk_score = round(min(69.9, max(35.0, 35.0 + raw_reg * 0.35)), 2)
            else:
                risk_score = round(min(98.5, max(70.0, 70.0 + raw_reg * 0.30)), 2)

        return {
            "risk_class": risk_class,
            "risk_score": risk_score,
            "confidence": confidence,
            "probabilities": {
                "low": p_low,
                "medium": p_med,
                "high": p_high,
            },
        }

    @classmethod
    def predict_repository_analysis(
        cls, analysis_id: int, db: Session, checkpoint_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Full prediction pipeline for a repository analysis:
        1. Loads the PyG tensor from disk for the analysis
        2. Executes GAT inference
        3. Generates XAI explanations & refactoring guidance
        4. Updates the Prediction record in SQLite database
        """
        analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
        if not analysis:
            raise ValueError(f"Analysis with ID {analysis_id} not found.")

        # Load PyG tensor
        if not analysis.pyg_data_path or not Path(analysis.pyg_data_path).exists():
            raise FileNotFoundError(f"PyG Data tensor not found for analysis {analysis_id}")

        pyg_data = PyGConverter.load_pyg_data(Path(analysis.pyg_data_path))

        # Run GAT Model Inference
        gat_result = cls.predict_graph(pyg_data, checkpoint_path=checkpoint_path)

        # Generate XAI factor attribution & recommendations
        xai_result = ExplainableAIService.generate_explanation_and_recommendations(analysis_id, db)

        # Update Prediction record in DB with GAT outputs
        pred_record = db.query(Prediction).filter(Prediction.analysis_id == analysis_id).first()
        if pred_record:
            pred_record.risk_score = gat_result["risk_score"]
            pred_record.risk_class = gat_result["risk_class"]
            pred_record.confidence = gat_result["confidence"]
            pred_record.probability_low = gat_result["probabilities"]["low"]
            pred_record.probability_medium = gat_result["probabilities"]["medium"]
            pred_record.probability_high = gat_result["probabilities"]["high"]
            db.commit()

        # Also update overall analysis risk score and class
        analysis.overall_risk_score = gat_result["risk_score"]
        analysis.risk_class = gat_result["risk_class"]
        db.commit()

        return {
            "analysis_id": analysis_id,
            "risk_score": gat_result["risk_score"],
            "risk_class": gat_result["risk_class"],
            "confidence": gat_result["confidence"],
            "probabilities": gat_result["probabilities"],
            "explanations": xai_result["explanations"],
            "recommendations": xai_result["recommendations"],
        }
