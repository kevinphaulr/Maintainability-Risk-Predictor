import json
import datetime
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.database_models import AnalysisResult, Prediction, TrainingHistory
from backend.schemas.pydantic_schemas import (
    TrainRequest,
    TrainResponse,
    PredictRequest,
    PredictionResponse,
    RiskProbabilitySchema,
    RiskExplanationFactor,
)
from backend.ml.trainer import GATTrainer, DEFAULT_MODEL_PATH
from backend.ml.predictor import GATPredictor
from backend.utils.config import TRAINED_MODELS_DIR
from backend.utils.logger import get_logger

logger = get_logger("ml_router")

router = APIRouter(tags=["Machine Learning & Predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict_maintainability_risk(payload: PredictRequest, db: Session = Depends(get_db)):
    """
    Generates explainable maintainability risk prediction for a repository analysis:
    - Loads trained Graph Attention Network (GAT) model
    - Runs graph inference over the repository evolution graph
    - Computes Risk Score (0-100), Risk Class (Low, Medium, High)
    - Generates Softmax Probability Distribution & Confidence
    - Performs XAI feature attribution and produces concrete refactoring guidance.
    """
    try:
        result = GATPredictor.predict_repository_analysis(payload.analysis_id, db)
        return PredictionResponse(
            analysis_id=result["analysis_id"],
            risk_score=result["risk_score"],
            risk_class=result["risk_class"],
            confidence=result["confidence"],
            probabilities=RiskProbabilitySchema(**result["probabilities"]),
            explanations=[RiskExplanationFactor(**exp) for exp in result["explanations"]],
            recommendations=result["recommendations"],
        )
    except Exception as e:
        logger.error(f"Prediction failed for analysis {payload.analysis_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to generate prediction: {str(e)}",
        )


@router.post("/train", response_model=TrainResponse)
def trigger_training(payload: TrainRequest, db: Session = Depends(get_db)):
    """
    Orchestration endpoint for Graph Attention Network (GAT) training pipeline:
    - Loads Repository Evolution Graph dataset
    - Runs multi-epoch GAT training loop with Adam optimizer and ReduceLROnPlateau scheduler
    - Evaluates on test set calculating Accuracy, Precision, Recall, F1, and Confusion Matrix
    - Saves trained model checkpoint to trained_models/gat_model.pt
    - Persists training run in SQLite training_history table.
    """
    logger.info(
        f"Initiating real GAT training run: epochs={payload.epochs}, lr={payload.learning_rate}, hidden_dim={payload.hidden_dim}"
    )

    try:
        trainer = GATTrainer(
            in_channels=13,
            hidden_channels=payload.hidden_dim,
            num_classes=3,
            heads=4,
            dropout=0.2,
            learning_rate=payload.learning_rate,
        )

        training_results = trainer.run_training(
            epochs=payload.epochs,
            batch_size=8,
            save_path=DEFAULT_MODEL_PATH,
        )

        # Log into SQLite training_history table
        history_record = TrainingHistory(
            model_name=training_results["model_name"],
            epochs=training_results["epochs"],
            train_loss=training_results["train_loss"],
            val_loss=training_results["val_loss"],
            accuracy=training_results["accuracy"],
            precision=training_results["precision"],
            recall=training_results["recall"],
            f1_score=training_results["f1_score"],
            confusion_matrix_json=json.dumps(training_results["confusion_matrix"]),
            training_curves_json=json.dumps(training_results["training_curves"]),
            model_checkpoint_path=training_results["checkpoint_path"],
        )
        db.add(history_record)
        db.commit()
        db.refresh(history_record)

        return TrainResponse(
            status=training_results["status"],
            model_name=training_results["model_name"],
            epochs=training_results["epochs"],
            train_loss=training_results["train_loss"],
            val_loss=training_results["val_loss"],
            accuracy=training_results["accuracy"],
            precision=training_results["precision"],
            recall=training_results["recall"],
            f1_score=training_results["f1_score"],
            confusion_matrix=training_results["confusion_matrix"],
            training_curves=training_results["training_curves"],
        )
    except Exception as e:
        logger.error(f"GAT training failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training pipeline execution failed: {str(e)}",
        )


@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    """Lists trained model history and active checkpoints."""
    history = db.query(TrainingHistory).order_by(TrainingHistory.id.desc()).all()
    return [
        {
            "id": h.id,
            "model_name": h.model_name,
            "epochs": h.epochs,
            "accuracy": h.accuracy,
            "precision": h.precision,
            "recall": h.recall,
            "f1_score": h.f1_score,
            "train_loss": h.train_loss,
            "val_loss": h.val_loss,
            "trained_at": h.trained_at,
            "checkpoint": h.model_checkpoint_path,
        }
        for h in history
    ]
