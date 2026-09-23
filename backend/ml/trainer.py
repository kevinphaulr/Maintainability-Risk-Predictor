import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from backend.ml.model import GATMaintainabilityPredictor
from backend.ml.dataset import MaintainabilityGraphDataset
from backend.utils.config import TRAINED_MODELS_DIR
from backend.utils.logger import get_logger

logger = get_logger("gat_trainer")

DEFAULT_MODEL_PATH = TRAINED_MODELS_DIR / "gat_model.pt"


class GATTrainer:
    """
    Orchestrates the end-to-end training, validation, testing, and checkpointing
    pipeline for the Graph Attention Network (GAT).
    """

    def __init__(
        self,
        in_channels: int = 13,
        hidden_channels: int = 64,
        num_classes: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GATMaintainabilityPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            heads=heads,
            dropout=dropout,
            edge_dim=4,
        ).to(self.device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()

    def train_epoch(self, train_loader) -> float:
        """Trains the GAT model for one single epoch."""
        self.model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits, pred_risk = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                edge_attr=batch.edge_attr,
            )

            # Combined Classification + Auxiliary Regression Loss
            loss_cls = self.cls_criterion(logits, batch.y)
            loss_reg = self.reg_criterion(pred_risk.squeeze(-1), batch.risk_score)
            loss = loss_cls + 0.3 * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        return round(total_loss / max(1, batches), 4)

    def evaluate(self, val_loader) -> Tuple[float, List[int], List[int]]:
        """Evaluates loss and extracts predictions over a validation or test DataLoader."""
        self.model.eval()
        total_loss = 0.0
        batches = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                logits, pred_risk = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    edge_attr=batch.edge_attr,
                )

                loss_cls = self.cls_criterion(logits, batch.y)
                loss_reg = self.reg_criterion(pred_risk.squeeze(-1), batch.risk_score)
                loss = loss_cls + 0.3 * loss_reg

                total_loss += loss.item()
                batches += 1

                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                targets = batch.y.cpu().tolist()

                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_loss = round(total_loss / max(1, batches), 4)
        return avg_loss, all_preds, all_targets

    def run_training(
        self,
        epochs: int = 50,
        batch_size: int = 8,
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete training workflow:
        1. Loads & splits Repository Evolution Graphs into Train/Val/Test
        2. Trains model over requested epochs
        3. Tracks training and validation loss convergence curves
        4. Evaluates final performance on unseen Test set
        5. Computes Accuracy, Precision, Recall, F1 Score, and 3x3 Confusion Matrix
        6. Saves trained model checkpoint to disk (trained_models/gat_model.pt)
        """
        save_file = Path(save_path or DEFAULT_MODEL_PATH).resolve()
        logger.info(f"Starting GAT Training: epochs={epochs}, batch_size={batch_size}, device={self.device}")

        # 1. Prepare Dataset and DataLoaders
        dataset_manager = MaintainabilityGraphDataset()
        train_loader, val_loader, test_loader = dataset_manager.get_train_val_test_split(
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, batch_size=batch_size
        )

        train_losses: List[float] = []
        val_losses: List[float] = []

        # 2. Multi-epoch training loop
        for epoch in range(1, epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_loss, _, _ = self.evaluate(val_loader)
            self.scheduler.step(v_loss)

            train_losses.append(t_loss)
            val_losses.append(v_loss)

            if epoch % 5 == 0 or epoch == epochs or epoch == 1:
                logger.info(f"Epoch {epoch:03d}/{epochs:03d} - Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

        # 3. Test Evaluation on Unseen Test Split
        test_loss, test_preds, test_targets = self.evaluate(test_loader)
        accuracy = round(float(accuracy_score(test_targets, test_preds)), 4)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_targets, test_preds, average="weighted", zero_division=0
        )
        conf_matrix = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2]).tolist()

        logger.info(
            f"Test Evaluation: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        # 4. Save Checkpoint
        metadata = {
            "epochs": epochs,
            "accuracy": accuracy,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
        }
        self.model.save_checkpoint(save_file, metadata=metadata)

        # Downsample curve points if many epochs for clean frontend rendering
        curve_stride = max(1, len(train_losses) // 15)
        sampled_train_loss = train_losses[::curve_stride]
        sampled_val_loss = val_losses[::curve_stride]
        if train_losses[-1] not in sampled_train_loss:
            sampled_train_loss.append(train_losses[-1])
            sampled_val_loss.append(val_losses[-1])

        return {
            "status": "completed",
            "model_name": "GAT-MaintainabilityPredictor",
            "epochs": epochs,
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1],
            "accuracy": accuracy,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": conf_matrix,
            "training_curves": {
                "train_loss": sampled_train_loss,
                "val_loss": sampled_val_loss,
            },
            "checkpoint_path": str(save_file),
        }
