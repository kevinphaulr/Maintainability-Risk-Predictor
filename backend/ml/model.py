import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from backend.utils.logger import get_logger

logger = get_logger("gat_model")


class GATMaintainabilityPredictor(nn.Module):
    """
    Graph Attention Network (GAT) for Software Maintainability Risk Prediction.
    
    Architecture:
    - Input Layer: 13 normalized software engineering & evolution metrics
    - Layer 1: GATConv with multi-head attention + ELU + Dropout
    - Layer 2: GATConv with multi-head attention + ELU + Dropout
    - Layer 3: GATConv aggregating multi-head representations + ELU
    - Global Pooling: Concatenation of global_mean_pool and global_max_pool
    - Dense MLP: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> ReLU
    - Output Head 1 (Classifier): 3 Logits (Low Risk, Medium Risk, High Risk)
    - Output Head 2 (Regressor): Continuous risk score normalized [0.0, 1.0]
    """

    def __init__(
        self,
        in_channels: int = 13,
        hidden_channels: int = 64,
        num_classes: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        edge_dim: Optional[int] = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.heads = heads
        self.dropout_rate = dropout
        self.edge_dim = edge_dim

        # GAT Convolutional Layers
        # Layer 1: in_channels -> hidden_channels * heads
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
            add_self_loops=False,
        )

        # Layer 2: (hidden_channels * heads) -> hidden_channels * heads
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
            add_self_loops=False,
        )

        # Layer 3: (hidden_channels * heads) -> hidden_channels (single head for aggregation)
        self.conv3 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False,
            add_self_loops=False,
        )

        # Global pooling combines mean and max pooling -> 2 * hidden_channels
        pooled_dim = hidden_channels * 2

        # Fully Connected MLP Layers
        self.fc1 = nn.Linear(pooled_dim, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)

        # Output Prediction Heads
        self.classifier = nn.Linear(hidden_channels // 2, num_classes)
        self.regressor = nn.Linear(hidden_channels // 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GAT network.
        
        Args:
            x: Node feature tensor [num_nodes, in_channels]
            edge_index: Graph connectivity tensor [2, num_edges]
            batch: Batch vector assigning each node to a graph [num_nodes]
            edge_attr: Edge attribute tensor [num_edges, edge_dim]
            
        Returns:
            Tuple of (class_logits [batch_size, num_classes], continuous_risk [batch_size, 1])
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Ensure edge_index and edge_attr match expected empty shape if no edges
        if edge_index is None or edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            if self.edge_dim is not None:
                edge_attr = torch.empty((0, self.edge_dim), dtype=torch.float, device=x.device)

        # 1. First GAT Attention Block
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # 2. Second GAT Attention Block
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # 3. Third GAT Aggregation Block
        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        h = F.elu(h)

        # 4. Global Graph Readout / Pooling (Mean + Max)
        mean_pooled = global_mean_pool(h, batch)
        max_pooled = global_max_pool(h, batch)
        graph_repr = torch.cat([mean_pooled, max_pooled], dim=1)

        # 5. Dense Transformation Layers
        z = self.fc1(graph_repr)
        if z.size(0) > 1:
            z = self.bn1(z)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = F.relu(self.fc2(z))

        # 6. Prediction Heads
        class_logits = self.classifier(z)
        risk_score = torch.sigmoid(self.regressor(z))

        return class_logits, risk_score

    def save_checkpoint(self, checkpoint_path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Saves model weights, hyperparameter configuration, and training metadata."""
        checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.state_dict(),
            "config": {
                "in_channels": self.in_channels,
                "hidden_channels": self.hidden_channels,
                "num_classes": self.num_classes,
                "heads": self.heads,
                "dropout": self.dropout_rate,
                "edge_dim": self.edge_dim,
            },
            "metadata": metadata or {},
        }
        torch.save(payload, checkpoint_path)
        logger.info(f"Model checkpoint successfully saved to {checkpoint_path}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path, device: torch.device = torch.device("cpu")) -> "GATMaintainabilityPredictor":
        """Loads model instance from a saved checkpoint file."""
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        
        model = cls(
            in_channels=config.get("in_channels", 13),
            hidden_channels=config.get("hidden_channels", 64),
            num_classes=config.get("num_classes", 3),
            heads=config.get("heads", 4),
            dropout=config.get("dropout", 0.2),
            edge_dim=config.get("edge_dim", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully from {checkpoint_path}")
        return model
