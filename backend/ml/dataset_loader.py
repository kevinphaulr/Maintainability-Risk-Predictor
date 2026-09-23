from pathlib import Path
from typing import List, Optional
import torch
from torch_geometric.data import Data

from backend.utils.config import DATASET_DIR
from backend.utils.logger import get_logger

logger = get_logger("dataset_loader")


class DatasetLoader:
    """Manages loading and batching of PyG Evolution Graph datasets from disk."""

    @staticmethod
    def load_all_graphs(dataset_dir: Optional[Path] = None) -> List[Data]:
        """Loads all saved PyTorch Geometric .pt graph files."""
        target_dir = dataset_dir or DATASET_DIR
        graph_files = list(target_dir.glob("*_pyg.pt"))
        loaded_graphs: List[Data] = []

        for gf in graph_files:
            try:
                data = torch.load(gf, weights_only=False)
                loaded_graphs.append(data)
            except Exception as e:
                logger.error(f"Error loading graph from {gf}: {e}")

        logger.info(f"Loaded {len(loaded_graphs)} PyG graphs from {target_dir}")
        return loaded_graphs
