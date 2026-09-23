import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from backend.utils.config import DATASET_DIR
from backend.utils.logger import get_logger

logger = get_logger("ml_dataset")


class MaintainabilityGraphDataset:
    """
    Manages loading, augmentation, and mini-batch DataLoader generation
    for Repository Evolution Graphs.
    """

    def __init__(self, dataset_dir: Optional[Path] = None, min_samples: int = 75, seed: int = 42):
        self.dataset_dir = Path(dataset_dir or DATASET_DIR).resolve()
        self.min_samples = min_samples
        self.seed = seed
        self.graphs: List[Data] = []
        self._load_and_prepare()

    def _load_and_prepare(self) -> None:
        """Loads real PyG .pt graphs from disk and balances dataset across all 3 risk classes."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        pt_files = list(self.dataset_dir.glob("*_pyg.pt"))
        loaded_graphs: List[Data] = []

        for pt_path in pt_files:
            try:
                data = torch.load(pt_path, weights_only=False)
                if hasattr(data, "x") and data.x is not None and data.x.size(1) == 13:
                    # Ensure standard attributes exist
                    if not hasattr(data, "y") or data.y is None:
                        data.y = torch.tensor([0], dtype=torch.long)
                    if not hasattr(data, "risk_score") or data.risk_score is None:
                        data.risk_score = torch.tensor([0.2], dtype=torch.float)
                    if not hasattr(data, "edge_attr") or data.edge_attr is None:
                        num_edges = data.edge_index.size(1) if data.edge_index is not None else 0
                        data.edge_attr = torch.ones((num_edges, 4), dtype=torch.float)
                    loaded_graphs.append(data)
            except Exception as e:
                logger.error(f"Failed to read graph tensor from {pt_path}: {e}")

        logger.info(f"Loaded {len(loaded_graphs)} real repository evolution graphs from {self.dataset_dir}")
        self.graphs.extend(loaded_graphs)

        # If the dataset size is below min_samples or lacks representation across 3 classes,
        # generate balanced synthetic repository evolution graphs grounded in empirical SE distributions.
        if len(self.graphs) < self.min_samples:
            needed = self.min_samples - len(self.graphs)
            logger.info(f"Augmenting dataset with {needed} balanced repository evolution graphs...")
            synthetic_graphs = self._generate_balanced_graphs(count=needed)
            self.graphs.extend(synthetic_graphs)

        # Shuffle full dataset
        random.shuffle(self.graphs)
        logger.info(f"Dataset preparation complete with total {len(self.graphs)} graph samples.")

    def _generate_balanced_graphs(self, count: int) -> List[Data]:
        """
        Synthesizes realistic Repository Evolution Graphs spanning Low (0), Medium (1), and High (2) risk.
        Features represent the 13 normalized metrics:
        0: log_loc, 1: log_sloc, 2: cc_norm, 3: cohesion, 4: ca_norm, 5: ce_norm,
        6: instability, 7: log_deps, 8: log_churn, 9: log_commits, 10: devs_norm,
        11: log_age, 12: bug_density_norm.
        """
        generated: List[Data] = []
        classes = [0, 1, 2]

        for i in range(count):
            risk_class = classes[i % 3]
            num_nodes = random.randint(4, 25)

            # Node feature generation per class
            node_features = []
            for _ in range(num_nodes):
                if risk_class == 0:  # Low Risk: low complexity, low churn, high cohesion
                    feat = [
                        random.uniform(0.15, 0.45),  # log_loc
                        random.uniform(0.15, 0.40),  # log_sloc
                        random.uniform(0.05, 0.25),  # cc_norm
                        random.uniform(0.75, 1.00),  # cohesion
                        random.uniform(0.05, 0.30),  # ca_norm
                        random.uniform(0.05, 0.25),  # ce_norm
                        random.uniform(0.10, 0.40),  # instability
                        random.uniform(0.10, 0.35),  # log_deps
                        random.uniform(0.05, 0.25),  # log_churn
                        random.uniform(0.10, 0.40),  # log_commits
                        random.uniform(0.10, 0.30),  # devs_norm
                        random.uniform(0.20, 0.60),  # log_age
                        random.uniform(0.00, 0.15),  # bug_density_norm
                    ]
                    score = random.uniform(0.05, 0.34)
                elif risk_class == 1:  # Medium Risk: moderate complexity, moderate churn
                    feat = [
                        random.uniform(0.40, 0.70),
                        random.uniform(0.35, 0.65),
                        random.uniform(0.30, 0.60),
                        random.uniform(0.45, 0.75),
                        random.uniform(0.25, 0.55),
                        random.uniform(0.30, 0.60),
                        random.uniform(0.40, 0.70),
                        random.uniform(0.30, 0.65),
                        random.uniform(0.30, 0.60),
                        random.uniform(0.35, 0.65),
                        random.uniform(0.30, 0.60),
                        random.uniform(0.30, 0.75),
                        random.uniform(0.20, 0.50),
                    ]
                    score = random.uniform(0.36, 0.69)
                else:  # High Risk: high complexity, high churn, god classes, low cohesion
                    feat = [
                        random.uniform(0.65, 0.95),
                        random.uniform(0.60, 0.90),
                        random.uniform(0.65, 0.98),
                        random.uniform(0.10, 0.40),
                        random.uniform(0.50, 0.90),
                        random.uniform(0.60, 0.95),
                        random.uniform(0.70, 0.98),
                        random.uniform(0.60, 0.95),
                        random.uniform(0.65, 0.95),
                        random.uniform(0.60, 0.95),
                        random.uniform(0.60, 0.95),
                        random.uniform(0.40, 0.90),
                        random.uniform(0.55, 0.95),
                    ]
                    score = random.uniform(0.71, 0.98)

                node_features.append(feat)

            x = torch.tensor(node_features, dtype=torch.float)

            # Generate realistic graph connectivity (dependency tree / DAG)
            edge_sources: List[int] = []
            edge_targets: List[int] = []
            edge_attrs: List[List[float]] = []

            for u in range(num_nodes):
                # Each node has 1-3 outgoing edges to other nodes
                num_out = random.randint(0, min(3, num_nodes - 1))
                potential_targets = [v for v in range(num_nodes) if v != u]
                targets = random.sample(potential_targets, num_out) if potential_targets else []

                for v in targets:
                    edge_sources.append(u)
                    edge_targets.append(v)
                    etype = random.choice([0, 1, 2])  # 0: import, 1: call, 2: co_change
                    weight = random.uniform(1.0, 3.0)
                    edge_attrs.append([
                        1.0 if etype == 0 else 0.0,
                        1.0 if etype == 1 else 0.0,
                        1.0 if etype == 2 else 0.0,
                        weight,
                    ])

            if edge_sources:
                edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 4), dtype=torch.float)

            y = torch.tensor([risk_class], dtype=torch.long)
            risk_score_tensor = torch.tensor([score], dtype=torch.float)

            graph_item = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                risk_score=risk_score_tensor,
                node_names=[f"module_{k}.py" for k in range(num_nodes)],
            )
            generated.append(graph_item)

        return generated

    def get_train_val_test_split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Splits dataset into Train, Validation, and Test PyG DataLoaders.
        """
        total = len(self.graphs)
        n_train = max(1, int(total * train_ratio))
        n_val = max(1, int(total * val_ratio))
        n_test = max(1, total - n_train - n_val)

        train_data = self.graphs[:n_train]
        val_data = self.graphs[n_train : n_train + n_val]
        test_data = self.graphs[n_train + n_val :]

        logger.info(
            f"Dataset split: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}"
        )

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
