import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import torch
from torch_geometric.data import Data

from backend.utils.logger import get_logger

logger = get_logger("pyg_converter")

# 13 Standard Features for Node Embeddings
FEATURE_NAMES = [
    "log_loc",
    "log_sloc",
    "cyclomatic_complexity_norm",
    "cohesion",
    "coupling_afferent_norm",
    "coupling_efferent_norm",
    "coupling_instability",
    "log_dependency_count",
    "log_code_churn",
    "log_commit_count",
    "developer_count_norm",
    "log_file_age_days",
    "bug_density_norm",
]


class PyGConverter:
    """Converts NetworkX Evolution Graphs into PyTorch Geometric Data tensors."""

    @staticmethod
    def convert(G: nx.DiGraph, overall_risk_class: str = "Low", overall_risk_score: float = 0.0) -> Data:
        """
        Transforms a NetworkX DiGraph into a PyTorch Geometric Data object:
        - x: [num_nodes, 13] node feature tensor
        - edge_index: [2, num_edges] directed edge connectivity
        - edge_attr: [num_edges, 4] edge type one-hot & weight
        - y: graph risk classification target (0: Low, 1: Medium, 2: High)
        - risk_score: continuous target tensor
        - node_names: list of file path strings
        """
        node_list = list(G.nodes())
        num_nodes = len(node_list)

        if num_nodes == 0:
            # Fallback for empty graph
            x = torch.zeros((1, len(FEATURE_NAMES)), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)
            y = torch.tensor([0], dtype=torch.long)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, node_names=["empty"])

        # Mapping node identifier (file_path) to integer index
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

        # 1. Build Node Feature Matrix (N x 13)
        features: List[List[float]] = []
        node_risk_labels: List[int] = []

        risk_class_map = {"Low": 0, "Medium": 1, "High": 2}

        for node_id in node_list:
            attrs = G.nodes[node_id]

            loc = attrs.get("loc", 0)
            sloc = attrs.get("sloc", 0)
            cc = attrs.get("cyclomatic_complexity", 1.0)
            cohesion = attrs.get("cohesion", 1.0)
            ca = attrs.get("coupling_afferent", 0)
            ce = attrs.get("coupling_efferent", 0)
            instability = attrs.get("coupling_instability", 0.0)
            deps = attrs.get("dependency_count", 0)
            churn = attrs.get("code_churn", 0)
            commits = attrs.get("commit_count", 1)
            devs = attrs.get("developer_count", 1)
            age = attrs.get("file_age_days", 1.0)
            bug_density = attrs.get("bug_density", 0.0)

            node_feat = [
                round(math.log1p(max(0, loc)) / 8.0, 4),               # 0: log_loc (scaled ~0 to 1)
                round(math.log1p(max(0, sloc)) / 8.0, 4),              # 1: log_sloc
                round(min(1.0, cc / 30.0), 4),                         # 2: cc_norm
                round(max(0.0, min(1.0, cohesion)), 4),                # 3: cohesion
                round(min(1.0, ca / 20.0), 4),                         # 4: ca_norm
                round(min(1.0, ce / 20.0), 4),                         # 5: ce_norm
                round(max(0.0, min(1.0, instability)), 4),             # 6: instability
                round(math.log1p(max(0, deps)) / 5.0, 4),              # 7: log_deps
                round(math.log1p(max(0, churn)) / 10.0, 4),            # 8: log_churn
                round(math.log1p(max(0, commits)) / 6.0, 4),           # 9: log_commits
                round(min(1.0, devs / 10.0), 4),                       # 10: devs_norm
                round(math.log1p(max(0, age)) / 8.0, 4),               # 11: log_age
                round(min(1.0, bug_density / 10.0), 4),                # 12: bug_density_norm
            ]
            features.append(node_feat)

            r_class = attrs.get("risk_class", "Low")
            node_risk_labels.append(risk_class_map.get(r_class, 0))

        x_tensor = torch.tensor(features, dtype=torch.float)

        # 2. Build Edge Index & Edge Attributes
        edges = list(G.edges(data=True))
        if edges:
            sources: List[int] = []
            targets: List[int] = []
            edge_attrs: List[List[float]] = []

            for u, v, attrs in edges:
                if u in node_to_idx and v in node_to_idx:
                    sources.append(node_to_idx[u])
                    targets.append(node_to_idx[v])

                    etype = attrs.get("edge_type", "import")
                    weight = attrs.get("weight", 1.0)

                    # One-hot encoding for edge types: [is_import, is_call, is_co_change, weight]
                    attr_vec = [
                        1.0 if etype == "import" else 0.0,
                        1.0 if etype == "function_call" else 0.0,
                        1.0 if etype == "co_change" else 0.0,
                        float(weight),
                    ]
                    edge_attrs.append(attr_vec)

            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)

        # Graph-level target
        graph_label = risk_class_map.get(overall_risk_class, 0)
        y = torch.tensor([graph_label], dtype=torch.long)
        risk_score_tensor = torch.tensor([overall_risk_score / 100.0], dtype=torch.float)

        pyg_data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            risk_score=risk_score_tensor,
            node_names=node_list,
            node_risk_labels=torch.tensor(node_risk_labels, dtype=torch.long),
        )

        return pyg_data

    @staticmethod
    def save_pyg_data(data: Data, save_path: Path) -> Path:
        """Saves PyG Data to disk in .pt format."""
        save_path = Path(save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)
        logger.info(f"Saved PyG Data tensor to {save_path}")
        return save_path

    @staticmethod
    def load_pyg_data(load_path: Path) -> Data:
        """Loads PyG Data tensor from disk."""
        load_path = Path(load_path).resolve()
        return torch.load(load_path, weights_only=False)
