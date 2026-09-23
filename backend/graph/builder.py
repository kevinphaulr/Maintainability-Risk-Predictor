import networkx as nx
from typing import Dict, List, Tuple, Any
from pathlib import Path

from backend.services.parser_service import ParsedFileInfo
from backend.utils.logger import get_logger

logger = get_logger("graph_builder")


class EvolutionGraphBuilder:
    """
    Constructs a NetworkX Directed Repository Evolution Graph.
    - Each File is a Node carrying all 13+ extracted metrics.
    - Edges represent Imports, Function Calls, Dependencies, and Co-Change Coupling.
    """

    @staticmethod
    def build_graph(
        file_metrics_list: List[Dict[str, Any]],
        parsed_files: Dict[str, ParsedFileInfo],
        co_changes: Dict[Tuple[str, str], int],
    ) -> nx.DiGraph:
        """Constructs the repository evolution graph."""
        G = nx.DiGraph()

        # Index metrics by file_path
        metrics_by_file = {m["file_path"]: m for m in file_metrics_list}

        # 1. Add all File Nodes
        for file_path, metrics in metrics_by_file.items():
            file_name = Path(file_path).name
            ext = Path(file_path).suffix.lower()

            G.add_node(
                file_path,
                label=file_name,
                file_type=ext,
                risk_class=metrics.get("risk_class", "Low"),
                risk_score=metrics.get("risk_score", 0.0),
                loc=metrics.get("loc", 0),
                sloc=metrics.get("sloc", 0),
                cyclomatic_complexity=metrics.get("cyclomatic_complexity", 1.0),
                cohesion=metrics.get("cohesion", 1.0),
                coupling_afferent=metrics.get("coupling_afferent", 0),
                coupling_efferent=metrics.get("coupling_efferent", 0),
                coupling_instability=metrics.get("coupling_instability", 0.0),
                dependency_count=metrics.get("dependency_count", 0),
                code_churn=metrics.get("code_churn", 0),
                commit_count=metrics.get("commit_count", 1),
                developer_count=metrics.get("developer_count", 1),
                file_age_days=metrics.get("file_age_days", 1.0),
                commit_frequency=metrics.get("commit_frequency", 1.0),
                commit_interval_avg=metrics.get("commit_interval_avg", 1.0),
                bug_density=metrics.get("bug_density", 0.0),
                maintainability_index=metrics.get("maintainability_index", 100.0),
            )

        # 2. Add Import and Dependency Edges
        for source_file, info in parsed_files.items():
            if source_file not in G:
                continue

            for target_file in info.internal_imported_files:
                if target_file in G and source_file != target_file:
                    G.add_edge(
                        source_file,
                        target_file,
                        edge_type="import",
                        weight=1.0,
                        label="imports",
                    )

        # 3. Add Function Call Edges (where caller in File A calls function defined in File B)
        # Build global function-to-defining-file mapping
        func_to_file: Dict[str, str] = {}
        for file_path, info in parsed_files.items():
            for func_name in info.function_defs:
                # Avoid generic single character or common names
                if len(func_name) > 2 and func_name not in {"get", "set", "run", "init"}:
                    func_to_file[func_name] = file_path

        for source_file, info in parsed_files.items():
            for call in info.function_calls:
                target_file = func_to_file.get(call)
                if target_file and target_file != source_file and target_file in G:
                    if G.has_edge(source_file, target_file):
                        # Elevate weight if both import and call exist
                        G[source_file][target_file]["weight"] += 0.5
                    else:
                        G.add_edge(
                            source_file,
                            target_file,
                            edge_type="function_call",
                            weight=1.0,
                            label=f"calls {call}",
                        )

        # 4. Add Co-Change Git Evolution Edges
        for (f1, f2), count in co_changes.items():
            if f1 in G and f2 in G and f1 != f2 and count >= 2:
                norm_weight = round(min(5.0, count * 0.2), 2)
                # Co-change is bidirectional coupling in evolution
                if not G.has_edge(f1, f2):
                    G.add_edge(f1, f2, edge_type="co_change", weight=norm_weight, label="co-changed")
                if not G.has_edge(f2, f1):
                    G.add_edge(f2, f1, edge_type="co_change", weight=norm_weight, label="co-changed")

        logger.info(f"Built evolution graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    @staticmethod
    def compute_layout(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        Computes 2D spatial positions using spring layout scaled for React Flow rendering.
        """
        if G.number_of_nodes() == 0:
            return {}

        try:
            # Spring layout with optimal node distance
            k_distance = 2.0 / max(1, (G.number_of_nodes() ** 0.5))
            raw_pos = nx.spring_layout(G, k=k_distance, iterations=50, seed=42)
        except Exception as e:
            logger.warning(f"Spring layout failed, falling back to circular layout: {e}")
            raw_pos = nx.circular_layout(G)

        # Scale coordinates for pleasant React Flow canvas (e.g. 0 to 1200 px)
        scaled_pos: Dict[str, Tuple[float, float]] = {}
        for node_id, (x, y) in raw_pos.items():
            scaled_x = round((x + 1.0) * 450.0 + 50.0, 1)
            scaled_y = round((y + 1.0) * 350.0 + 50.0, 1)
            scaled_pos[node_id] = (scaled_x, scaled_y)

        return scaled_pos

    @staticmethod
    def to_react_flow_format(G: nx.DiGraph, layout_pos: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Serializes NetworkX DiGraph into React Flow compatible nodes and edges."""
        nodes = []
        for node_id, attrs in G.nodes(data=True):
            pos = layout_pos.get(node_id, (100.0, 100.0))
            nodes.append({
                "id": node_id,
                "label": attrs.get("label", node_id),
                "file_type": attrs.get("file_type", ""),
                "risk_class": attrs.get("risk_class", "Low"),
                "risk_score": attrs.get("risk_score", 0.0),
                "metrics": {
                    "loc": attrs.get("loc", 0),
                    "sloc": attrs.get("sloc", 0),
                    "cyclomatic_complexity": attrs.get("cyclomatic_complexity", 1.0),
                    "cohesion": attrs.get("cohesion", 1.0),
                    "coupling_afferent": attrs.get("coupling_afferent", 0),
                    "coupling_efferent": attrs.get("coupling_efferent", 0),
                    "coupling_instability": attrs.get("coupling_instability", 0.0),
                    "dependency_count": attrs.get("dependency_count", 0),
                    "code_churn": attrs.get("code_churn", 0),
                    "commit_count": attrs.get("commit_count", 1),
                    "developer_count": attrs.get("developer_count", 1),
                    "file_age_days": attrs.get("file_age_days", 1.0),
                    "maintainability_index": attrs.get("maintainability_index", 100.0),
                },
                "position": {"x": pos[0], "y": pos[1]},
            })

        edges = []
        edge_counter = 1
        for u, v, attrs in G.edges(data=True):
            edges.append({
                "id": f"edge_{edge_counter}_{u}_{v}",
                "source": u,
                "target": v,
                "type": attrs.get("edge_type", "import"),
                "weight": attrs.get("weight", 1.0),
            })
            edge_counter += 1

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }
