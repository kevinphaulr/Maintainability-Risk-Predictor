import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from backend.utils.config import settings, DATASET_DIR
from backend.utils.logger import get_logger
from backend.utils.file_utils import scan_repository_files
from backend.services.git_service import GitService
from backend.services.parser_service import ParserService, ParsedFileInfo
from backend.services.metrics_service import MetricsService
from backend.graph.builder import EvolutionGraphBuilder
from backend.graph.pyg_converter import PyGConverter
from backend.models.database_models import Repository, AnalysisResult, MetricRecord, Prediction

logger = get_logger("evolution_graph_service")


class EvolutionGraphService:
    """End-to-end orchestrator for repository analysis and evolution graph generation."""

    @staticmethod
    def analyze_repository(repo_id: int, db: Session) -> Dict[str, Any]:
        """
        Executes complete multi-step analysis pipeline:
        Git History -> AST Parsing -> Metrics Extraction -> Evolution Graph -> PyG Data -> SQLite persistence.
        """
        repo_record = db.query(Repository).filter(Repository.id == repo_id).first()
        if not repo_record:
            raise ValueError(f"Repository with ID {repo_id} not found.")

        repo_path = Path(repo_record.local_path).resolve()
        logger.info(f"Initiating analysis for repository '{repo_record.name}' at {repo_path}")
        repo_record.status = "analyzing"
        db.commit()

        try:
            # 1. Scan source files
            source_file_paths = scan_repository_files(repo_path)
            if not source_file_paths:
                raise ValueError("No supported source code files discovered in the repository.")

            # Set of relative file paths for resolving internal imports
            all_rel_paths = {
                f.resolve().relative_to(repo_path).as_posix()
                for f in source_file_paths
            }

            # 2. Mine Git History & Temporal Evolution
            git_history = GitService.analyze_git_history(repo_path)
            repo_record.total_commits = git_history["total_commits"]
            repo_record.total_developers = git_history["total_developers"]
            repo_record.repo_age_days = git_history["repo_age_days"]

            # 3. Parse Source Code AST across all files
            parsed_files: Dict[str, ParsedFileInfo] = {}
            for file_path in source_file_paths:
                parsed_info = ParserService.parse_file(file_path, all_rel_paths, repo_path)
                parsed_files[parsed_info.file_path] = parsed_info

            # 4. Extract Software & Evolution Metrics
            metrics_data = MetricsService.aggregate_repository_metrics(
                parsed_files=parsed_files,
                git_history=git_history,
                repo_name=repo_record.name,
            )
            file_records = metrics_data["files"]
            summary = metrics_data["summary"]

            # 5. Build Repository Evolution Graph (NetworkX)
            G = EvolutionGraphBuilder.build_graph(
                file_metrics_list=file_records,
                parsed_files=parsed_files,
                co_changes=git_history.get("co_changes", {}),
            )

            # 6. Compute 2D Spring Layout for React Flow Canvas
            layout_pos = EvolutionGraphBuilder.compute_layout(G)
            react_flow_graph = EvolutionGraphBuilder.to_react_flow_format(G, layout_pos)

            # 7. Convert Graph to PyTorch Geometric Data Tensor
            pyg_data = PyGConverter.convert(
                G,
                overall_risk_class=summary["risk_class"],
                overall_risk_score=summary["overall_risk_score"],
            )

            # 8. Save Graph Artifacts to Disk
            analysis_slug = f"repo_{repo_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            graph_json_file = DATASET_DIR / f"{analysis_slug}_graph.json"
            pyg_pt_file = DATASET_DIR / f"{analysis_slug}_pyg.pt"

            with open(graph_json_file, "w", encoding="utf-8") as f:
                json.dump(react_flow_graph, f, indent=2)

            PyGConverter.save_pyg_data(pyg_data, pyg_pt_file)

            # 9. Persist Analysis and Metric Records in Database
            analysis_result = AnalysisResult(
                repository_id=repo_record.id,
                total_files=summary["total_files"],
                total_loc=summary["total_loc"],
                avg_complexity=summary["avg_complexity"],
                avg_cohesion=summary["avg_cohesion"],
                avg_coupling=summary["avg_instability"],
                overall_risk_score=summary["overall_risk_score"],
                risk_class=summary["risk_class"],
                maintainability_index=summary["maintainability_index"],
                graph_node_count=react_flow_graph["node_count"],
                graph_edge_count=react_flow_graph["edge_count"],
                graph_json_path=str(graph_json_file),
                pyg_data_path=str(pyg_pt_file),
                summary=json.dumps(summary),
            )
            db.add(analysis_result)
            db.flush()  # populate analysis_result.id

            # Save individual file metrics
            for record in file_records:
                metric_row = MetricRecord(
                    analysis_id=analysis_result.id,
                    file_path=record["file_path"],
                    loc=record["loc"],
                    sloc=record["sloc"],
                    comments=record["comments"],
                    cyclomatic_complexity=record["cyclomatic_complexity"],
                    cohesion=record["cohesion"],
                    coupling_afferent=record["coupling_afferent"],
                    coupling_efferent=record["coupling_efferent"],
                    coupling_instability=record["coupling_instability"],
                    dependency_count=record["dependency_count"],
                    code_churn=record["code_churn"],
                    commit_count=record["commit_count"],
                    developer_count=record["developer_count"],
                    file_age_days=record["file_age_days"],
                    commit_frequency=record["commit_frequency"],
                    commit_interval_avg=record["commit_interval_avg"],
                    bug_density=record["bug_density"],
                    maintainability_index=record["maintainability_index"],
                    risk_score=record["risk_score"],
                    risk_class=record["risk_class"],
                )
                db.add(metric_row)

            repo_record.status = "analyzed"
            db.commit()
            db.refresh(analysis_result)

            logger.info(
                f"Analysis completed successfully for {repo_record.name} (Analysis ID: {analysis_result.id})"
            )
            return {
                "analysis_id": analysis_result.id,
                "repository_id": repo_record.id,
                "summary": summary,
                "graph_data": react_flow_graph,
                "pyg_data_path": str(pyg_pt_file),
            }

        except Exception as e:
            repo_record.status = "failed"
            db.commit()
            logger.error(f"Analysis failed for repository {repo_id}: {e}", exc_info=True)
            raise RuntimeError(f"Analysis pipeline error: {e}")
