import math
from typing import Dict, List, Set, Any
from pathlib import Path

from backend.services.parser_service import ParsedFileInfo
from backend.utils.logger import get_logger

logger = get_logger("metrics_service")


class MetricsService:
    """Computes, normalizes, and aggregates 13+ software metrics per file and repository."""

    @staticmethod
    def calculate_coupling(parsed_files: Dict[str, ParsedFileInfo]) -> Dict[str, Dict[str, float]]:
        """
        Calculates Afferent Coupling (Ca), Efferent Coupling (Ce), and Instability (I)
        across all repository files.
        Ca = number of other files importing this file
        Ce = number of other files imported by this file
        Instability = Ce / (Ca + Ce)
        """
        afferent_map: Dict[str, int] = {f: 0 for f in parsed_files}
        efferent_map: Dict[str, int] = {f: 0 for f in parsed_files}

        for file_path, info in parsed_files.items():
            efferent_count = len(info.internal_imported_files)
            efferent_map[file_path] = efferent_count

            for imported_target in info.internal_imported_files:
                if imported_target in afferent_map:
                    afferent_map[imported_target] += 1

        coupling_results: Dict[str, Dict[str, float]] = {}
        for file_path in parsed_files:
            ca = afferent_map[file_path]
            ce = efferent_map[file_path]
            total = ca + ce
            instability = (ce / total) if total > 0 else 0.0
            coupling_results[file_path] = {
                "ca": ca,
                "ce": ce,
                "instability": round(instability, 3),
            }

        return coupling_results

    @staticmethod
    def calculate_maintainability_index(loc: int, cyclomatic_complexity: float, comments: int) -> float:
        """
        Computes the standard Maintainability Index (MI) on a 0 - 100 scale.
        MI = max(0, min(100, (171 - 5.2 * ln(Halstead/LOC) - 0.23 * CC - 16.2 * ln(LOC)) * 100 / 171))
        Incorporates comment percentage boost.
        """
        safe_loc = max(1, loc)
        safe_cc = max(1.0, cyclomatic_complexity)
        
        # Estimate Halstead volume approximation from LOC: V ~= LOC * 30
        volume = max(1.0, safe_loc * 30.0)
        
        raw_mi = 171.0 - 5.2 * math.log(volume) - 0.23 * safe_cc - 16.2 * math.log(safe_loc)
        
        # Comment weight bonus (up to 15 points)
        comment_ratio = comments / safe_loc if safe_loc > 0 else 0.0
        bonus = min(15.0, comment_ratio * 40.0)
        
        normalized_mi = max(0.0, min(100.0, (raw_mi * 100.0 / 171.0) + bonus))
        return round(normalized_mi, 2)

    @staticmethod
    def calculate_file_risk_score(
        loc: int,
        complexity: float,
        churn: int,
        coupling_instability: float,
        cohesion: float,
        developer_count: int,
        bug_density: float,
        maintainability_index: float,
    ) -> float:
        """
        Synthesizes a multi-dimensional risk score from 0.0 (safest) to 100.0 (highest risk).
        """
        # 1. Complexity component (0 - 25)
        # CC: 1-10 is normal, 11-20 moderate risk, >20 high risk
        comp_score = min(25.0, (complexity / 20.0) * 25.0)

        # 2. Size & Churn component (0 - 25)
        # LOC: 500+ is large. Churn: 500+ is volatile.
        size_score = min(15.0, (loc / 600.0) * 15.0)
        churn_score = min(10.0, (churn / 500.0) * 10.0)

        # 3. Coupling & Cohesion component (0 - 20)
        # Low cohesion (<0.5) and high instability (>0.7) increase risk
        cohesion_penalty = (1.0 - cohesion) * 10.0
        instability_penalty = coupling_instability * 10.0

        # 4. Git Evolution & Defect component (0 - 30)
        # Many authors (>5) indicates diffusion of ownership
        dev_score = min(10.0, (developer_count / 6.0) * 10.0)
        # High bug density
        bug_score = min(10.0, (bug_density * 2.0) * 10.0)
        # Inverted maintainability index
        mi_penalty = ((100.0 - maintainability_index) / 100.0) * 10.0

        total_risk = comp_score + size_score + churn_score + cohesion_penalty + instability_penalty + dev_score + bug_score + mi_penalty
        return round(max(0.0, min(100.0, total_risk)), 2)

    @staticmethod
    def classify_risk(risk_score: float) -> str:
        """Classifies risk score into standard categories: Low, Medium, High."""
        if risk_score >= 70.0:
            return "High"
        elif risk_score >= 35.0:
            return "Medium"
        return "Low"

    @staticmethod
    def aggregate_repository_metrics(
        parsed_files: Dict[str, ParsedFileInfo],
        git_history: Dict[str, Any],
        repo_name: str,
    ) -> Dict[str, Any]:
        """
        Integrates AST metrics, coupling, and Git history into complete file records
        and repository-wide summary.
        """
        coupling_map = MetricsService.calculate_coupling(parsed_files)
        git_file_metrics = git_history.get("file_metrics", {})
        repo_age = git_history.get("repo_age_days", 1.0)

        file_metric_records: List[Dict[str, Any]] = []
        total_loc = 0
        total_complexity = 0.0
        total_cohesion = 0.0
        total_instability = 0.0
        total_mi = 0.0
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0

        for rel_path, info in parsed_files.items():
            coupling = coupling_map.get(rel_path, {"ca": 0, "ce": 0, "instability": 0.0})
            git_stats = git_file_metrics.get(
                rel_path,
                {
                    "commit_count": 1,
                    "developer_count": 1,
                    "code_churn": info.total_lines,
                    "file_age_days": repo_age,
                    "commit_frequency": 1.0,
                    "commit_interval_avg": repo_age,
                    "bug_fix_count": 0,
                },
            )

            # Bug density: bug-fixing commits per 100 LOC
            safe_loc = max(1, info.total_lines)
            bug_density = round((git_stats.get("bug_fix_count", 0) / safe_loc) * 100.0, 3)

            mi = MetricsService.calculate_maintainability_index(
                loc=info.total_lines,
                cyclomatic_complexity=info.cyclomatic_complexity,
                comments=info.comment_lines,
            )

            risk_score = MetricsService.calculate_file_risk_score(
                loc=info.total_lines,
                complexity=info.cyclomatic_complexity,
                churn=git_stats.get("code_churn", 0),
                coupling_instability=coupling["instability"],
                cohesion=info.cohesion,
                developer_count=git_stats.get("developer_count", 1),
                bug_density=bug_density,
                maintainability_index=mi,
            )
            risk_class = MetricsService.classify_risk(risk_score)

            if risk_class == "High":
                high_risk_count += 1
            elif risk_class == "Medium":
                medium_risk_count += 1
            else:
                low_risk_count += 1

            record = {
                "file_path": rel_path,
                "loc": info.total_lines,
                "sloc": info.sloc,
                "comments": info.comment_lines,
                "cyclomatic_complexity": info.cyclomatic_complexity,
                "cohesion": info.cohesion,
                "coupling_afferent": coupling["ca"],
                "coupling_efferent": coupling["ce"],
                "coupling_instability": coupling["instability"],
                "dependency_count": info.dependency_count,
                "code_churn": git_stats.get("code_churn", 0),
                "commit_count": git_stats.get("commit_count", 1),
                "developer_count": git_stats.get("developer_count", 1),
                "file_age_days": git_stats.get("file_age_days", repo_age),
                "commit_frequency": git_stats.get("commit_frequency", 1.0),
                "commit_interval_avg": git_stats.get("commit_interval_avg", repo_age),
                "bug_density": bug_density,
                "maintainability_index": mi,
                "risk_score": risk_score,
                "risk_class": risk_class,
            }
            file_metric_records.append(record)

            total_loc += info.total_lines
            total_complexity += info.cyclomatic_complexity
            total_cohesion += info.cohesion
            total_instability += coupling["instability"]
            total_mi += mi

        n = max(1, len(file_metric_records))
        avg_complexity = round(total_complexity / n, 2)
        avg_cohesion = round(total_cohesion / n, 2)
        avg_instability = round(total_instability / n, 2)
        avg_mi = round(total_mi / n, 2)

        # Repository-level risk score
        overall_risk = round(
            sum(r["risk_score"] for r in file_metric_records) / n, 2
        )
        repo_risk_class = MetricsService.classify_risk(overall_risk)

        # Top complex and churned files
        sorted_by_complexity = sorted(file_metric_records, key=lambda x: x["cyclomatic_complexity"], reverse=True)[:5]
        sorted_by_churn = sorted(file_metric_records, key=lambda x: x["code_churn"], reverse=True)[:5]

        return {
            "files": file_metric_records,
            "summary": {
                "total_files": len(file_metric_records),
                "total_loc": total_loc,
                "avg_complexity": avg_complexity,
                "avg_cohesion": avg_cohesion,
                "avg_instability": avg_instability,
                "maintainability_index": avg_mi,
                "overall_risk_score": overall_risk,
                "risk_class": repo_risk_class,
                "high_risk_files_count": high_risk_count,
                "medium_risk_files_count": medium_risk_count,
                "low_risk_files_count": low_risk_count,
                "top_complex_files": sorted_by_complexity,
                "top_churned_files": sorted_by_churn,
            },
        }
