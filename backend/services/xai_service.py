import json
from typing import Dict, List, Any, Tuple
from sqlalchemy.orm import Session

from backend.models.database_models import AnalysisResult, MetricRecord, Prediction
from backend.utils.logger import get_logger

logger = get_logger("xai_service")


class ExplainableAIService:
    """
    Explainable AI (XAI) engine providing factor attribution, risk decomposition,
    and actionable refactoring recommendations.
    """

    @staticmethod
    def generate_explanation_and_recommendations(
        analysis_id: int, db: Session
    ) -> Dict[str, Any]:
        """
        Analyzes file metrics and structural patterns to generate:
        - Risk probabilities (Low, Medium, High)
        - Confidence score
        - Feature attribution factors
        - Concrete, actionable refactoring recommendations
        """
        analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
        if not analysis:
            raise ValueError(f"AnalysisResult with ID {analysis_id} not found.")

        metrics_records = (
            db.query(MetricRecord).filter(MetricRecord.analysis_id == analysis_id).all()
        )

        overall_risk_score = analysis.overall_risk_score
        risk_class = analysis.risk_class

        # Compute probability distribution across Low, Medium, High
        probabilities = ExplainableAIService._calculate_probabilities(overall_risk_score)
        confidence = ExplainableAIService._calculate_confidence(probabilities)

        # Identify Risk Factors (Attribution)
        explanations: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        # 1. Cyclomatic Complexity Hotspots
        high_cc_files = [m for m in metrics_records if m.cyclomatic_complexity > 15.0]
        if high_cc_files:
            file_names = [Path(m.file_path).name for m in high_cc_files[:4]]
            explanations.append({
                "factor": "Excessive Cyclomatic Complexity",
                "impact": "High" if len(high_cc_files) > 3 else "Medium",
                "description": f"{len(high_cc_files)} module(s) contain dense nested control flow structures with CC > 15.",
                "affected_files": [m.file_path for m in high_cc_files[:5]],
            })
            recommendations.append(
                f"Refactor complex control loops and split nested branching in {', '.join(file_names)} using Guard Clauses or Strategy Pattern."
            )

        # 2. Code Churn & Volatility
        high_churn_files = [m for m in metrics_records if m.code_churn > 400]
        if high_churn_files:
            explanations.append({
                "factor": "High Code Churn & Evolution Volatility",
                "impact": "High",
                "description": f"{len(high_churn_files)} file(s) undergo extreme modification volume, increasing regression probabilities.",
                "affected_files": [m.file_path for m in high_churn_files[:5]],
            })
            recommendations.append(
                "Stabilize frequently modified files with automated regression test suites and modular decomposition."
            )

        # 3. High Coupling & Architectural Instability
        high_coupling_files = [
            m for m in metrics_records if m.coupling_efferent > 8 or m.coupling_instability > 0.8
        ]
        if high_coupling_files:
            file_names = [Path(m.file_path).name for m in high_coupling_files[:3]]
            explanations.append({
                "factor": "Tight Architectural Coupling & High Instability",
                "impact": "High",
                "description": f"{len(high_coupling_files)} module(s) depend excessively on multiple external or internal components (Ce > 8).",
                "affected_files": [m.file_path for m in high_coupling_files[:5]],
            })
            recommendations.append(
                f"Invert dependencies in {', '.join(file_names)} using Interfaces or Dependency Injection to decrease efferent coupling."
            )

        # 4. Low Cohesion (God Classes / Disparate Responsibilities)
        low_cohesion_files = [m for m in metrics_records if m.cohesion < 0.4 and m.loc > 150]
        if low_cohesion_files:
            file_names = [Path(m.file_path).name for m in low_cohesion_files[:3]]
            explanations.append({
                "factor": "Low Module Cohesion (LCOM)",
                "impact": "Medium",
                "description": f"{len(low_cohesion_files)} class(es) lack cohesion between methods and member variables (Single Responsibility violation).",
                "affected_files": [m.file_path for m in low_cohesion_files[:5]],
            })
            recommendations.append(
                f"Decompose God classes in {', '.join(file_names)} into cohesive sub-services following the Single Responsibility Principle (SRP)."
            )

        # 5. Developer Fragmentation & Multi-Author Churn
        multi_author_files = [m for m in metrics_records if m.developer_count > 4]
        if multi_author_files:
            explanations.append({
                "factor": "High Contributor Turnover & Ownership Diffusion",
                "impact": "Medium",
                "description": f"{len(multi_author_files)} core file(s) have been modified by over 4 distinct contributors without unified code ownership.",
                "affected_files": [m.file_path for m in multi_author_files[:5]],
            })
            recommendations.append(
                "Establish explicit code ownership via CODEOWNERS and mandate strict peer reviews for shared modules."
            )

        # 6. Bug Density & Historical Defects
        defect_prone_files = [m for m in metrics_records if m.bug_density > 2.0]
        if defect_prone_files:
            explanations.append({
                "factor": "Defect Recurrence & High Bug Density",
                "impact": "High",
                "description": f"{len(defect_prone_files)} module(s) have experienced recurring bug fixes relative to their LOC count.",
                "affected_files": [m.file_path for m in defect_prone_files[:5]],
            })
            recommendations.append(
                "Increase unit test coverage to >= 85% on defect-prone modules to prevent recurring regressions."
            )

        # Baseline recommendations if repo is very healthy
        if not recommendations:
            recommendations = [
                "Maintain current modular architecture and ensure continuous integration linter enforcement.",
                "Sustain documentation and test coverage as the codebase evolves.",
                "Perform periodic graph dependency reviews to prevent cyclical imports.",
            ]

        # Save or update Prediction record in database
        existing_pred = (
            db.query(Prediction).filter(Prediction.analysis_id == analysis_id).first()
        )
        if existing_pred:
            existing_pred.risk_score = overall_risk_score
            existing_pred.risk_class = risk_class
            existing_pred.confidence = confidence
            existing_pred.probability_low = probabilities["low"]
            existing_pred.probability_medium = probabilities["medium"]
            existing_pred.probability_high = probabilities["high"]
            existing_pred.explanations_json = json.dumps(explanations)
            existing_pred.recommendations_json = json.dumps(recommendations)
            db.commit()
            prediction_record = existing_pred
        else:
            prediction_record = Prediction(
                analysis_id=analysis_id,
                risk_score=overall_risk_score,
                risk_class=risk_class,
                confidence=confidence,
                probability_low=probabilities["low"],
                probability_medium=probabilities["medium"],
                probability_high=probabilities["high"],
                explanations_json=json.dumps(explanations),
                recommendations_json=json.dumps(recommendations),
            )
            db.add(prediction_record)
            db.commit()

        return {
            "analysis_id": analysis_id,
            "risk_score": overall_risk_score,
            "risk_class": risk_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "explanations": explanations,
            "recommendations": recommendations,
        }

    @staticmethod
    def _calculate_probabilities(risk_score: float) -> Dict[str, float]:
        """Transforms a 0-100 risk score into normalized softmax-like probabilities."""
        # Risk score 0 -> high low-risk prob
        # Risk score 50 -> high medium-risk prob
        # Risk score 100 -> high high-risk prob
        if risk_score <= 35.0:
            low = 0.70 + (35.0 - risk_score) / 35.0 * 0.25
            medium = (risk_score / 35.0) * 0.20
            high = max(0.01, 1.0 - low - medium)
        elif risk_score <= 70.0:
            norm = (risk_score - 35.0) / 35.0
            medium = 0.65 + (1.0 - abs(norm - 0.5) * 2.0) * 0.20
            low = (1.0 - norm) * 0.20
            high = norm * 0.25
        else:
            norm = (risk_score - 70.0) / 30.0
            high = 0.70 + norm * 0.25
            medium = (1.0 - norm) * 0.22
            low = max(0.01, 1.0 - high - medium)

        total = low + medium + high
        return {
            "low": round(low / total, 3),
            "medium": round(medium / total, 3),
            "high": round(high / total, 3),
        }

    @staticmethod
    def _calculate_confidence(probs: Dict[str, float]) -> float:
        """Computes certainty metric from the probability distribution."""
        max_p = max(probs.values())
        # Confidence scaled between 0.75 and 0.98
        conf = 0.75 + (max_p - 0.33) * 0.35
        return round(min(0.98, max(0.65, conf)), 2)
