import io
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from backend.models.database_models import (
    Repository,
    AnalysisResult,
    MetricRecord,
    Prediction,
)
from backend.utils.config import REPORTS_DIR
from backend.utils.logger import get_logger

logger = get_logger("report_service")


class ReportService:
    """Exports comprehensive analysis and risk reports in JSON, CSV, and PDF formats."""

    @staticmethod
    def generate_json_report(analysis_id: int, db: Session) -> Dict[str, Any]:
        """Generates a complete JSON data payload of the analysis, metrics, and XAI predictions."""
        analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
        if not analysis:
            raise ValueError(f"AnalysisResult with ID {analysis_id} not found.")

        repo = db.query(Repository).filter(Repository.id == analysis.repository_id).first()
        metrics = db.query(MetricRecord).filter(MetricRecord.analysis_id == analysis_id).all()
        prediction = db.query(Prediction).filter(Prediction.analysis_id == analysis_id).first()

        report_payload = {
            "title": "Software Maintainability Risk Analysis Report",
            "generated_at": datetime.utcnow().isoformat(),
            "repository": {
                "id": repo.id if repo else None,
                "name": repo.name if repo else "Unknown",
                "url": repo.url if repo else None,
                "default_branch": repo.default_branch if repo else "main",
                "total_commits": repo.total_commits if repo else 0,
                "total_developers": repo.total_developers if repo else 0,
                "repo_age_days": repo.repo_age_days if repo else 0.0,
            },
            "analysis": {
                "id": analysis.id,
                "analyzed_at": analysis.analyzed_at.isoformat(),
                "total_files": analysis.total_files,
                "total_loc": analysis.total_loc,
                "avg_complexity": analysis.avg_complexity,
                "avg_cohesion": analysis.avg_cohesion,
                "avg_coupling": analysis.avg_coupling,
                "overall_risk_score": analysis.overall_risk_score,
                "risk_class": analysis.risk_class,
                "maintainability_index": analysis.maintainability_index,
                "graph_node_count": analysis.graph_node_count,
                "graph_edge_count": analysis.graph_edge_count,
            },
            "prediction": {
                "risk_score": prediction.risk_score if prediction else analysis.overall_risk_score,
                "risk_class": prediction.risk_class if prediction else analysis.risk_class,
                "confidence": prediction.confidence if prediction else 0.85,
                "probabilities": {
                    "low": prediction.probability_low if prediction else 0.0,
                    "medium": prediction.probability_medium if prediction else 0.0,
                    "high": prediction.probability_high if prediction else 0.0,
                },
                "explanations": json.loads(prediction.explanations_json) if prediction else [],
                "recommendations": json.loads(prediction.recommendations_json) if prediction else [],
            },
            "metrics": [
                {
                    "file_path": m.file_path,
                    "loc": m.loc,
                    "sloc": m.sloc,
                    "comments": m.comments,
                    "cyclomatic_complexity": m.cyclomatic_complexity,
                    "cohesion": m.cohesion,
                    "coupling_afferent": m.coupling_afferent,
                    "coupling_efferent": m.coupling_efferent,
                    "coupling_instability": m.coupling_instability,
                    "dependency_count": m.dependency_count,
                    "code_churn": m.code_churn,
                    "commit_count": m.commit_count,
                    "developer_count": m.developer_count,
                    "file_age_days": m.file_age_days,
                    "commit_frequency": m.commit_frequency,
                    "commit_interval_avg": m.commit_interval_avg,
                    "bug_density": m.bug_density,
                    "maintainability_index": m.maintainability_index,
                    "risk_score": m.risk_score,
                    "risk_class": m.risk_class,
                }
                for m in metrics
            ],
        }
        return report_payload

    @staticmethod
    def generate_csv_report(analysis_id: int, db: Session) -> str:
        """Generates a CSV string containing all software metrics for every analyzed file."""
        metrics = db.query(MetricRecord).filter(MetricRecord.analysis_id == analysis_id).all()
        if not metrics:
            raise ValueError(f"No metrics records found for analysis {analysis_id}.")

        output = io.StringIO()
        fieldnames = [
            "file_path",
            "loc",
            "sloc",
            "comments",
            "cyclomatic_complexity",
            "cohesion",
            "coupling_afferent",
            "coupling_efferent",
            "coupling_instability",
            "dependency_count",
            "code_churn",
            "commit_count",
            "developer_count",
            "file_age_days",
            "commit_frequency",
            "commit_interval_avg",
            "bug_density",
            "maintainability_index",
            "risk_score",
            "risk_class",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for m in metrics:
            writer.writerow({
                "file_path": m.file_path,
                "loc": m.loc,
                "sloc": m.sloc,
                "comments": m.comments,
                "cyclomatic_complexity": m.cyclomatic_complexity,
                "cohesion": m.cohesion,
                "coupling_afferent": m.coupling_afferent,
                "coupling_efferent": m.coupling_efferent,
                "coupling_instability": m.coupling_instability,
                "dependency_count": m.dependency_count,
                "code_churn": m.code_churn,
                "commit_count": m.commit_count,
                "developer_count": m.developer_count,
                "file_age_days": m.file_age_days,
                "commit_frequency": m.commit_frequency,
                "commit_interval_avg": m.commit_interval_avg,
                "bug_density": m.bug_density,
                "maintainability_index": m.maintainability_index,
                "risk_score": m.risk_score,
                "risk_class": m.risk_class,
            })

        return output.getvalue()

    @staticmethod
    def generate_pdf_report(analysis_id: int, db: Session) -> Path:
        """Generates an academic-quality PDF document using ReportLab."""
        analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
        if not analysis:
            raise ValueError(f"AnalysisResult with ID {analysis_id} not found.")

        repo = db.query(Repository).filter(Repository.id == analysis.repository_id).first()
        prediction = db.query(Prediction).filter(Prediction.analysis_id == analysis_id).first()
        metrics = db.query(MetricRecord).filter(MetricRecord.analysis_id == analysis_id).all()

        pdf_path = REPORTS_DIR / f"Maintainability_Risk_Report_Analysis_{analysis_id}.pdf"
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=15,
        )
        heading2_style = ParagraphStyle(
            "DocH2",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=14,
            spaceAfter=8,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
        )
        bold_body = ParagraphStyle(
            "DocBoldBody",
            parent=body_style,
            fontName="Helvetica-Bold",
        )

        story = []

        # Header Title
        story.append(Paragraph("Software Maintainability Risk Analysis Report", title_style))
        story.append(Paragraph(
            f"Repository: <b>{repo.name if repo else 'N/A'}</b> | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            subtitle_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=15))

        # Executive Summary Table
        risk_color = "#16a34a" if analysis.risk_class == "Low" else ("#d97706" if analysis.risk_class == "Medium" else "#dc2626")
        risk_badge = f'<font color="{risk_color}"><b>{analysis.risk_class.upper()} ({analysis.overall_risk_score}/100)</b></font>'

        summary_data = [
            [Paragraph("<b>Metric</b>", bold_body), Paragraph("<b>Value</b>", bold_body), Paragraph("<b>Metric</b>", bold_body), Paragraph("<b>Value</b>", bold_body)],
            [Paragraph("Overall Risk Level", body_style), Paragraph(risk_badge, body_style), Paragraph("Maintainability Index", body_style), Paragraph(f"<b>{analysis.maintainability_index}/100</b>", body_style)],
            [Paragraph("Analyzed Files", body_style), Paragraph(str(analysis.total_files), body_style), Paragraph("Total Lines of Code", body_style), Paragraph(f"{analysis.total_loc:,}", body_style)],
            [Paragraph("Avg Cyclomatic Complexity", body_style), Paragraph(str(analysis.avg_complexity), body_style), Paragraph("Avg Module Cohesion", body_style), Paragraph(str(analysis.avg_cohesion), body_style)],
            [Paragraph("Graph Nodes (Files)", body_style), Paragraph(str(analysis.graph_node_count), body_style), Paragraph("Graph Edges (Dependencies)", body_style), Paragraph(str(analysis.graph_edge_count), body_style)],
        ]

        summary_table = Table(summary_data, colWidths=[140, 120, 140, 120])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 15))

        # Explainable AI & Risk Attribution Section
        story.append(Paragraph("Explainable AI: Key Risk Drivers", heading2_style))
        explanations = json.loads(prediction.explanations_json) if prediction else []
        if explanations:
            for exp in explanations:
                impact_color = "#dc2626" if exp.get("impact") == "High" else "#d97706"
                factor_title = f'<font color="{impact_color}"><b>[{exp.get("impact", "Medium")} Impact]</b></font> <b>{exp.get("factor", "")}</b>'
                story.append(Paragraph(factor_title, body_style))
                story.append(Paragraph(f"&nbsp;&nbsp;{exp.get('description', '')}", body_style))
                story.append(Spacer(1, 4))
        else:
            story.append(Paragraph("No critical risk hotspots detected. Codebase conforms to standard maintainability thresholds.", body_style))

        story.append(Spacer(1, 10))

        # Refactoring Recommendations
        story.append(Paragraph("Actionable Refactoring Recommendations", heading2_style))
        recommendations = json.loads(prediction.recommendations_json) if prediction else []
        for idx, rec in enumerate(recommendations, start=1):
            story.append(Paragraph(f"<b>{idx}.</b> {rec}", body_style))
            story.append(Spacer(1, 3))

        story.append(Spacer(1, 15))

        # Top High-Risk Files Table
        story.append(Paragraph("Top Critical Modules by Risk Score", heading2_style))
        sorted_files = sorted(metrics, key=lambda x: x.risk_score, reverse=True)[:8]

        file_table_data = [
            [
                Paragraph("<b>File Path</b>", bold_body),
                Paragraph("<b>LOC</b>", bold_body),
                Paragraph("<b>Complexity</b>", bold_body),
                Paragraph("<b>Churn</b>", bold_body),
                Paragraph("<b>Coupling (Ce)</b>", bold_body),
                Paragraph("<b>Risk Score</b>", bold_body),
            ]
        ]
        for f in sorted_files:
            file_table_data.append([
                Paragraph(f.file_path, body_style),
                Paragraph(str(f.loc), body_style),
                Paragraph(str(round(f.cyclomatic_complexity, 1)), body_style),
                Paragraph(str(f.code_churn), body_style),
                Paragraph(str(f.coupling_efferent), body_style),
                Paragraph(f"<b>{f.risk_score}</b> ({f.risk_class})", body_style),
            ])

        file_table = Table(file_table_data, colWidths=[200, 50, 65, 55, 75, 75])
        file_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(file_table)

        doc.build(story)
        logger.info(f"Generated PDF report at {pdf_path}")
        return pdf_path
