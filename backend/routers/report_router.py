from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.services.report_service import ReportService
from backend.utils.logger import get_logger

logger = get_logger("report_router")

router = APIRouter(tags=["Reports"])


@router.get("/reports/{analysis_id}/json")
def export_json_report(analysis_id: int, db: Session = Depends(get_db)):
    """Exports full analysis and risk prediction report as a structured JSON object."""
    try:
        report = ReportService.generate_json_report(analysis_id, db)
        return report
    except Exception as e:
        logger.error(f"JSON export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to generate JSON report: {str(e)}",
        )


@router.get("/reports/{analysis_id}/csv")
def export_csv_report(analysis_id: int, db: Session = Depends(get_db)):
    """Exports all file-level software metrics in CSV tabular format."""
    try:
        csv_content = ReportService.generate_csv_report(analysis_id, db)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=maintainability_metrics_{analysis_id}.csv"
            },
        )
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to generate CSV report: {str(e)}",
        )


@router.get("/reports/{analysis_id}/pdf")
def export_pdf_report(analysis_id: int, db: Session = Depends(get_db)):
    """Generates and downloads an academic-grade PDF maintainability evaluation report."""
    try:
        pdf_path = ReportService.generate_pdf_report(analysis_id, db)
        return FileResponse(
            path=str(pdf_path),
            filename=f"Maintainability_Risk_Report_{analysis_id}.pdf",
            media_type="application/pdf",
        )
    except Exception as e:
        logger.error(f"PDF export failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF report: {str(e)}",
        )
