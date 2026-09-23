import shutil
import zipfile
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.database_models import Repository
from backend.schemas.pydantic_schemas import RepositoryCloneRequest, RepositoryResponse
from backend.services.git_service import GitService
from backend.utils.config import REPOSITORIES_DIR, UPLOADS_DIR
from backend.utils.logger import get_logger

logger = get_logger("repository_router")

router = APIRouter(tags=["Repositories"])


@router.post("/clone", response_model=RepositoryResponse, status_code=status.HTTP_201_CREATED)
def clone_repository(payload: RepositoryCloneRequest, db: Session = Depends(get_db)):
    """Clones a GitHub repository or registers a local directory path."""
    try:
        local_path, repo_name = GitService.clone_repository(
            repo_url=payload.url,
            branch=payload.branch,
            custom_name=payload.name,
        )

        repo = Repository(
            name=repo_name,
            url=payload.url,
            default_branch=payload.branch or "main",
            local_path=str(local_path),
            status="cloned",
        )
        db.add(repo)
        db.commit()
        db.refresh(repo)
        return repo
    except Exception as e:
        logger.error(f"Clone error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to clone repository: {str(e)}",
        )


@router.post("/upload", response_model=RepositoryResponse, status_code=status.HTTP_201_CREATED)
async def upload_repository_zip(
    file: UploadFile = File(...),
    custom_name: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Uploads a ZIP archive containing a repository."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip archive files are supported.",
        )

    repo_name = custom_name or Path(file.filename).stem
    extract_dir = REPOSITORIES_DIR / f"{repo_name}_uploaded"
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_path = UPLOADS_DIR / file.filename
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        zip_path.unlink(missing_ok=True)

        # Prepare as git repo if not already
        GitService.clone_repository(str(extract_dir), custom_name=repo_name)

        repo = Repository(
            name=repo_name,
            url=f"local://{file.filename}",
            default_branch="main",
            local_path=str(extract_dir),
            status="cloned",
        )
        db.add(repo)
        db.commit()
        db.refresh(repo)
        return repo
    except Exception as e:
        logger.error(f"Failed to extract uploaded archive: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting ZIP archive: {str(e)}",
        )


@router.get("/repositories", response_model=List[RepositoryResponse])
def list_repositories(db: Session = Depends(get_db)):
    """Lists all registered repositories."""
    return db.query(Repository).order_by(Repository.id.desc()).all()


@router.get("/repositories/{repo_id}", response_model=RepositoryResponse)
def get_repository(repo_id: int, db: Session = Depends(get_db)):
    """Retrieves details of a specific repository."""
    repo = db.query(Repository).filter(Repository.id == repo_id).first()
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository with ID {repo_id} not found.",
        )
    return repo
