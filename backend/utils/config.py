import os
from pathlib import Path

# Resolve base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATASET_DIR = BASE_DIR / "dataset"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
REPORTS_DIR = BASE_DIR / "reports"
REPOSITORIES_DIR = BACKEND_DIR / "repositories"
UPLOADS_DIR = BACKEND_DIR / "uploads"
DATABASE_DIR = BACKEND_DIR / "database"

# Ensure all essential directories exist
for path in [
    DATASET_DIR,
    TRAINED_MODELS_DIR,
    REPORTS_DIR,
    REPOSITORIES_DIR,
    UPLOADS_DIR,
    DATABASE_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


class Settings:
    PROJECT_NAME: str = "Maintainability Risk Predictor"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # SQLite Database URL
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{DATABASE_DIR / 'maintainability_risk.db'}"
    )
    
    # Git & Analysis settings
    MAX_REPO_FILES: int = int(os.getenv("MAX_REPO_FILES", "1000"))
    GIT_CLONE_TIMEOUT: int = int(os.getenv("GIT_CLONE_TIMEOUT", "300"))
    DEFAULT_BRANCH: str = "main"
    
    # Graph Feature dimensions
    NUM_NODE_FEATURES: int = 13
    
    # Allowed source code extensions
    SUPPORTED_EXTENSIONS: set = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".go",
    }
    
    # Risk Thresholds
    RISK_THRESHOLD_LOW: float = 35.0
    RISK_THRESHOLD_HIGH: float = 70.0


settings = Settings()
