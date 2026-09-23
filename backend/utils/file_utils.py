import os
from pathlib import Path
from typing import List, Generator
from backend.utils.config import settings

def is_source_file(file_path: Path) -> bool:
    """Checks whether a given file path is a supported source code file."""
    return file_path.suffix.lower() in settings.SUPPORTED_EXTENSIONS

def scan_repository_files(repo_path: Path) -> List[Path]:
    """
    Recursively scans the repository directory for source code files,
    ignoring .git, node_modules, build directories, and virtual environments.
    """
    ignored_patterns = {
        ".git",
        "node_modules",
        "venv",
        ".venv",
        "env",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
        ".pytest_cache",
        "site-packages",
    }
    
    source_files: List[Path] = []
    repo_path = Path(repo_path).resolve()
    
    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to avoid descending into ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_patterns and not d.startswith(".")]
        
        for file in files:
            full_path = Path(root) / file
            if is_source_file(full_path):
                source_files.append(full_path)
                if len(source_files) >= settings.MAX_REPO_FILES:
                    return source_files
                    
    return source_files

def get_relative_repo_path(file_path: Path, repo_root: Path) -> str:
    """Computes a normalized forward-slash relative path within the repo."""
    try:
        rel = file_path.resolve().relative_to(repo_root.resolve())
        return rel.as_posix()
    except ValueError:
        return file_path.name
