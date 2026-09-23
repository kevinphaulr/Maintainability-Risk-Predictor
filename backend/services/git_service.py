import os
import shutil
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import git
from git import Repo, GitCommandError

from backend.utils.config import settings, REPOSITORIES_DIR
from backend.utils.logger import get_logger

logger = get_logger("git_service")

# Regex to detect bug-fixing commits
BUG_FIX_PATTERN = re.compile(
    r"\b(fix|bug|issue|defect|patch|resolve|resolves|fixed|fixing|crash|error|problem)\b",
    re.IGNORECASE,
)


class GitService:
    """Provides Git repository cloning, history extraction, and evolution metrics mining."""

    @staticmethod
    def clone_repository(repo_url: str, branch: Optional[str] = None, custom_name: Optional[str] = None) -> Tuple[Path, str]:
        """
        Clones a remote repository or prepares a local repository.
        Returns: (local_path, repository_name)
        """
        # Determine repository name
        if custom_name:
            repo_name = custom_name
        elif repo_url.startswith(("http://", "https://", "git@")):
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        else:
            repo_name = Path(repo_url).name
            
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target_dir = REPOSITORIES_DIR / f"{repo_name}_{timestamp}"

        # If it's a local folder path
        if os.path.exists(repo_url):
            logger.info(f"Using local path as repository source: {repo_url}")
            src_path = Path(repo_url).resolve()
            
            # Avoid copying destination, caches, or virtualenvs
            ignore_filter = shutil.ignore_patterns(
                "repositories", "uploads", "__pycache__", ".git", "venv", ".venv", "*.db", "dist", "build"
            )
            shutil.copytree(src_path, target_dir, dirs_exist_ok=True, ignore=ignore_filter)
            
            # If not a git repo, initialize one and commit
            if not (target_dir / ".git").exists():
                logger.info(f"Initializing Git repository for local directory: {target_dir}")
                local_repo = Repo.init(target_dir)
                local_repo.git.add(A=True)
                local_repo.index.commit("Initial commit from local files")
            return target_dir, repo_name

        # Remote Git clone
        logger.info(f"Cloning remote repository {repo_url} into {target_dir}")
        try:
            clone_kwargs = {"depth": 500}  # generous history depth for evolution graphs
            if branch:
                clone_kwargs["branch"] = branch
            Repo.clone_from(repo_url, target_dir, **clone_kwargs)
            logger.info(f"Successfully cloned {repo_url} to {target_dir}")
            return target_dir, repo_name
        except GitCommandError as e:
            logger.error(f"Git clone failed: {e}")
            # Attempt shallow fallback without branch constraint
            try:
                Repo.clone_from(repo_url, target_dir, depth=100)
                return target_dir, repo_name
            except Exception as inner_e:
                if target_dir.exists():
                    shutil.rmtree(target_dir, ignore_errors=True)
                raise RuntimeError(f"Failed to clone repository: {inner_e}")

    @staticmethod
    def analyze_git_history(repo_path: Path) -> Dict[str, Any]:
        """
        Mines commit history, developer activity, code churn, file age, and co-changes.
        """
        repo_path = Path(repo_path).resolve()
        if not (repo_path / ".git").exists():
            logger.warning(f"No .git directory found at {repo_path}. Returning baseline history.")
            return GitService._get_default_history(repo_path)

        try:
            repo = Repo(repo_path)
            commits = list(repo.iter_commits(max_count=1000))
        except Exception as e:
            logger.error(f"Error reading commits from {repo_path}: {e}")
            return GitService._get_default_history(repo_path)

        if not commits:
            return GitService._get_default_history(repo_path)

        commits.reverse()  # Chronological order: oldest to newest
        first_commit_time = datetime.fromtimestamp(commits[0].committed_date, tz=timezone.utc)
        latest_commit_time = datetime.fromtimestamp(commits[-1].committed_date, tz=timezone.utc)
        
        repo_age_days = max(1.0, (latest_commit_time - first_commit_time).total_seconds() / 86400.0)
        repo_age_months = max(1.0 / 30.0, repo_age_days / 30.4375)

        total_commits = len(commits)
        overall_commit_frequency = total_commits / repo_age_months

        # Per-file statistics
        file_commits: Dict[str, List[datetime]] = {}
        file_authors: Dict[str, set] = {}
        file_churn: Dict[str, int] = {}
        file_bug_fixes: Dict[str, int] = {}
        co_changes: Dict[Tuple[str, str], int] = {}

        # Author statistics
        author_commits: Dict[str, int] = {}

        # Timeline trends
        commit_trend_dict: Dict[str, Dict[str, int]] = {}

        for commit in commits:
            author_name = commit.author.name or "Unknown"
            author_commits[author_name] = author_commits.get(author_name, 0) + 1

            c_date = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)
            date_str = c_date.strftime("%Y-%m-%d")

            if date_str not in commit_trend_dict:
                commit_trend_dict[date_str] = {"commits": 0, "churn": 0}
            commit_trend_dict[date_str]["commits"] += 1

            is_bug_fix = bool(BUG_FIX_PATTERN.search(commit.message))

            # Inspect modified files
            touched_files: List[str] = []
            try:
                stats = commit.stats.files
                for file_rel_path, stat in stats.items():
                    norm_path = Path(file_rel_path).as_posix()
                    touched_files.append(norm_path)

                    churn = stat.get("insertions", 0) + stat.get("deletions", 0)
                    commit_trend_dict[date_str]["churn"] += churn

                    # File records
                    if norm_path not in file_commits:
                        file_commits[norm_path] = []
                        file_authors[norm_path] = set()
                        file_churn[norm_path] = 0
                        file_bug_fixes[norm_path] = 0

                    file_commits[norm_path].append(c_date)
                    file_authors[norm_path].add(author_name)
                    file_churn[norm_path] += churn
                    if is_bug_fix:
                        file_bug_fixes[norm_path] += 1
            except Exception:
                pass

            # Track co-change coupling between files modified in the same commit
            if len(touched_files) > 1 and len(touched_files) <= 50:  # ignore massive bulk refactors
                sorted_files = sorted(touched_files)
                for i in range(len(sorted_files)):
                    for j in range(i + 1, len(sorted_files)):
                        pair = (sorted_files[i], sorted_files[j])
                        co_changes[pair] = co_changes.get(pair, 0) + 1

        # File metrics calculation
        file_metrics_map: Dict[str, Dict[str, Any]] = {}
        for file_path, commit_dates in file_commits.items():
            first_seen = commit_dates[0]
            file_age_days = max(1.0, (latest_commit_time - first_seen).total_seconds() / 86400.0)
            file_age_months = max(1.0 / 30.0, file_age_days / 30.4375)

            commit_count = len(commit_dates)
            commit_freq = commit_count / file_age_months

            # Calculate average interval between consecutive commits (in days)
            if len(commit_dates) > 1:
                intervals = [
                    (commit_dates[i] - commit_dates[i - 1]).total_seconds() / 86400.0
                    for i in range(1, len(commit_dates))
                ]
                avg_interval = sum(intervals) / len(intervals)
            else:
                avg_interval = file_age_days

            file_metrics_map[file_path] = {
                "commit_count": commit_count,
                "developer_count": len(file_authors.get(file_path, set())),
                "code_churn": file_churn.get(file_path, 0),
                "file_age_days": round(file_age_days, 1),
                "commit_frequency": round(commit_freq, 2),
                "commit_interval_avg": round(avg_interval, 2),
                "bug_fix_count": file_bug_fixes.get(file_path, 0),
            }

        # Developer activity list
        developer_activity = [
            {
                "author": author,
                "commits": count,
                "impact_score": round((count / total_commits) * 100, 1),
            }
            for author, count in sorted(
                author_commits.items(), key=lambda x: x[1], reverse=True
            )
        ]

        # Commit trends list
        commit_trends = [
            {
                "date": date_str,
                "commit_count": data["commits"],
                "churn": data["churn"],
            }
            for date_str, data in sorted(commit_trend_dict.items())
        ]

        return {
            "total_commits": total_commits,
            "total_developers": len(author_commits),
            "repo_age_days": round(repo_age_days, 1),
            "overall_commit_frequency": round(overall_commit_frequency, 2),
            "file_metrics": file_metrics_map,
            "co_changes": co_changes,
            "developer_activity": developer_activity,
            "commit_trends": commit_trends[-30:],  # Last 30 active days
        }

    @staticmethod
    def _get_default_history(repo_path: Path) -> Dict[str, Any]:
        """Provides default values when no git history is present."""
        return {
            "total_commits": 1,
            "total_developers": 1,
            "repo_age_days": 1.0,
            "overall_commit_frequency": 1.0,
            "file_metrics": {},
            "co_changes": {},
            "developer_activity": [{"author": "Author", "commits": 1, "impact_score": 100.0}],
            "commit_trends": [{"date": datetime.utcnow().strftime("%Y-%m-%d"), "commit_count": 1, "churn": 10}],
        }
