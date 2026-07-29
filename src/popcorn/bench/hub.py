"""Fetch and publish pinned benchmark reports on the Hugging Face Hub."""

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from popcorn.bench.store import BUNDLED_REPORTS, FETCHED, PUBLISHED

REPO = "tilde-research/popcorn-reports"
REVISION = "REVISION"


def repository() -> str:
    return os.getenv("POPCORN_REPORTS_REPO", REPO)


def pinned(directory: Path = BUNDLED_REPORTS) -> str | None:
    """The revision this checkout is pinned to, or None to track the default branch."""
    path = directory / REVISION
    if not path.exists():
        return None
    revision = path.read_text().strip()
    return revision or None


def pull(revision: str | None = None, repo: str | None = None, directory: Path = BUNDLED_REPORTS) -> tuple[int, str]:
    """Copy the published Parquet into `directory`, returning how many files and which revision.

    The pin wins unless a revision is named explicitly, so a build is reproducible by default.
    """
    repo = repo or repository()
    revision = revision or pinned(directory)
    snapshot = Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, allow_patterns=[f"*{PUBLISHED}"]))
    directory.mkdir(parents=True, exist_ok=True)
    sources = sorted(snapshot.glob(f"*{PUBLISHED}"))
    if not sources:
        raise SystemExit(f"{repo} at {revision or 'the default branch'} holds no {PUBLISHED} files")
    for source in sources:
        shutil.copyfile(source, directory / source.name)
    published = {source.name for source in sources}
    for stale in directory.glob(f"*{PUBLISHED}"):
        if stale.name not in published:
            stale.unlink()
    resolved = revision or snapshot.name
    (directory / FETCHED).write_text(f"{resolved}\n")
    return len(sources), resolved


def publish(repo: str | None = None, directory: Path = BUNDLED_REPORTS, message: str = "Update reports") -> str:
    """Upload the published Parquet, write the resulting commit into `REVISION`, and return it."""
    repo = repo or repository()
    if not sorted(directory.glob(f"*{PUBLISHED}")):
        raise SystemExit(f"nothing to publish: {directory} holds no {PUBLISHED} files")
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, private=False)
    commit = api.upload_folder(
        folder_path=str(directory),
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[f"*{PUBLISHED}"],
        commit_message=message,
    )
    revision = commit.oid
    (directory / REVISION).write_text(f"{revision}\n")
    (directory / FETCHED).write_text(f"{revision}\n")
    return revision
