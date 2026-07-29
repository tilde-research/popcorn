"""Fetch the pinned report cache before setuptools builds the package."""

import os
import shutil
from pathlib import Path

from setuptools import setup

REPO = "tilde-research/popcorn-reports"
REPORTS = Path(__file__).parent / "src" / "popcorn" / "reports"
FETCHED = REPORTS / ".revision"


def _skip() -> bool:
    return os.getenv("POPCORN_SKIP_REPORTS", "").lower() in {"1", "true", "yes", "on"}


def _fetch() -> None:
    if _skip():
        print("popcorn: POPCORN_SKIP_REPORTS is set, building without the report cache")
        return
    pin = REPORTS / "REVISION"
    revision = pin.read_text().strip() if pin.exists() else ""
    fetched = FETCHED.read_text().strip() if FETCHED.exists() else ""
    # The marker ships with source distributions. A checkout refreshes only when its tracked
    # pin changes, so ignored Parquet cannot outlive REVISION across updates.
    if any(REPORTS.glob("*.parquet")) and revision and fetched == revision:
        return
    repo = os.getenv("POPCORN_REPORTS_REPO", REPO)
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision or None,
                allow_patterns=["*.parquet"],
            )
        )
    except Exception as error:
        raise SystemExit(
            f"popcorn: could not fetch the report cache from {repo} at {revision or 'the default branch'}:\n"
            f"  {type(error).__name__}: {error}\n"
            "Reports are what dispatch selects implementations with. Retry with network access, or set\n"
            "POPCORN_SKIP_REPORTS=1 to install without them and fetch later with `python -m popcorn.bench pull`."
        ) from error
    REPORTS.mkdir(parents=True, exist_ok=True)
    sources = sorted(snapshot.glob("*.parquet"))
    if not sources:
        raise SystemExit(f"popcorn: {repo} at {revision or 'the default branch'} holds no Parquet reports")
    for source in sources:
        shutil.copyfile(source, REPORTS / source.name)
    published = {source.name for source in sources}
    for stale in REPORTS.glob("*.parquet"):
        if stale.name not in published:
            stale.unlink()
    FETCHED.write_text(f"{revision or snapshot.name}\n")
    print(f"popcorn: fetched {len(sources)} report files at {revision or 'the default branch'}")


_fetch()
setup()
