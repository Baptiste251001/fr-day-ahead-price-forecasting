"""Parquet-based caching for raw data pulls.

Every external data pull (ENTSO-E, weather, RTE) goes through `cached_fetch`
so that re-running the pipeline never re-hits an API unnecessarily, and so
tests never need live network access.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def cache_path(cache_dir: str | Path, name: str) -> Path:
    """Return the parquet path for a named cached dataset within `cache_dir`."""
    return Path(cache_dir) / f"{name}.parquet"


def read_cache(path: str | Path) -> pd.DataFrame | None:
    """Read a cached parquet file, or return None if it doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return None
    logger.info("Reading cached data from %s", path)
    return pd.read_parquet(path)


def write_cache(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to `path` as parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Cached %d rows to %s", len(df), path)


def cached_fetch(
    cache_dir: str | Path,
    name: str,
    fetch_fn: Callable[[], pd.DataFrame],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return the cached dataset `name` under `cache_dir`, fetching and caching it if absent.

    Set `force_refresh=True` to bypass an existing cache and re-fetch.
    """
    path = cache_path(cache_dir, name)
    if not force_refresh:
        cached = read_cache(path)
        if cached is not None:
            return cached
    df = fetch_fn()
    write_cache(df, path)
    return df
