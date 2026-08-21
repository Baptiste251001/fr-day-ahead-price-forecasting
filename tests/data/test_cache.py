"""Tests for the parquet caching helper."""

from pathlib import Path

import pandas as pd

from fr_power.data.cache import cache_path, cached_fetch, read_cache, write_cache


def _sample_df() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="Europe/Paris")
    return pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=index)


def test_write_and_read_cache_roundtrip(tmp_path: Path) -> None:
    df = _sample_df()
    path = tmp_path / "nested" / "sample.parquet"

    write_cache(df, path)

    assert path.exists()
    loaded = read_cache(path)
    assert loaded is not None
    # Parquet round-tripping does not preserve DatetimeIndex.freq metadata; the
    # data itself is what matters here.
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_read_cache_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_cache(tmp_path / "does_not_exist.parquet") is None


def test_cache_path_builds_expected_filename(tmp_path: Path) -> None:
    assert cache_path(tmp_path, "foo") == tmp_path / "foo.parquet"


def test_cached_fetch_calls_fetch_fn_once_then_uses_cache(tmp_path: Path) -> None:
    call_count = 0

    def fetch_fn() -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return _sample_df()

    first = cached_fetch(tmp_path, "series", fetch_fn)
    second = cached_fetch(tmp_path, "series", fetch_fn)

    assert call_count == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_cached_fetch_force_refresh_bypasses_cache(tmp_path: Path) -> None:
    call_count = 0

    def fetch_fn() -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return _sample_df()

    cached_fetch(tmp_path, "series", fetch_fn)
    cached_fetch(tmp_path, "series", fetch_fn, force_refresh=True)

    assert call_count == 2
