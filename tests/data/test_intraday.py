"""Tests for the pluggable intraday data source."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from fr_power.config import IntradaySourceConfig
from fr_power.data.intraday import CsvIntradaySource, StubIntradaySource, get_intraday_source


def _sample_intraday_df() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="Europe/Paris")
    return pd.DataFrame({"intraday_price_eur_mwh": [55.0, 61.0, 58.0]}, index=index)


def test_csv_source_loads_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "intraday.parquet"
    _sample_intraday_df().to_parquet(path)

    source = CsvIntradaySource(path)
    df = source.load()

    assert df is not None
    assert list(df.columns) == ["intraday_price_eur_mwh"]
    assert len(df) == 3


def test_csv_source_loads_csv_file(tmp_path: Path) -> None:
    path = tmp_path / "intraday.csv"
    _sample_intraday_df().to_csv(path)

    source = CsvIntradaySource(path)
    df = source.load()

    assert df is not None
    assert len(df) == 3


def test_csv_source_returns_none_and_warns_when_file_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = CsvIntradaySource(tmp_path / "missing.parquet")
    with caplog.at_level(logging.WARNING):
        result = source.load()

    assert result is None
    assert "not available" in caplog.text


def test_csv_source_raises_on_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"wrong_column": [1.0]}).to_parquet(path)

    source = CsvIntradaySource(path)
    with pytest.raises(ValueError, match="missing required columns"):
        source.load()


def test_stub_source_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        result = StubIntradaySource().load()

    assert result is None
    assert "not available" in caplog.text


def test_get_intraday_source_factory_dispatches(tmp_path: Path) -> None:
    csv_source = get_intraday_source(
        IntradaySourceConfig(source="csv", path=str(tmp_path / "x.parquet"))
    )
    assert isinstance(csv_source, CsvIntradaySource)

    stub_source = get_intraday_source(IntradaySourceConfig(source="stub", path=""))
    assert isinstance(stub_source, StubIntradaySource)
