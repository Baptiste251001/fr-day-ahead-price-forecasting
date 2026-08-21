"""Pluggable intraday (continuous-trading) price data source.

Granular EPEX SPOT intraday indices (ID1/ID3) are typically a paid feed and
are not bundled with this repository or available for free via ENTSO-E at
useful granularity. This module defines a small interface so a local
drop-in file can be plugged in when available.

With no intraday file configured, the pipeline must degrade gracefully to
a day-ahead-price forecasting task rather than fail: `load()` returns
`None` and logs a loud, explicit warning so the limitation is never
silently hidden from downstream consumers of this module.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from fr_power.config import IntradaySourceConfig

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"intraday_price_eur_mwh"}

_DEGRADED_WARNING = (
    "Intraday data is not available (%s). Falling back to a day-ahead-price "
    "forecasting task only -- the intraday-vs-day-ahead spread cannot be "
    "modelled without a granular intraday feed. See README.md for how to "
    "plug in a local EPEX ID3/ID1 drop-in file at `data.intraday.path`."
)


class IntradaySource(ABC):
    """Abstract source of hourly-product intraday prices."""

    @abstractmethod
    def load(self) -> pd.DataFrame | None:
        """Return intraday data indexed by delivery start, or None if unavailable."""


class CsvIntradaySource(IntradaySource):
    """Loads a local CSV or parquet drop-in file of intraday prices.

    The file must have a `intraday_price_eur_mwh` column and a datetime
    index (delivery start of the hourly product). CSV files are parsed with
    the first column as a parsed datetime index; parquet files are read
    as-is.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame | None:
        if not self.path.exists():
            logger.warning(_DEGRADED_WARNING, f"no file found at {self.path}")
            return None

        if self.path.suffix == ".parquet":
            df = pd.read_parquet(self.path)
        else:
            df = pd.read_csv(self.path, index_col=0, parse_dates=True)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Intraday source at {self.path} is missing required columns: {missing}"
            )

        logger.info("Loaded %d rows of intraday data from %s.", len(df), self.path)
        return df


class StubIntradaySource(IntradaySource):
    """Explicit stand-in for a free ENTSO-E intraday source, should one become available.

    ENTSO-E Transparency does not currently publish granular intraday
    continuous-trading indices comparable to EPEX ID1/ID3 for FR. This stub
    exists so the pluggable interface has a clearly-marked slot to fill in
    if/when such a source appears, without changing the calling code.
    """

    def load(self) -> pd.DataFrame | None:
        logger.warning(_DEGRADED_WARNING, "no free ENTSO-E intraday source is implemented")
        return None


def get_intraday_source(config: IntradaySourceConfig) -> IntradaySource:
    """Factory: build the configured intraday source."""
    if config.source == "csv":
        return CsvIntradaySource(config.path)
    if config.source == "stub":
        return StubIntradaySource()
    raise ValueError(f"Unknown intraday source: {config.source!r}")
