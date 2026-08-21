"""Optional RTE Data Portal enrichment (eCO2mix national real-time data).

Disabled by default (`data.rte_eco2mix.enabled: false` in config) -- the
ENTSO-E pull already covers generation by type and load. This module adds
RTE's own eCO2mix feed as an optional cross-check / enrichment source for
users who register a free application at https://data.rte-france.com/.

Auth is OAuth2 client-credentials: RTE_CLIENT_ID/RTE_CLIENT_SECRET (see
.env.example) are exchanged for a short-lived bearer token, which is then
used to call the eCO2mix endpoint.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from fr_power.config import RteEco2mixConfig, Secrets
from fr_power.data.cache import cached_fetch

logger = logging.getLogger(__name__)

TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
ECO2MIX_URL = "https://digital.iservices.rte-france.com/open_api/eco2mix/v4/national_real_time"


def fetch_access_token(client_id: str, client_secret: str, session: Any = None) -> str:
    """Exchange RTE client credentials for a short-lived OAuth2 bearer token."""
    session = session or requests
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {credentials}"}
    response = session.post(TOKEN_URL, headers=headers, timeout=30)
    response.raise_for_status()
    token: str = response.json()["access_token"]
    return token


def fetch_eco2mix_national_real_time(
    access_token: str, start: pd.Timestamp, end: pd.Timestamp, session: Any = None
) -> pd.DataFrame:
    """Fetch RTE's national real-time eCO2mix data for [start, end]."""
    session = session or requests
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"start_date": start.isoformat(), "end_date": end.isoformat()}
    response = session.get(ECO2MIX_URL, headers=headers, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    records = payload.get("national_real_time") or payload.get("eco2mix_national_tr") or []
    df = pd.DataFrame(records)
    if "date_heure" in df.columns:
        df["date_heure"] = pd.to_datetime(df["date_heure"])
        df = df.set_index("date_heure")
    return df


def fetch_and_cache_eco2mix(
    config: RteEco2mixConfig,
    secrets: Secrets,
    cache_dir: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    force_refresh: bool = False,
    session: Any = None,
) -> pd.DataFrame | None:
    """Fetch (or load from cache) RTE eCO2mix data. Returns None if disabled or unconfigured."""
    if not config.enabled:
        logger.info("RTE eCO2mix enrichment disabled in config; skipping.")
        return None
    client_id = secrets.rte_client_id
    client_secret = secrets.rte_client_secret
    if not client_id or not client_secret:
        logger.warning(
            "RTE eCO2mix enrichment is enabled but RTE_CLIENT_ID/RTE_CLIENT_SECRET are "
            "not set. Skipping. See .env.example for how to register a free RTE Data "
            "Portal application."
        )
        return None

    def _fetch() -> pd.DataFrame:
        token = fetch_access_token(client_id, client_secret, session=session)
        return fetch_eco2mix_national_real_time(token, start, end, session=session)

    return cached_fetch(
        cache_dir, "rte_eco2mix_national_real_time", _fetch, force_refresh=force_refresh
    )
