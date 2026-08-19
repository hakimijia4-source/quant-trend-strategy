from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd

from .config import StrategyConfig
from .http import get_bytes, get_json, with_query


MARKET_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "source",
]


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _save_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def collect_market(
    config: StrategyConfig,
    *,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    provider: str | None = None,
) -> list[Path]:
    market = config.section("market")
    selected = list(symbols or [config.target_symbol, *config.context_symbols])
    selected = list(dict.fromkeys(x.upper() for x in selected))
    provider_name = (provider or market.get("provider", "alpaca")).lower()
    start_date = start or str(market.get("start"))
    end_date = end or str(market.get("end"))
    outputs: list[Path] = []
    for symbol in selected:
        if provider_name == "alpaca":
            frame = collect_alpaca_bars(config, symbol, start_date, end_date)
        elif provider_name in {"massive", "polygon"}:
            frame = collect_massive_bars(config, symbol, start_date, end_date)
        else:
            raise ValueError(f"Unsupported live data provider: {provider_name}")
        path = config.data_dir / "raw" / "market" / f"{symbol}.csv"
        outputs.append(_save_frame(frame, path))
    return outputs


def collect_alpaca_bars(
    config: StrategyConfig, symbol: str, start: str, end: str
) -> pd.DataFrame:
    market = config.section("market")
    key = _required_env(str(market.get("api_key_env", "ALPACA_API_KEY")))
    secret = _required_env(str(market.get("api_secret_env", "ALPACA_API_SECRET")))
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    url = f"https://data.alpaca.markets/v2/stocks/{quote(symbol)}/bars"
    params: dict[str, Any] = {
        "timeframe": "1Min",
        "start": f"{start}T00:00:00Z" if "T" not in start else start,
        "end": f"{end}T23:59:59Z" if "T" not in end else end,
        "feed": market.get("feed", "sip"),
        "adjustment": market.get("adjustment", "all"),
        "limit": 10000,
        "sort": "asc",
    }
    records: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        if page_token:
            params["page_token"] = page_token
        payload = get_json(url, params=params, headers=headers)
        for bar in payload.get("bars", []) or []:
            records.append(
                {
                    "timestamp": bar.get("t"),
                    "symbol": symbol,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "trade_count": bar.get("n"),
                    "vwap": bar.get("vw"),
                    "source": "alpaca_sip",
                }
            )
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return pd.DataFrame.from_records(records, columns=MARKET_COLUMNS)


def collect_massive_bars(
    config: StrategyConfig, symbol: str, start: str, end: str
) -> pd.DataFrame:
    market = config.section("market")
    key_name = str(market.get("massive_api_key_env", "MASSIVE_API_KEY"))
    key = _required_env(key_name)
    url = (
        f"https://api.massive.com/v2/aggs/ticker/{quote(symbol)}/range/1/minute/"
        f"{quote(start)}/{quote(end)}"
    )
    params: dict[str, Any] = {
        "adjusted": str(market.get("adjustment", "all") != "raw").lower(),
        "sort": "asc",
        "limit": 50000,
        "apiKey": key,
    }
    records: list[dict[str, Any]] = []
    while url:
        payload = get_json(url, params=params)
        for bar in payload.get("results", []) or []:
            records.append(
                {
                    "timestamp": pd.to_datetime(bar.get("t"), unit="ms", utc=True),
                    "symbol": symbol,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "trade_count": bar.get("n"),
                    "vwap": bar.get("vw"),
                    "source": "massive_sip",
                }
            )
        next_url = payload.get("next_url")
        url = str(next_url) if next_url else ""
        params = {"apiKey": key} if url else {}
    return pd.DataFrame.from_records(records, columns=MARKET_COLUMNS)


def collect_public_events(config: StrategyConfig, start: str, end: str) -> list[Path]:
    outputs: list[Path] = []
    outputs.append(collect_sec_submissions(config))
    for feed in config.get("company", "ir_feeds", []):
        outputs.append(collect_rss_atom(config, str(feed)))
    if config.get("events", "gdelt_enabled", True):
        outputs.append(collect_gdelt(config, start, end))
    if config.get("events", "fred_series", []):
        outputs.extend(collect_fred(config, start, end))
    return outputs


def collect_sec_submissions(config: StrategyConfig) -> Path:
    cik = str(config.get("company", "cik", "")).zfill(10)
    env_name = str(config.get("events", "sec_user_agent_env", "SEC_USER_AGENT"))
    user_agent = _required_env(env_name)
    payload = get_json(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers={"User-Agent": user_agent},
    )
    recent = payload.get("filings", {}).get("recent", {})
    keys = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "primaryDocDescription",
    ]
    length = max((len(recent.get(key, [])) for key in keys), default=0)
    rows: list[dict[str, Any]] = []
    for i in range(length):
        row = {key: (recent.get(key, [None] * length) + [None] * length)[i] for key in keys}
        accession = str(row.get("accessionNumber") or "")
        document = str(row.get("primaryDocument") or "")
        accession_clean = accession.replace("-", "")
        rows.append(
            {
                "event_id": f"sec:{accession}",
                "event_time": row.get("acceptanceDateTime"),
                "first_seen_time": row.get("acceptanceDateTime"),
                "symbol": config.target_symbol,
                "event_type": f"sec_{row.get('form')}",
                "direction": 0,
                "evidence_score": 95,
                "surprise_score": 0,
                "is_retrospective": False,
                "headline": row.get("primaryDocDescription") or row.get("form"),
                "source_url": (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accession_clean}/{document}"
                ),
                "report_date": row.get("reportDate"),
                "filing_date": row.get("filingDate"),
            }
        )
    return _save_frame(
        pd.DataFrame(rows), config.data_dir / "raw" / "events" / "sec_submissions.csv"
    )


def collect_rss_atom(config: StrategyConfig, feed_url: str) -> Path:
    payload = get_bytes(feed_url, headers={"User-Agent": "quant-trend-research/0.1"})
    root = ET.fromstring(payload)
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    entries = root.findall(".//item") + root.findall(".//{*}entry")
    for idx, entry in enumerate(entries):
        title = entry.findtext("title") or entry.findtext("{*}title") or ""
        link = entry.findtext("link") or ""
        link_node = entry.find("{*}link")
        if not link and link_node is not None:
            link = link_node.attrib.get("href", "")
        published = (
            entry.findtext("pubDate")
            or entry.findtext("{*}published")
            or entry.findtext("{*}updated")
            or now
        )
        rows.append(
            {
                "event_id": f"rss:{abs(hash((feed_url, link, title)))}:{idx}",
                "event_time": published,
                "first_seen_time": now,
                "symbol": config.target_symbol,
                "event_type": "company_ir",
                "direction": 0,
                "evidence_score": 90,
                "surprise_score": 0,
                "is_retrospective": False,
                "headline": title,
                "source_url": link,
            }
        )
    safe_name = str(abs(hash(feed_url)))
    return _save_frame(
        pd.DataFrame(rows), config.data_dir / "raw" / "events" / f"ir_{safe_name}.csv"
    )


def collect_gdelt(config: StrategyConfig, start: str, end: str) -> Path:
    aliases = config.get("company", "aliases", [config.target_symbol])
    query = " OR ".join(f'"{str(alias)}"' for alias in aliases)
    requested_start = pd.Timestamp(start, tz="UTC")
    requested_end = pd.Timestamp(end, tz="UTC")
    effective_start = max(requested_start, requested_end - pd.Timedelta(days=89))
    params = {
        "query": f"({query}) sourcelang:english",
        "mode": "artlist",
        "maxrecords": int(config.get("events", "gdelt_max_records", 250)),
        "format": "json",
        "sort": "datedesc",
        "startdatetime": effective_start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": requested_end.strftime("%Y%m%d235959"),
    }
    payload = get_json("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for idx, article in enumerate(payload.get("articles", []) or []):
        rows.append(
            {
                "event_id": f"gdelt:{abs(hash(article.get('url', '')))}:{idx}",
                "event_time": article.get("seendate"),
                "first_seen_time": now,
                "symbol": config.target_symbol,
                "event_type": "media_discovery",
                "direction": 0,
                "evidence_score": 30,
                "surprise_score": 0,
                "is_retrospective": False,
                "headline": article.get("title"),
                "source_url": article.get("url"),
                "domain": article.get("domain"),
                "language": article.get("language"),
            }
        )
    return _save_frame(
        pd.DataFrame(rows), config.data_dir / "raw" / "events" / "gdelt.csv"
    )


def collect_fred(config: StrategyConfig, start: str, end: str) -> list[Path]:
    key_name = str(config.get("events", "fred_api_key_env", "FRED_API_KEY"))
    api_key = _required_env(key_name)
    outputs: list[Path] = []
    for series_id in config.get("events", "fred_series", []):
        payload = get_json(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start,
                "observation_end": end,
                "output_type": 4,
            },
        )
        rows = []
        for obs in payload.get("observations", []) or []:
            rows.append(
                {
                    "series_id": series_id,
                    "date": obs.get("date"),
                    "value": obs.get("value"),
                    "realtime_start": obs.get("realtime_start"),
                    "realtime_end": obs.get("realtime_end"),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        outputs.append(
            _save_frame(
                pd.DataFrame(rows),
                config.data_dir / "raw" / "events" / f"fred_{series_id}.csv",
            )
        )
        time.sleep(0.15)
    return outputs
