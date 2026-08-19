from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .environment import build_offline_trajectories
from .features import build_feature_frame
from .labels import build_labeled_dataset


def generate_synthetic_data(config: StrategyConfig, sessions: int = 180) -> Path:
    """Generate a deterministic market-like dataset for an end-to-end smoke test."""
    rng = np.random.default_rng(int(config.get("backtest", "random_seed", 42)))
    timezone = str(config.get("project", "timezone", "America/New_York"))
    symbols = list(dict.fromkeys([config.target_symbol, *config.context_symbols]))
    dates = pd.bdate_range(str(config.get("market", "start", "2022-01-03")), periods=sessions)
    prices = {symbol: 100.0 + 30.0 * rng.random() for symbol in symbols}
    betas = {symbol: 0.75 + 0.55 * rng.random() for symbol in symbols}
    betas["SPY"] = 1.0
    if "SOXX" in betas:
        betas["SOXX"] = 1.35
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}
    event_rows: list[dict[str, Any]] = []
    thesis_rows: list[dict[str, Any]] = []
    for day_index, date in enumerate(dates):
        start = pd.Timestamp(date.date()).tz_localize(timezone) + pd.Timedelta(hours=9, minutes=30)
        timestamps = pd.date_range(start, periods=78, freq="5min").tz_convert("UTC")
        market_noise = rng.normal(0.0, 0.00045, len(timestamps))
        market_drift = rng.normal(0.0, 0.00008)
        semis_crash = day_index % 45 in {20, 21}
        relief_rebound = day_index % 45 == 22
        target_trend = day_index % 30 in {8, 9}
        exhaustion_day = day_index % 55 == 33
        for symbol in symbols:
            symbol_noise = rng.normal(0.0, 0.00035, len(timestamps))
            drift = market_drift
            volume_multiplier = 1.0
            if symbol == "SOXX" and semis_crash:
                drift -= 0.00055
                volume_multiplier = 2.2
            if symbol == "SOXX" and relief_rebound:
                drift += 0.00065
                volume_multiplier = 1.9
            if symbol == config.target_symbol and semis_crash:
                drift += 0.00020
                volume_multiplier = 1.5
            if symbol == config.target_symbol and target_trend:
                drift += 0.00032
                volume_multiplier = 1.7
            if symbol == config.target_symbol and exhaustion_day:
                drift += 0.00042
                volume_multiplier = 0.72
            returns = betas[symbol] * market_noise + symbol_noise + drift
            if symbol == config.target_symbol and exhaustion_day:
                returns[-18:] -= 0.00105
            for bar_index, (timestamp, log_return) in enumerate(zip(timestamps, returns)):
                open_price = prices[symbol]
                close_price = open_price * float(np.exp(log_return))
                spread = abs(rng.normal(0.00035, 0.00012))
                high = max(open_price, close_price) * (1.0 + spread)
                low = min(open_price, close_price) * (1.0 - spread)
                u_shape = 1.0 + 1.2 * abs(bar_index - 38.5) / 38.5
                volume = max(100.0, rng.lognormal(10.2, 0.28) * u_shape * volume_multiplier)
                rows_by_symbol[symbol].append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": symbol,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close_price,
                        "volume": volume,
                        "trade_count": max(1, int(volume / 120)),
                        "vwap": (open_price + high + low + close_price) / 4.0,
                        "source": "synthetic",
                    }
                )
                prices[symbol] = close_price
        if relief_rebound:
            event_time = timestamps[54]
            event_rows.append(
                {
                    "event_id": f"demo_fomc_{day_index}",
                    "event_time": event_time.isoformat(),
                    "first_seen_time": event_time.isoformat(),
                    "symbol": config.target_symbol,
                    "event_type": "macro_policy",
                    "direction": 1,
                    "evidence_score": 90,
                    "surprise_score": 65,
                    "is_retrospective": False,
                    "headline": "Synthetic no-hike relief event",
                    "source_url": "synthetic://fomc",
                }
            )
            thesis_rows.append(
                {
                    "thesis_id": f"demo_fed_{day_index}",
                    "created_at": event_time.isoformat(),
                    "valid_from": event_time.isoformat(),
                    "expires_at": (event_time + pd.Timedelta(days=1)).isoformat(),
                    "target_symbol": config.target_symbol,
                    "regime_code": "fed_no_hike_short_squeeze",
                    "expected_direction": 1,
                    "expected_horizon": "15m-1d",
                    "prior_confidence": 70,
                    "causal_chain": "Synthetic policy relief",
                    "confirmation_rules": ["Rates proxy rises", "SOXX reclaims VWAP"],
                    "invalidation_rules": ["SOXX loses event VWAP"],
                    "is_retrospective": False,
                    "original_text": "Synthetic point-in-time thesis",
                }
            )
        if semis_crash and day_index % 45 == 20:
            event_time = timestamps[12]
            thesis_rows.append(
                {
                    "thesis_id": f"demo_rotation_{day_index}",
                    "created_at": event_time.isoformat(),
                    "valid_from": event_time.isoformat(),
                    "expires_at": (event_time + pd.Timedelta(days=3)).isoformat(),
                    "target_symbol": config.target_symbol,
                    "regime_code": "semis_to_target_rotation",
                    "expected_direction": 1,
                    "expected_horizon": "1d-3d",
                    "prior_confidence": 68,
                    "causal_chain": "Synthetic semiconductor deleveraging and target rotation",
                    "confirmation_rules": ["SOXX high-volume weakness", "Target above VWAP"],
                    "invalidation_rules": ["Target loses VWAP"],
                    "is_retrospective": False,
                    "original_text": "Synthetic point-in-time rotation thesis",
                }
            )
    market_dir = config.data_dir / "raw" / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    for symbol, rows in rows_by_symbol.items():
        pd.DataFrame(rows).to_csv(market_dir / f"{symbol}.csv", index=False)
    event_dir = config.data_dir / "raw" / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(event_rows).to_csv(event_dir / "synthetic_events.csv", index=False)
    thesis_path = config.data_dir / "theses.jsonl"
    with thesis_path.open("w", encoding="utf-8") as handle:
        for row in thesis_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return thesis_path


def prepare_demo(config: StrategyConfig, sessions: int = 180) -> pd.DataFrame:
    thesis_path = generate_synthetic_data(config, sessions=sessions)
    features = build_feature_frame(config, thesis_path=thesis_path)
    labeled = build_labeled_dataset(config, features)
    return build_offline_trajectories(labeled, config)


def run_demo(config: StrategyConfig, sessions: int = 180, train: bool = True) -> dict[str, Any]:
    trajectories = prepare_demo(config, sessions=sessions)
    result: dict[str, Any] = {
        "sessions": int(trajectories["session_date"].nunique()),
        "trajectories": int(trajectories["trajectory_id"].nunique()),
        "rows": int(len(trajectories)),
    }
    if train:
        try:
            from .trainer import train_walk_forward
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    'PyTorch is required for training. Install with: pip install -e ".[train]"'
                ) from exc
            raise
        folds = train_walk_forward(config, trajectories=trajectories, max_folds=1)
        result["trained_folds"] = len(folds)
        result["test_metrics"] = folds[-1]["test_metrics"]
    return result

