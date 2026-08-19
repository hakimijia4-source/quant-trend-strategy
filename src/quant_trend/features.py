from __future__ import annotations

import calendar
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .cognition import (
    assert_point_in_time_safe,
    attach_event_features,
    attach_thesis_features,
    load_event_files,
    load_theses,
)
from .config import StrategyConfig


NON_FEATURE_COLUMNS = {
    "timestamp",
    "session_date",
    "symbol",
    "source",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "future_return",
    "trend_label",
    "trend_quality",
    "sample_weight",
    "trajectory_id",
    "step",
    "option",
    "action",
    "reward",
    "return_to_go",
    "option_terminate",
    "done",
    "trajectory_return",
    "behavior_quality_weight",
}


def _rolling_r2(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return np.nan
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    y_std = y.std()
    if y_std <= 1e-12:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr * corr) if np.isfinite(corr) else 0.0


def _rolling_efficiency(values: np.ndarray) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    path = np.abs(np.diff(values)).sum()
    return float(abs(values[-1] - values[0]) / path) if path > 1e-12 else 0.0


def _third_friday(year: int, month: int) -> pd.Timestamp:
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)
    fridays = [day for day in cal.itermonthdates(year, month) if day.month == month and day.weekday() == 4]
    return pd.Timestamp(fridays[2])


def _days_to_monthly_opex(value: object) -> int:
    date = pd.Timestamp(value).normalize()
    expiry = _third_friday(date.year, date.month)
    if date > expiry:
        next_month = date + pd.offsets.MonthBegin(1)
        expiry = _third_friday(next_month.year, next_month.month)
    return int((expiry - date).days)


def load_market_frame(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Market file {path} is missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["symbol"] = frame.get("symbol", symbol or Path(path).stem).astype(str).str.upper()
    for column in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        if column not in frame:
            frame[column] = np.nan if column != "trade_count" else 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["vwap"] = frame["vwap"].fillna(frame["close"])
    frame["trade_count"] = frame["trade_count"].fillna(0.0)
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def resample_bars(
    frame: pd.DataFrame,
    *,
    minutes: int,
    timezone: str,
    regular_session_only: bool,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    local = work["timestamp"].dt.tz_convert(timezone)
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    if regular_session_only:
        work = work[(minute_of_day >= 570) & (minute_of_day < 960)].copy()
        local = work["timestamp"].dt.tz_convert(timezone)
    work["session_date"] = local.dt.date
    work["pv"] = work["vwap"].fillna(work["close"]) * work["volume"].fillna(0.0)
    rows: list[pd.DataFrame] = []
    rule = f"{int(minutes)}min"
    for _, day in work.groupby("session_date", sort=True):
        symbol = str(day["symbol"].iloc[0])
        source = str(day.get("source", pd.Series(["unknown"])).iloc[0])
        sampled = (
            day.set_index("timestamp")
            .resample(rule, label="left", closed="left")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
                trade_count=("trade_count", "sum"),
                pv=("pv", "sum"),
            )
            .dropna(subset=["close"])
            .reset_index()
        )
        sampled["vwap"] = sampled["pv"] / sampled["volume"].replace(0, np.nan)
        sampled["vwap"] = sampled["vwap"].fillna(sampled["close"])
        sampled["symbol"] = symbol
        sampled["source"] = source
        rows.append(sampled.drop(columns="pv"))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("timestamp")


def engineer_symbol_features(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    if frame.empty:
        return frame
    features = config.section("features")
    timezone = str(config.get("project", "timezone", "America/New_York"))
    snapshot_minutes = int(config.get("market", "snapshot_minutes", 5))
    lookback_days = int(features.get("volume_lookback_days", 20))
    rolling_bars = int(features.get("rolling_bars", 12))
    trend_window = int(features.get("trend_window_bars", 12))
    output = resample_bars(
        frame,
        minutes=snapshot_minutes,
        timezone=timezone,
        regular_session_only=bool(config.get("market", "regular_session_only", True)),
    )
    if output.empty:
        return output
    local = output["timestamp"].dt.tz_convert(timezone)
    output["session_date"] = local.dt.date
    output["minute_index"] = output.groupby("session_date").cumcount()
    grouped = output.groupby("session_date", sort=False)
    output["day_open"] = grouped["open"].transform("first")
    output["day_high"] = grouped["high"].cummax()
    output["day_low"] = grouped["low"].cummin()
    output["cum_volume"] = grouped["volume"].cumsum()
    typical = (output["high"] + output["low"] + output["close"]) / 3.0
    output["cum_pv"] = (typical * output["volume"]).groupby(output["session_date"]).cumsum()
    output["session_vwap"] = output["cum_pv"] / output["cum_volume"].replace(0, np.nan)
    output["session_vwap"] = output["session_vwap"].fillna(output["close"])

    min_periods = max(3, lookback_days // 4)
    output["baseline_cum_volume"] = output.groupby("minute_index")["cum_volume"].transform(
        lambda values: values.shift(1).rolling(lookback_days, min_periods=min_periods).median()
    )
    output["rvol"] = output["cum_volume"] / output["baseline_cum_volume"].replace(0, np.nan)
    output["rvol"] = output["rvol"].clip(0, 10)
    vwap_delta = (output["close"] - output["session_vwap"]) / output["session_vwap"].replace(0, np.nan)
    historical_sigma = vwap_delta.groupby(output["minute_index"]).transform(
        lambda values: values.shift(1).rolling(lookback_days, min_periods=min_periods).std()
    )
    output["vwap_z"] = (vwap_delta / historical_sigma.replace(0, np.nan)).clip(-10, 10)
    output["vwap_side"] = np.sign(output["close"] - output["session_vwap"])

    log_close = np.log(output["close"].clip(lower=1e-9))
    for bars in (1, 3, 6, 12):
        output[f"return_{bars}"] = log_close.groupby(output["session_date"]).diff(bars)
    output["return_from_open"] = np.log(output["close"] / output["day_open"].replace(0, np.nan))
    output["efficiency"] = log_close.groupby(output["session_date"]).transform(
        lambda values: values.rolling(trend_window, min_periods=max(3, trend_window // 2)).apply(
            _rolling_efficiency, raw=True
        )
    )
    output["trend_r2"] = log_close.groupby(output["session_date"]).transform(
        lambda values: values.rolling(trend_window, min_periods=max(3, trend_window // 2)).apply(
            _rolling_r2, raw=True
        )
    )

    daily = (
        output.groupby("session_date")
        .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .sort_index()
    )
    previous_close = daily["close"].shift(1)
    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - previous_close).abs(),
            (daily["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr20"] = true_range.shift(1).rolling(20, min_periods=5).mean()
    output["atr20"] = output["session_date"].map(daily["atr20"])
    output["atr20"] = output["atr20"].fillna(
        output.groupby("session_date")["high"].transform("max")
        - output.groupby("session_date")["low"].transform("min")
    )
    output["return_from_open_atr"] = (output["close"] - output["day_open"]) / output[
        "atr20"
    ].replace(0, np.nan)

    above = (output["close"] > output["session_vwap"]).astype(float)
    below = (output["close"] < output["session_vwap"]).astype(float)
    count = output.groupby("session_date").cumcount() + 1
    output["above_vwap_persistence"] = above.groupby(output["session_date"]).cumsum() / count
    output["below_vwap_persistence"] = below.groupby(output["session_date"]).cumsum() / count
    direction = np.sign(output["close"] - output["day_open"])
    output["vwap_persistence"] = np.where(
        direction >= 0, output["above_vwap_persistence"], output["below_vwap_persistence"]
    )
    day_range = (output["day_high"] - output["day_low"]).replace(0, np.nan)
    up_clv = (output["close"] - output["day_low"]) / day_range
    down_clv = (output["day_high"] - output["close"]) / day_range
    output["directional_clv"] = np.where(direction >= 0, up_clv, down_clv).clip(0, 1)
    adverse = np.where(
        direction >= 0,
        output["day_high"] - output["close"],
        output["close"] - output["day_low"],
    )
    output["retrace_score"] = (1.0 - adverse / output["atr20"].replace(0, np.nan)).clip(0, 1)
    output["volume_score"] = ((output["rvol"] - 0.8) / 1.2).clip(0, 1)
    output["trend_stability_score"] = 100.0 * (
        0.25 * output["efficiency"].clip(0, 1)
        + 0.20 * output["trend_r2"].clip(0, 1)
        + 0.20 * output["vwap_persistence"].clip(0, 1)
        + 0.15 * output["directional_clv"].clip(0, 1)
        + 0.10 * output["retrace_score"].clip(0, 1)
        + 0.10 * output["volume_score"].clip(0, 1)
    )
    low = float(features.get("rvol_low", 0.8))
    high = float(features.get("rvol_high", 1.5))
    output["volume_tier"] = np.select(
        [output["rvol"] < low, output["rvol"] < high], [1, 2], default=3
    ).astype(float)

    final_rvol = output.groupby("session_date")["rvol"].last()
    previous_final_rvol = final_rvol.shift(1)
    output["previous_day_rvol"] = output["session_date"].map(previous_final_rvol)
    output["volume_divergence"] = (
        (output["return_from_open"] > 0)
        & (output["rvol"] < 0.70 * output["previous_day_rvol"])
    ).astype(float)
    output["squeeze_exhaustion"] = (
        (output["return_from_open_atr"] > 0.50)
        & (output["volume_divergence"] > 0)
        & ((output["vwap_side"] < 0) | (output["directional_clv"] < 0.50))
    ).astype(float)

    output["is_friday"] = local.dt.weekday.eq(4).astype(float)
    output["days_to_monthly_opex"] = output["session_date"].map(_days_to_monthly_opex)
    output["is_monthly_opex"] = output["days_to_monthly_opex"].eq(0).astype(float)
    output["opex_exhaustion"] = output["is_friday"] * output["squeeze_exhaustion"]
    output["sin_time"] = np.sin(2 * np.pi * output["minute_index"] / max(1, 390 // snapshot_minutes))
    output["cos_time"] = np.cos(2 * np.pi * output["minute_index"] / max(1, 390 // snapshot_minutes))
    return output.replace([np.inf, -np.inf], np.nan)


def _context_view(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    columns = [
        "timestamp",
        "return_from_open",
        "return_from_open_atr",
        "return_6",
        "return_12",
        "rvol",
        "vwap_side",
        "trend_stability_score",
    ]
    available = [column for column in columns if column in frame]
    view = frame[available].copy()
    return view.rename(columns={column: f"ctx_{symbol}_{column}" for column in available if column != "timestamp"})


def attach_lagged_fred_features(frame: pd.DataFrame, event_dir: str | Path) -> pd.DataFrame:
    """Attach official daily macro values no earlier than the next session.

    The exact publication timestamp differs by series.  A one-session lag is a
    conservative default that prevents a same-day intraday look-ahead.  For
    revised economic releases, an ALFRED vintage export should replace these
    current-vintage files before production research.
    """
    output = frame.copy()
    directory = Path(event_dir)
    paths = sorted(directory.glob("fred_*.csv")) if directory.exists() else []
    if not paths or output.empty:
        return output
    sessions = pd.DataFrame(
        {"session_date": sorted(pd.unique(output["session_date"]))}
    )
    # pandas may preserve a seconds-resolution dtype for Python ``date`` values
    # while CSV timestamps arrive at microsecond resolution.  merge_asof requires
    # an exact dtype match, so normalize both join keys explicitly to nanoseconds.
    sessions["session_ts"] = pd.to_datetime(
        sessions["session_date"], errors="coerce"
    ).astype("datetime64[ns]")
    for path in paths:
        macro = pd.read_csv(path)
        if macro.empty or not {"date", "value"}.issubset(macro.columns):
            continue
        series_id = str(macro.get("series_id", pd.Series([path.stem.removeprefix("fred_")])).iloc[0])
        macro["source_date"] = pd.to_datetime(macro["date"], errors="coerce").dt.normalize()
        macro["value"] = pd.to_numeric(macro["value"], errors="coerce")
        macro = macro.dropna(subset=["source_date", "value"]).sort_values("source_date")
        if macro.empty:
            continue
        if "realtime_start" in macro:
            release_date = pd.to_datetime(macro["realtime_start"], errors="coerce").dt.normalize()
            release_date = release_date.fillna(macro["source_date"])
        else:
            release_date = macro["source_date"]
        # The API gives a release date but not a guaranteed intraday availability
        # time, so make the value usable from the following business session.
        macro["available_session"] = release_date + pd.offsets.BDay(1)
        macro["available_session"] = macro["available_session"].astype("datetime64[ns]")
        macro = macro.drop_duplicates("available_session", keep="last")
        merged = pd.merge_asof(
            sessions.sort_values("session_ts"),
            macro[["available_session", "value"]].sort_values("available_session"),
            left_on="session_ts",
            right_on="available_session",
            direction="backward",
        )
        safe_id = "".join(character if character.isalnum() else "_" for character in series_id)
        level_name = f"macro_{safe_id}_level"
        change_name = f"macro_{safe_id}_change"
        merged[level_name] = merged["value"]
        merged[change_name] = merged["value"].diff()
        level_map = dict(zip(merged["session_date"], merged[level_name]))
        change_map = dict(zip(merged["session_date"], merged[change_name]))
        output[level_name] = output["session_date"].map(level_map)
        output[change_name] = output["session_date"].map(change_map)
    return output


def build_feature_frame(
    config: StrategyConfig,
    *,
    thesis_path: str | Path | None = None,
    manual_events: str | Path | None = None,
) -> pd.DataFrame:
    market_dir = config.data_dir / "raw" / "market"
    symbols = list(dict.fromkeys([config.target_symbol, *config.context_symbols]))
    engineered: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = market_dir / f"{symbol}.csv"
        if not path.exists():
            if symbol == config.target_symbol:
                raise FileNotFoundError(f"Missing target market data: {path}")
            continue
        engineered[symbol] = engineer_symbol_features(load_market_frame(path, symbol), config)
    target = engineered[config.target_symbol].copy()
    for symbol, frame in engineered.items():
        if symbol == config.target_symbol or frame.empty:
            continue
        target = target.merge(_context_view(frame, symbol), on="timestamp", how="left")

    market_ref = str(config.get("features", "market_reference", "SPY")).upper()
    rotation_ref = str(config.get("features", "rotation_reference", "SOXX")).upper()
    market_col = f"ctx_{market_ref}_return_from_open"
    rotation_col = f"ctx_{rotation_ref}_return_from_open"
    rotation_atr_col = f"ctx_{rotation_ref}_return_from_open_atr"
    rotation_rvol_col = f"ctx_{rotation_ref}_rvol"
    rotation_vwap_col = f"ctx_{rotation_ref}_vwap_side"
    target["target_excess_market"] = target["return_from_open"] - target.get(market_col, 0.0)
    target["target_excess_rotation"] = target["return_from_open"] - target.get(rotation_col, 0.0)
    semis_down = (
        -target.get(rotation_atr_col, pd.Series(0.0, index=target.index))
    ).clip(0, 2) / 2.0
    semis_volume = (
        (target.get(rotation_rvol_col, pd.Series(1.0, index=target.index)) - 1.0) / 1.0
    ).clip(0, 1)
    target_relative = (target["target_excess_rotation"] * 30.0).clip(0, 1)
    target_volume = ((target["rvol"] - 0.8) / 0.7).clip(0, 1)
    target_vwap = target["vwap_side"].gt(0).astype(float)
    target["rotation_score"] = (
        25 * semis_down
        + 20 * semis_volume
        + 25 * target_relative
        + 15 * target_volume
        + 15 * target_vwap
    )
    target["rotation_confirmed"] = (target["rotation_score"] >= 65).astype(float)

    events = load_event_files(config.data_dir / "raw" / "events", manual_events)
    theses = load_theses(thesis_path)
    assert_point_in_time_safe(events, theses)
    target = attach_event_features(
        target,
        events,
        config.target_symbol,
        int(config.get("events", "event_decay_minutes", 2880)),
    )
    target = attach_thesis_features(target, theses, config.target_symbol)
    target = attach_lagged_fred_features(target, config.data_dir / "raw" / "events")
    bond_relief = target.get("ctx_TLT_return_6", pd.Series(0.0, index=target.index)).clip(lower=0)
    semis_reclaim = target.get(rotation_vwap_col, pd.Series(0.0, index=target.index)).gt(0).astype(float)
    target["macro_relief_confirmation"] = (
        target["event_macro"]
        * (bond_relief * 100.0).clip(0, 1)
        * semis_reclaim
    )
    target["event_exhaustion_score"] = (
        45 * target["volume_divergence"]
        + 35 * target["squeeze_exhaustion"]
        + 20 * target["opex_exhaustion"]
    )
    target["evidence_score"] = 100.0 * np.maximum.reduce(
        [
            target["event_evidence"].fillna(0).to_numpy(),
            target["cognition_prior"].fillna(0).to_numpy(),
            (target["rotation_score"].fillna(0) / 100.0).to_numpy(),
        ]
    )
    target = target.sort_values("timestamp").reset_index(drop=True)
    numeric = target.select_dtypes(include=[np.number]).columns
    target[numeric] = target[numeric].replace([np.inf, -np.inf], np.nan)
    output_path = config.data_dir / "processed" / "features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(output_path, index=False)
    return target


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column not in NON_FEATURE_COLUMNS:
            columns.append(column)
    return columns
