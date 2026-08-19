from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIME_FEATURES = {
    "semis_to_target_rotation": "cog_semis_rotation",
    "fed_no_hike_short_squeeze": "cog_fed_squeeze",
    "event_squeeze_exhaustion": "cog_event_exhaustion",
    "opex_exhaustion_interaction": "cog_opex_interaction",
}


def parse_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")


@dataclass(frozen=True)
class Thesis:
    thesis_id: str
    created_at: pd.Timestamp
    valid_from: pd.Timestamp
    expires_at: pd.Timestamp
    target_symbol: str
    regime_code: str
    expected_direction: int
    prior_confidence: float
    is_retrospective: bool
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Thesis":
        created = pd.to_datetime(payload["created_at"], utc=True)
        valid_from = pd.to_datetime(payload["valid_from"], utc=True)
        expires = pd.to_datetime(payload["expires_at"], utc=True)
        retrospective = bool(payload.get("is_retrospective", False))
        if not retrospective and created > valid_from:
            raise ValueError(
                f"Thesis {payload.get('thesis_id')} was created after its valid_from time"
            )
        if expires < valid_from:
            raise ValueError(f"Thesis {payload.get('thesis_id')} expires before it starts")
        return cls(
            thesis_id=str(payload["thesis_id"]),
            created_at=created,
            valid_from=valid_from,
            expires_at=expires,
            target_symbol=str(payload["target_symbol"]).upper(),
            regime_code=str(payload["regime_code"]),
            expected_direction=int(payload.get("expected_direction", 0)),
            prior_confidence=float(payload.get("prior_confidence", 0.0)),
            is_retrospective=retrospective,
            payload=payload,
        )


def load_theses(path: str | Path | None) -> list[Thesis]:
    if path is None:
        return []
    source = Path(path)
    if not source.exists():
        return []
    theses: list[Thesis] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                theses.append(Thesis.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid thesis at {source}:{line_number}: {exc}") from exc
    return theses


def attach_thesis_features(
    frame: pd.DataFrame, theses: list[Thesis], symbol: str
) -> pd.DataFrame:
    output = frame.copy()
    feature_names = list(REGIME_FEATURES.values())
    for name in feature_names:
        output[name] = 0.0
    output["cognition_active"] = 0.0
    output["cognition_prior"] = 0.0
    output["cognition_direction"] = 0.0
    if output.empty:
        return output
    timestamps = pd.to_datetime(output["timestamp"], utc=True)
    for thesis in theses:
        if thesis.is_retrospective or thesis.target_symbol != symbol.upper():
            continue
        available_from = max(thesis.created_at, thesis.valid_from)
        active = (timestamps >= available_from) & (timestamps <= thesis.expires_at)
        if not active.any():
            continue
        confidence = float(np.clip(thesis.prior_confidence / 100.0, 0.0, 1.0))
        duration = max((thesis.expires_at - available_from).total_seconds(), 1.0)
        elapsed = (timestamps[active] - available_from).dt.total_seconds()
        decay = np.exp(-elapsed / duration)
        values = confidence * decay.to_numpy()
        feature = REGIME_FEATURES.get(thesis.regime_code)
        if feature:
            output.loc[active, feature] = np.maximum(output.loc[active, feature], values)
        output.loc[active, "cognition_active"] = 1.0
        output.loc[active, "cognition_prior"] = np.maximum(
            output.loc[active, "cognition_prior"], values
        )
        stronger = values >= output.loc[active, "cognition_prior"].to_numpy()
        active_index = output.index[active]
        output.loc[active_index[stronger], "cognition_direction"] = thesis.expected_direction
    return output


def load_event_files(event_dir: str | Path, manual_events: str | Path | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    directory = Path(event_dir)
    if directory.exists():
        for path in sorted(directory.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if {"event_time", "symbol"}.issubset(frame.columns):
                frames.append(frame)
    if manual_events:
        path = Path(manual_events)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    events = pd.concat(frames, ignore_index=True, sort=False)
    events["event_time"] = parse_utc(events["event_time"])
    if "first_seen_time" not in events:
        events["first_seen_time"] = events["event_time"]
    events["first_seen_time"] = parse_utc(events["first_seen_time"])
    events["available_time"] = events[["event_time", "first_seen_time"]].max(axis=1)
    if "is_retrospective" not in events:
        events["is_retrospective"] = False
    events["is_retrospective"] = (
        events["is_retrospective"].astype(str).str.lower().isin({"true", "1", "yes"})
    )
    events = events[~events["is_retrospective"]].copy()
    events["symbol"] = events["symbol"].astype(str).str.upper()
    for column, default in (
        ("evidence_score", 0.0),
        ("surprise_score", 0.0),
        ("direction", 0.0),
    ):
        events[column] = pd.to_numeric(events.get(column, default), errors="coerce").fillna(default)
    return events.sort_values("available_time")


def attach_event_features(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    symbol: str,
    decay_minutes: int,
) -> pd.DataFrame:
    output = frame.sort_values("timestamp").copy()
    defaults = {
        "event_active": 0.0,
        "event_evidence": 0.0,
        "event_surprise": 0.0,
        "event_direction": 0.0,
        "minutes_since_event": float(decay_minutes + 1),
        "event_sec": 0.0,
        "event_company": 0.0,
        "event_media": 0.0,
        "event_macro": 0.0,
    }
    for column, value in defaults.items():
        output[column] = value
    if events.empty or output.empty:
        return output
    relevant = events[(events["symbol"] == symbol.upper()) | (events["symbol"] == "ALL")].copy()
    relevant = relevant.dropna(subset=["available_time"])
    if relevant.empty:
        return output
    right_columns = [
        "available_time",
        "event_type",
        "evidence_score",
        "surprise_score",
        "direction",
    ]
    merged = pd.merge_asof(
        output.sort_values("timestamp"),
        relevant[right_columns].sort_values("available_time"),
        left_on="timestamp",
        right_on="available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    age = (merged["timestamp"] - merged["available_time"]).dt.total_seconds() / 60.0
    active = age.between(0, decay_minutes, inclusive="both")
    decay = np.exp(-age.clip(lower=0).fillna(decay_minutes + 1) / max(decay_minutes, 1))
    merged["event_active"] = active.astype(float)
    merged["minutes_since_event"] = age.where(active, decay_minutes + 1).fillna(decay_minutes + 1)
    merged["event_evidence"] = (
        merged["evidence_score"].fillna(0.0) / 100.0 * decay * active
    )
    merged["event_surprise"] = merged["surprise_score"].fillna(0.0) / 100.0 * decay * active
    merged["event_direction"] = merged["direction"].fillna(0.0) * active
    event_type = merged["event_type"].fillna("").astype(str).str.lower()
    merged["event_sec"] = (event_type.str.startswith("sec_") & active).astype(float)
    merged["event_company"] = (event_type.str.contains("company|technology|earn") & active).astype(float)
    merged["event_media"] = (event_type.str.contains("media") & active).astype(float)
    merged["event_macro"] = (event_type.str.contains("macro|fomc|policy") & active).astype(float)
    return merged.drop(
        columns=[
            "available_time",
            "event_type",
            "evidence_score",
            "surprise_score",
            "direction",
        ],
        errors="ignore",
    )


def assert_point_in_time_safe(events: pd.DataFrame, theses: list[Thesis]) -> None:
    if not events.empty:
        invalid = events[events["first_seen_time"] < events["event_time"]]
        if not invalid.empty:
            raise ValueError("Event first_seen_time is earlier than event_time")
    for thesis in theses:
        if not thesis.is_retrospective and thesis.created_at > thesis.valid_from:
            raise ValueError(f"Thesis {thesis.thesis_id} leaks future information")

