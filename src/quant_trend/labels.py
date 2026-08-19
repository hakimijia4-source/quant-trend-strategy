from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyConfig


def add_forward_labels(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    output = frame.copy().sort_values("timestamp").reset_index(drop=True)
    snapshot = int(config.get("market", "snapshot_minutes", 5))
    horizon = int(config.get("labels", "horizon_minutes", 60))
    steps = max(1, horizon // snapshot)
    min_bps = float(config.get("labels", "min_abs_return_bps", 20.0))
    min_score = float(config.get("labels", "min_trend_score", 62.0))
    max_adverse_bps = float(config.get("labels", "max_adverse_bps", 45.0))
    output["future_return"] = np.nan
    output["trend_quality"] = np.nan
    output["max_adverse_bps"] = np.nan
    output["trend_label"] = np.nan
    for _, indices in output.groupby("session_date", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        close = output.loc[idx, "close"].to_numpy(dtype=float)
        for position in range(0, len(idx) - steps):
            path = close[position : position + steps + 1]
            if not np.isfinite(path).all() or path[0] <= 0:
                continue
            future_return = float(np.log(path[-1] / path[0]))
            direction = 1 if future_return >= 0 else -1
            path_changes = np.abs(np.diff(path)).sum()
            efficiency = abs(path[-1] - path[0]) / path_changes if path_changes > 1e-12 else 0.0
            persistence = float(np.mean(direction * (path[1:] - path[0]) >= 0))
            path_min = float(path.min())
            path_max = float(path.max())
            span = max(path_max - path_min, 1e-12)
            close_location = (
                (path[-1] - path_min) / span
                if direction > 0
                else (path_max - path[-1]) / span
            )
            if direction > 0:
                adverse = max(0.0, (path[0] - path_min) / path[0])
            else:
                adverse = max(0.0, (path_max - path[0]) / path[0])
            quality = 100.0 * (
                0.45 * np.clip(efficiency, 0, 1)
                + 0.30 * np.clip(persistence, 0, 1)
                + 0.25 * np.clip(close_location, 0, 1)
            )
            adverse_bps = adverse * 10000.0
            magnitude_bps = abs(future_return) * 10000.0
            label = 0
            if magnitude_bps >= min_bps and quality >= min_score and adverse_bps <= max_adverse_bps:
                label = direction
            row = idx[position]
            output.at[row, "future_return"] = future_return
            output.at[row, "trend_quality"] = quality
            output.at[row, "max_adverse_bps"] = adverse_bps
            output.at[row, "trend_label"] = label
    evidence = pd.to_numeric(output.get("evidence_score", 0.0), errors="coerce").fillna(0.0)
    output["sample_weight"] = 0.10 + 0.90 * np.square((evidence / 100.0).clip(0, 1))
    explicit = pd.to_numeric(output.get("event_active", 0.0), errors="coerce").fillna(0.0) > 0
    output.loc[explicit, "sample_weight"] = np.maximum(output.loc[explicit, "sample_weight"], 0.70)
    return output


def build_labeled_dataset(config: StrategyConfig, feature_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if feature_frame is None:
        path = config.data_dir / "processed" / "features.csv"
        feature_frame = pd.read_csv(path)
        feature_frame["timestamp"] = pd.to_datetime(
            feature_frame["timestamp"], utc=True, errors="coerce", format="mixed"
        )
    labeled = add_forward_labels(feature_frame, config)
    path = config.data_dir / "processed" / "labeled.csv"
    labeled.to_csv(path, index=False)
    return labeled

