from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class RobustFeatureScaler:
    columns: list[str]
    median: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "RobustFeatureScaler":
        values = frame.reindex(columns=self.columns).apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.median = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        iqr = q75 - q25
        std = np.nanstd(values, axis=0)
        scale = np.where(iqr > 1e-8, iqr, np.where(std > 1e-8, std, 1.0))
        self.median = np.nan_to_num(self.median, nan=0.0)
        self.scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.median is None or self.scale is None:
            raise RuntimeError("Scaler has not been fit")
        values = frame.reindex(columns=self.columns).apply(pd.to_numeric, errors="coerce").to_numpy(float)
        values = np.where(np.isfinite(values), values, self.median)
        return np.clip((values - self.median) / self.scale, -10.0, 10.0).astype(np.float32)

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez(target / "scaler.npz", median=self.median, scale=self.scale)
        (target / "feature_spec.json").write_text(
            json.dumps({"columns": self.columns}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "RobustFeatureScaler":
        target = Path(directory)
        spec = json.loads((target / "feature_spec.json").read_text(encoding="utf-8"))
        arrays = np.load(target / "scaler.npz")
        return cls(
            columns=list(spec["columns"]),
            median=arrays["median"],
            scale=arrays["scale"],
        )


@dataclass(frozen=True)
class TimeSplit:
    fold: int
    train_dates: tuple[object, ...]
    validation_dates: tuple[object, ...]
    test_dates: tuple[object, ...]


def walk_forward_splits(
    frame: pd.DataFrame,
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
) -> list[TimeSplit]:
    dates = tuple(sorted(pd.unique(frame["session_date"])))
    required = train_days + validation_days + test_days
    if len(dates) < required:
        raise ValueError(
            f"Need at least {required} sessions for walk-forward training; found {len(dates)}"
        )
    splits: list[TimeSplit] = []
    end = required
    fold = 0
    while end <= len(dates):
        start = end - required
        train_end = start + train_days
        validation_end = train_end + validation_days
        splits.append(
            TimeSplit(
                fold=fold,
                train_dates=dates[start:train_end],
                validation_dates=dates[train_end:validation_end],
                test_dates=dates[validation_end:end],
            )
        )
        fold += 1
        end += step_days
    return splits


def slice_dates(frame: pd.DataFrame, dates: tuple[object, ...]) -> pd.DataFrame:
    return frame[frame["session_date"].isin(set(dates))].copy()


class TrajectorySequenceIndex:
    def __init__(
        self,
        frame: pd.DataFrame,
        state_values: np.ndarray,
        sequence_length: int,
    ) -> None:
        if len(frame) != len(state_values):
            raise ValueError("Frame and state array lengths do not match")
        self.frame = frame.reset_index(drop=True)
        self.states = np.asarray(state_values, dtype=np.float32)
        self.sequence_length = int(sequence_length)
        self.samples: list[tuple[int, int]] = []
        for _, indices in self.frame.groupby("trajectory_id", sort=False).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            if len(positions) < self.sequence_length:
                continue
            for end_offset in range(self.sequence_length - 1, len(positions)):
                end_position = positions[end_offset]
                start_position = positions[end_offset - self.sequence_length + 1]
                self.samples.append((start_position, end_position + 1))

    def __len__(self) -> int:
        return len(self.samples)

    def arrays(self, index: int) -> dict[str, np.ndarray]:
        start, end = self.samples[index]
        window = self.frame.iloc[start:end]
        actions = window["action"].to_numpy(dtype=np.int64)
        options = window["option"].to_numpy(dtype=np.int64)
        previous_actions = np.concatenate([[0], actions[:-1]]).astype(np.int64)
        previous_options = np.concatenate([[0], options[:-1]]).astype(np.int64)
        return {
            "states": self.states[start:end],
            "actions": actions,
            "options": options,
            "previous_actions": previous_actions,
            "previous_options": previous_options,
            "returns_to_go": window["return_to_go"].to_numpy(dtype=np.float32)[:, None],
            "rewards": window["reward"].to_numpy(dtype=np.float32),
            "termination": window["option_terminate"].to_numpy(dtype=np.float32),
            "weights": (
                window.get("sample_weight", pd.Series(1.0, index=window.index)).to_numpy(dtype=np.float32)
                * window.get("behavior_quality_weight", pd.Series(1.0, index=window.index)).to_numpy(dtype=np.float32)
            ),
            "timesteps": window["step"].to_numpy(dtype=np.int64),
            "mask": np.ones(len(window), dtype=np.float32),
        }


class TransitionIndex:
    def __init__(self, frame: pd.DataFrame, states: np.ndarray) -> None:
        self.frame = frame.reset_index(drop=True)
        self.states = np.asarray(states, dtype=np.float32)
        self.samples: list[tuple[int, int]] = []
        for _, indices in self.frame.groupby("trajectory_id", sort=False).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            for offset in range(len(positions) - 1):
                self.samples.append((positions[offset], positions[offset + 1]))

    def __len__(self) -> int:
        return len(self.samples)

    def arrays(self, index: int) -> dict[str, np.ndarray | float | int]:
        current, next_index = self.samples[index]
        row = self.frame.iloc[current]
        return {
            "state": self.states[current],
            "next_state": self.states[next_index],
            "action": int(row["action"]),
            "reward": float(row["reward"]),
            "done": float(row["done"]),
        }

