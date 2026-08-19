from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyConfig


ACTION_TO_POSITION = np.asarray([0.0, 1.0, -1.0], dtype=float)
ACTION_NAMES = ("flat", "long", "short")
OPTION_NAMES = (
    "observe",
    "trend_follow",
    "event_momentum",
    "squeeze_reversal",
    "macro_relief",
)


@dataclass(frozen=True)
class BehaviorDecision:
    option: int
    action: int


def _trend_action(row: pd.Series) -> BehaviorDecision:
    stable = float(row.get("trend_stability_score", 0.0)) >= 55.0
    volume = float(row.get("rvol", 0.0)) >= 0.8
    move = float(row.get("return_from_open_atr", 0.0))
    if stable and volume and abs(move) >= 0.20:
        return BehaviorDecision(1, 1 if move > 0 else 2)
    return BehaviorDecision(0, 0)


def _event_action(row: pd.Series) -> BehaviorDecision:
    if float(row.get("event_macro", 0.0)) > 0 and float(row.get("macro_relief_confirmation", 0.0)) > 0:
        return BehaviorDecision(4, 1)
    if float(row.get("event_active", 0.0)) > 0:
        direction = float(row.get("event_direction", 0.0))
        if direction == 0:
            direction = np.sign(float(row.get("return_from_open", 0.0)))
        if direction != 0:
            return BehaviorDecision(2, 1 if direction > 0 else 2)
    if float(row.get("rotation_confirmed", 0.0)) > 0:
        return BehaviorDecision(2, 1)
    return BehaviorDecision(0, 0)


def _contrarian_action(row: pd.Series) -> BehaviorDecision:
    if float(row.get("squeeze_exhaustion", 0.0)) > 0 or float(row.get("event_exhaustion_score", 0.0)) >= 55:
        return BehaviorDecision(3, 2)
    zscore = float(row.get("vwap_z", 0.0))
    if zscore >= 2.5 and float(row.get("vwap_side", 0.0)) < 0:
        return BehaviorDecision(3, 2)
    if zscore <= -2.5 and float(row.get("vwap_side", 0.0)) > 0:
        return BehaviorDecision(3, 1)
    return BehaviorDecision(0, 0)


def behavior_decision(policy: str, row: pd.Series, rng: np.random.Generator, epsilon: float) -> BehaviorDecision:
    if policy == "flat":
        decision = BehaviorDecision(0, 0)
    elif policy == "trend":
        decision = _trend_action(row)
    elif policy == "event":
        decision = _event_action(row)
    elif policy == "contrarian":
        decision = _contrarian_action(row)
    elif policy == "exploratory":
        candidates = [_trend_action(row), _event_action(row), _contrarian_action(row)]
        decision = candidates[int(rng.integers(0, len(candidates)))]
    else:
        raise ValueError(f"Unknown behavior policy: {policy}")
    if rng.random() < epsilon:
        action = int(rng.integers(0, len(ACTION_NAMES)))
        option = 0 if action == 0 else int(rng.integers(1, len(OPTION_NAMES)))
        return BehaviorDecision(option, action)
    return decision


def _trajectory_rewards(
    close: np.ndarray,
    actions: np.ndarray,
    turnover_bps: float,
    holding_bps: float,
    drawdown_penalty: float,
) -> np.ndarray:
    next_return = np.zeros(len(close), dtype=float)
    next_return[:-1] = np.log(np.clip(close[1:], 1e-9, None) / np.clip(close[:-1], 1e-9, None))
    positions = ACTION_TO_POSITION[actions]
    previous = np.concatenate([[0.0], positions[:-1]])
    turnover = np.abs(positions - previous) * turnover_bps / 10000.0
    holding = np.abs(positions) * holding_bps / 10000.0
    adverse = np.maximum(-positions * next_return, 0.0)
    reward = positions * next_return - turnover - holding - drawdown_penalty * np.square(adverse)
    reward[-1] -= abs(positions[-1]) * turnover_bps / 10000.0
    return reward


def _returns_to_go(rewards: np.ndarray, discount: float) -> np.ndarray:
    result = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        result[index] = running
    return result


def build_offline_trajectories(
    labeled: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    rl = config.section("rl")
    policies = [str(x) for x in rl.get("behavior_policies", ["flat", "trend", "event", "contrarian"])]
    epsilon = float(rl.get("behavior_epsilon", 0.12))
    turnover = float(rl.get("turnover_cost_bps", 2.5))
    holding = float(rl.get("holding_cost_bps", 0.1))
    drawdown_penalty = float(rl.get("drawdown_penalty", 0.15))
    discount = float(rl.get("discount", 1.0))
    seed = int(config.get("backtest", "random_seed", 42))
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    ordered = labeled.sort_values("timestamp").copy()
    for session_date, day in ordered.groupby("session_date", sort=True):
        day = day.reset_index(drop=True)
        close = day["close"].to_numpy(dtype=float)
        for policy in policies:
            decisions = [behavior_decision(policy, row, rng, epsilon) for _, row in day.iterrows()]
            actions = np.asarray([item.action for item in decisions], dtype=int)
            options = np.asarray([item.option for item in decisions], dtype=int)
            rewards = _trajectory_rewards(close, actions, turnover, holding, drawdown_penalty)
            return_to_go = _returns_to_go(rewards, discount)
            trajectory = day.copy()
            trajectory["trajectory_id"] = f"{session_date}:{policy}"
            trajectory["behavior_policy"] = policy
            trajectory["step"] = np.arange(len(day), dtype=int)
            trajectory["option"] = options
            trajectory["action"] = actions
            trajectory["reward"] = rewards
            trajectory["return_to_go"] = return_to_go
            next_option = np.concatenate([options[1:], options[-1:]])
            trajectory["option_terminate"] = ((options != next_option) | (np.arange(len(day)) == len(day) - 1)).astype(float)
            trajectory["done"] = 0.0
            trajectory.loc[len(day) - 1, "done"] = 1.0
            trajectory["trajectory_return"] = float(rewards.sum())
            rows.append(trajectory)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True, sort=False)
    totals = result.groupby("trajectory_id")["trajectory_return"].first()
    scale = max(float(totals.std()), 1e-6)
    quality = 1.0 / (1.0 + np.exp(-(totals - float(totals.median())) / scale))
    result["behavior_quality_weight"] = result["trajectory_id"].map(0.25 + 0.75 * quality)
    path = config.data_dir / "processed" / "trajectories.csv"
    result.to_csv(path, index=False)
    return result

