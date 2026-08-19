from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .config import StrategyConfig
from .environment import ACTION_NAMES, ACTION_TO_POSITION, OPTION_NAMES
from .features import feature_columns
from .rl_data import (
    RobustFeatureScaler,
    TimeSplit,
    TrajectorySequenceIndex,
    TransitionIndex,
    slice_dates,
    walk_forward_splits,
)
from .rl_models import (
    ExogenousWorldModel,
    HierarchicalDecisionTransformer,
    ModelDimensions,
    hierarchical_dt_loss,
    world_model_loss,
)


class TorchSequenceDataset(Dataset):
    def __init__(self, index: TrajectorySequenceIndex) -> None:
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        arrays = self.index.arrays(item)
        return {key: torch.from_numpy(value) for key, value in arrays.items()}


class TorchTransitionDataset(Dataset):
    def __init__(self, index: TransitionIndex) -> None:
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        arrays = self.index.arrays(item)
        return {
            "state": torch.as_tensor(arrays["state"], dtype=torch.float32),
            "next_state": torch.as_tensor(arrays["next_state"], dtype=torch.float32),
            "action": torch.as_tensor(arrays["action"], dtype=torch.long),
            "reward": torch.as_tensor(arrays["reward"], dtype=torch.float32),
            "done": torch.as_tensor(arrays["done"], dtype=torch.float32),
        }


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    names = set().union(*(row.keys() for row in metrics))
    return {name: float(np.mean([row[name] for row in metrics if name in row])) for name in names}


def _world_epoch(
    model: ExogenousWorldModel,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    kl_weight: float,
    free_nats: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics: list[dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            outputs = model(batch["state"], batch["action"], batch["next_state"])
            loss, row = world_model_loss(
                outputs,
                next_state=batch["next_state"],
                reward=batch["reward"],
                done=batch["done"],
                kl_weight=kl_weight,
                free_nats=free_nats,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics.append(row)
    return _mean_metrics(metrics)


def _dt_epoch(
    model: HierarchicalDecisionTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    rl: dict[str, Any],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics: list[dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            outputs = model(
                batch["states"],
                batch["returns_to_go"],
                batch["previous_actions"],
                batch["previous_options"],
                batch["timesteps"],
                options_for_action=batch["options"],
            )
            loss, row = hierarchical_dt_loss(
                outputs,
                actions=batch["actions"],
                options=batch["options"],
                termination=batch["termination"],
                returns_to_go=batch["returns_to_go"],
                weights=batch["weights"],
                mask=batch["mask"],
                option_weight=float(rl.get("option_loss_weight", 0.6)),
                action_weight=float(rl.get("action_loss_weight", 1.0)),
                value_weight=float(rl.get("value_loss_weight", 0.25)),
                termination_weight=float(rl.get("termination_loss_weight", 0.15)),
                label_smoothing=float(rl.get("label_smoothing", 0.03)),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics.append(row)
    return _mean_metrics(metrics)


def _train_world_model(
    model: ExogenousWorldModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: StrategyConfig,
    device: torch.device,
) -> tuple[ExogenousWorldModel, list[dict[str, Any]]]:
    model = model.to(device)
    model_cfg = config.section("model")
    rl = config.section("rl")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
    )
    epochs = max(3, int(model_cfg.get("epochs", 40)) // 2)
    patience = int(model_cfg.get("patience", 6))
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        train_metrics = _world_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            kl_weight=float(rl.get("world_model_kl_weight", 0.05)),
            free_nats=float(rl.get("world_model_free_nats", 1.0)),
        )
        validation_metrics = _world_epoch(
            model,
            validation_loader,
            device,
            optimizer=None,
            kl_weight=float(rl.get("world_model_kl_weight", 0.05)),
            free_nats=float(rl.get("world_model_free_nats", 1.0)),
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        value = validation_metrics.get("total", float("inf"))
        if value < best_loss - 1e-6:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def _train_decision_transformer(
    model: HierarchicalDecisionTransformer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: StrategyConfig,
    device: torch.device,
) -> tuple[HierarchicalDecisionTransformer, list[dict[str, Any]]]:
    model = model.to(device)
    model_cfg = config.section("model")
    rl = config.section("rl")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(model_cfg.get("epochs", 40))
    patience = int(model_cfg.get("patience", 6))
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        train_metrics = _dt_epoch(model, train_loader, device, optimizer=optimizer, rl=rl)
        validation_metrics = _dt_epoch(model, validation_loader, device, optimizer=None, rl=rl)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        value = validation_metrics.get("total", float("inf"))
        if value < best_loss - 1e-6:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return float((centre - spread) / denominator)


@torch.no_grad()
def rollout_policy(
    model: HierarchicalDecisionTransformer,
    world_model: ExogenousWorldModel,
    frame: pd.DataFrame,
    scaler: RobustFeatureScaler,
    config: StrategyConfig,
    *,
    desired_return: float,
    probability_threshold: float,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    world_model.eval()
    unique = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="first").copy()
    states = scaler.transform(unique)
    sequence_length = int(config.get("model", "sequence_length", 24))
    snapshot = int(config.get("market", "snapshot_minutes", 5))
    min_gap_steps = max(1, int(config.get("backtest", "min_gap_minutes", 60)) // snapshot)
    rl = config.section("rl")
    turnover = float(rl.get("turnover_cost_bps", 2.5)) / 10000.0
    holding = float(rl.get("holding_cost_bps", 0.1)) / 10000.0
    drawdown_penalty = float(rl.get("drawdown_penalty", 0.15))
    rerank_weight = float(rl.get("world_model_rerank_weight", 0.20))
    imagination_horizon = int(rl.get("imagination_horizon", 12))
    discount = float(rl.get("discount", 1.0))
    records: list[dict[str, Any]] = []
    offset = 0
    for _, day in unique.groupby("session_date", sort=True):
        day = day.reset_index(drop=True)
        day_states = states[offset : offset + len(day)]
        offset += len(day)
        action_history: list[int] = []
        option_history: list[int] = []
        rtg_history: list[float] = [float(desired_return)]
        current_option = 0
        last_entry_step = -10_000
        close = day["close"].to_numpy(float)
        for step in range(len(day)):
            start = max(0, step - sequence_length + 1)
            state_sequence = torch.from_numpy(day_states[start : step + 1]).unsqueeze(0).to(device)
            previous_actions = np.asarray([0, *action_history], dtype=np.int64)[start : step + 1]
            previous_options = np.asarray([0, *option_history], dtype=np.int64)[start : step + 1]
            rtg_sequence = np.asarray(rtg_history[start : step + 1], dtype=np.float32)[:, None]
            time_sequence = np.arange(start, step + 1, dtype=np.int64)
            outputs = model(
                state_sequence,
                torch.from_numpy(rtg_sequence).unsqueeze(0).to(device),
                torch.from_numpy(previous_actions).unsqueeze(0).to(device),
                torch.from_numpy(previous_options).unsqueeze(0).to(device),
                torch.from_numpy(time_sequence).unsqueeze(0).to(device),
            )
            option_probabilities = torch.softmax(outputs["option_logits"][0, -1], dim=-1)
            proposed_option = int(option_probabilities.argmax().item())
            if current_option != 0:
                termination_probability = torch.sigmoid(
                    outputs["all_termination_logits"][0, -1, current_option]
                ).item()
                chosen_option = proposed_option if termination_probability >= 0.5 else current_option
            else:
                termination_probability = 1.0
                chosen_option = proposed_option
            action_logits = outputs["all_action_logits"][0, -1, chosen_option].clone()
            counterfactual = world_model.imagine_action_returns(
                state_sequence[:, -1], horizon=imagination_horizon, discount=discount
            )[0]
            reward_scale = counterfactual.std().clamp_min(1e-6)
            action_logits = action_logits + rerank_weight * (counterfactual - counterfactual.mean()) / reward_scale
            action_probabilities = torch.softmax(action_logits, dim=-1)
            chosen_action = int(action_probabilities.argmax().item())
            directional_confidence = float(action_probabilities[chosen_action].item())
            previous_action = action_history[-1] if action_history else 0
            if chosen_action != 0 and directional_confidence < probability_threshold:
                chosen_action = 0
                chosen_option = 0
            is_new_entry = chosen_action != 0 and chosen_action != previous_action
            if is_new_entry and step - last_entry_step < min_gap_steps:
                chosen_action = previous_action
                chosen_option = current_option if chosen_action != 0 else 0
                is_new_entry = False
            if is_new_entry:
                last_entry_step = step
            next_return = 0.0
            if step + 1 < len(day):
                next_return = float(np.log(max(close[step + 1], 1e-9) / max(close[step], 1e-9)))
            position = float(ACTION_TO_POSITION[chosen_action])
            previous_position = float(ACTION_TO_POSITION[previous_action])
            reward = (
                position * next_return
                - abs(position - previous_position) * turnover
                - abs(position) * holding
                - drawdown_penalty * max(-position * next_return, 0.0) ** 2
            )
            if step == len(day) - 1:
                reward -= abs(position) * turnover
            label = int(day.loc[step, "trend_label"]) if pd.notna(day.loc[step, "trend_label"]) else 0
            horizon_return = float(day.loc[step, "future_return"]) if pd.notna(day.loc[step, "future_return"]) else 0.0
            entry_net_return = position * horizon_return - 2.0 * abs(position) * turnover
            success = bool(
                is_new_entry
                and entry_net_return > 0
                and ((chosen_action == 1 and label == 1) or (chosen_action == 2 and label == -1))
            )
            record = {
                "timestamp": day.loc[step, "timestamp"],
                "session_date": day.loc[step, "session_date"],
                "action": chosen_action,
                "action_name": ACTION_NAMES[chosen_action],
                "option": chosen_option,
                "option_name": OPTION_NAMES[chosen_option],
                "action_probability": directional_confidence,
                "option_probability": float(option_probabilities[chosen_option].item()),
                "termination_probability": float(termination_probability),
                "predicted_flat_reward": float(counterfactual[0].item()),
                "predicted_long_reward": float(counterfactual[1].item()),
                "predicted_short_reward": float(counterfactual[2].item()),
                "reward": reward,
                "entry_horizon_net_return": entry_net_return if is_new_entry else 0.0,
                "is_entry": bool(is_new_entry),
                "success": success,
                "trend_label": label,
                "trend_quality": day.loc[step, "trend_quality"],
                "evidence_score": day.loc[step].get("evidence_score", 0.0),
            }
            records.append(record)
            action_history.append(chosen_action)
            option_history.append(chosen_option)
            current_option = chosen_option
            rtg_history.append(rtg_history[-1] - reward)
    predictions = pd.DataFrame(records)
    entries = predictions[predictions["is_entry"]]
    wins = int(entries["success"].sum()) if not entries.empty else 0
    decisions = int(len(entries))
    rewards = predictions["reward"].to_numpy(float) if not predictions.empty else np.zeros(0)
    cumulative = np.cumsum(rewards)
    running_max = np.maximum.accumulate(cumulative) if len(cumulative) else np.zeros(0)
    max_drawdown = float(np.min(cumulative - running_max)) if len(cumulative) else 0.0
    positive = float(rewards[rewards > 0].sum())
    negative = float(-rewards[rewards < 0].sum())
    metrics = {
        "decisions": float(decisions),
        "wins": float(wins),
        "hit_rate": float(wins / decisions) if decisions else 0.0,
        "wilson_lower": wilson_lower_bound(wins, decisions),
        "total_reward": float(rewards.sum()),
        "average_bar_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "profit_factor": float(positive / negative) if negative > 0 else (float("inf") if positive > 0 else 0.0),
        "max_drawdown_log": max_drawdown,
        "exposure": float((predictions["action"] != 0).mean()) if not predictions.empty else 0.0,
    }
    return predictions, metrics


def _select_threshold(
    model: HierarchicalDecisionTransformer,
    world_model: ExogenousWorldModel,
    validation: pd.DataFrame,
    scaler: RobustFeatureScaler,
    config: StrategyConfig,
    desired_return: float,
    device: torch.device,
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    minimum = int(config.get("backtest", "min_validation_decisions", 20))
    for threshold in config.get("backtest", "probability_thresholds", [0.5, 0.6, 0.7]):
        _, metrics = rollout_policy(
            model,
            world_model,
            validation,
            scaler,
            config,
            desired_return=desired_return,
            probability_threshold=float(threshold),
            device=device,
        )
        row = {"threshold": float(threshold), **metrics}
        rows.append(row)
    eligible = [row for row in rows if row["decisions"] >= minimum and row["total_reward"] > 0]
    candidates = eligible or [row for row in rows if row["decisions"] > 0] or rows
    best = max(candidates, key=lambda row: (row["wilson_lower"], row["total_reward"]))
    return float(best["threshold"]), rows


def train_fold(
    trajectories: pd.DataFrame,
    config: StrategyConfig,
    split: TimeSplit,
    output_dir: str | Path,
) -> dict[str, Any]:
    seed = int(config.get("backtest", "random_seed", 42)) + split.fold
    _seed_everything(seed)
    device = _device(str(config.get("model", "device", "auto")))
    train = slice_dates(trajectories, split.train_dates)
    validation = slice_dates(trajectories, split.validation_dates)
    test = slice_dates(trajectories, split.test_dates)
    columns = feature_columns(train)
    scaler = RobustFeatureScaler(columns).fit(train)
    train_states = scaler.transform(train)
    validation_states = scaler.transform(validation)
    sequence_length = int(config.get("model", "sequence_length", 24))
    train_sequences = TrajectorySequenceIndex(train, train_states, sequence_length)
    validation_sequences = TrajectorySequenceIndex(validation, validation_states, sequence_length)
    train_transitions = TransitionIndex(train, train_states)
    validation_transitions = TransitionIndex(validation, validation_states)
    if len(train_sequences) == 0 or len(validation_sequences) == 0:
        raise ValueError("Not enough within-session sequence samples for this fold")
    batch_size = int(config.get("model", "batch_size", 256))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_sequence_loader = DataLoader(TorchSequenceDataset(train_sequences), shuffle=True, **loader_kwargs)
    validation_sequence_loader = DataLoader(
        TorchSequenceDataset(validation_sequences), shuffle=False, **loader_kwargs
    )
    train_transition_loader = DataLoader(
        TorchTransitionDataset(train_transitions), shuffle=True, **loader_kwargs
    )
    validation_transition_loader = DataLoader(
        TorchTransitionDataset(validation_transitions), shuffle=False, **loader_kwargs
    )
    hidden = int(config.get("model", "hidden_dim", 96))
    dimensions = ModelDimensions(
        state_dim=len(columns),
        hidden_dim=hidden,
        n_actions=len(ACTION_NAMES),
        n_options=len(OPTION_NAMES),
        attention_heads=int(config.get("model", "attention_heads", 4)),
        transformer_layers=int(config.get("model", "transformer_layers", 2)),
        dropout=float(config.get("model", "dropout", 0.15)),
    )
    world_model = ExogenousWorldModel(
        state_dim=len(columns), latent_dim=max(16, hidden // 2), n_actions=len(ACTION_NAMES), hidden_dim=hidden
    )
    world_model, world_history = _train_world_model(
        world_model, train_transition_loader, validation_transition_loader, config, device
    )
    model = HierarchicalDecisionTransformer(**dimensions.as_dict())
    model, model_history = _train_decision_transformer(
        model, train_sequence_loader, validation_sequence_loader, config, device
    )
    trajectory_returns = train.groupby("trajectory_id")["trajectory_return"].first()
    desired_return = float(
        trajectory_returns.quantile(float(config.get("rl", "target_return_quantile", 0.85)))
    )
    threshold, threshold_search = _select_threshold(
        model, world_model, validation, scaler, config, desired_return, device
    )
    predictions, test_metrics = rollout_policy(
        model,
        world_model,
        test,
        scaler,
        config,
        desired_return=desired_return,
        probability_threshold=threshold,
        device=device,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    scaler.save(target)
    torch.save(
        {
            "decision_transformer": model.state_dict(),
            "world_model": world_model.state_dict(),
            "dimensions": dimensions.as_dict(),
            "world_latent_dim": max(16, hidden // 2),
            "desired_return": desired_return,
            "probability_threshold": threshold,
            "imagination_horizon": int(config.get("rl", "imagination_horizon", 12)),
            "world_model_rerank_weight": float(
                config.get("rl", "world_model_rerank_weight", 0.20)
            ),
        },
        target / "checkpoint.pt",
    )
    predictions.to_csv(target / "test_predictions.csv", index=False)
    metadata: dict[str, Any] = {
        "fold": split.fold,
        "device": str(device),
        "train_start": str(split.train_dates[0]),
        "train_end": str(split.train_dates[-1]),
        "validation_start": str(split.validation_dates[0]),
        "validation_end": str(split.validation_dates[-1]),
        "test_start": str(split.test_dates[0]),
        "test_end": str(split.test_dates[-1]),
        "desired_return": desired_return,
        "probability_threshold": threshold,
        "threshold_search": threshold_search,
        "test_metrics": test_metrics,
        "world_history": world_history,
        "model_history": model_history,
        "feature_count": len(columns),
    }
    (target / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    return metadata


def train_walk_forward(
    config: StrategyConfig,
    *,
    trajectories: pd.DataFrame | None = None,
    max_folds: int | None = None,
) -> list[dict[str, Any]]:
    if trajectories is None:
        path = config.data_dir / "processed" / "trajectories.csv"
        trajectories = pd.read_csv(path)
        trajectories["timestamp"] = pd.to_datetime(
            trajectories["timestamp"], utc=True, errors="coerce", format="mixed"
        )
    splits = walk_forward_splits(
        trajectories,
        train_days=int(config.get("backtest", "train_days", 756)),
        validation_days=int(config.get("backtest", "validation_days", 126)),
        test_days=int(config.get("backtest", "test_days", 63)),
        step_days=int(config.get("backtest", "step_days", 63)),
    )
    if max_folds is not None:
        splits = splits[-max(1, int(max_folds)) :]
    results: list[dict[str, Any]] = []
    for split in splits:
        fold_dir = config.reports_dir / f"fold_{split.fold:03d}"
        results.append(train_fold(trajectories, config, split, fold_dir))
    summary = {
        "folds": len(results),
        "total_decisions": float(sum(row["test_metrics"]["decisions"] for row in results)),
        "total_wins": float(sum(row["test_metrics"]["wins"] for row in results)),
        "total_reward": float(sum(row["test_metrics"]["total_reward"] for row in results)),
    }
    summary["aggregate_hit_rate"] = (
        summary["total_wins"] / summary["total_decisions"]
        if summary["total_decisions"]
        else 0.0
    )
    summary["aggregate_wilson_lower"] = wilson_lower_bound(
        int(summary["total_wins"]), int(summary["total_decisions"])
    )
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.reports_dir / "walk_forward_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results
