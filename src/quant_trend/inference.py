from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .environment import ACTION_NAMES, OPTION_NAMES
from .rl_data import RobustFeatureScaler
from .rl_models import ExogenousWorldModel, HierarchicalDecisionTransformer


@dataclass(frozen=True)
class ResearchSignal:
    timestamp: str
    symbol: str
    action: str
    option: str
    action_probability: float
    option_probability: float
    termination_probability: float
    predicted_flat_reward: float
    predicted_long_reward: float
    predicted_short_reward: float
    evidence_score: float
    model_path: str
    execution_authorized: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class OfflinePolicyRuntime:
    """Loads a research checkpoint and emits signals, never orders."""

    def __init__(self, artifact_dir: str | Path, device: str = "cpu") -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        self.device = torch.device(device)
        self.scaler = RobustFeatureScaler.load(self.artifact_dir)
        checkpoint = torch.load(self.artifact_dir / "checkpoint.pt", map_location=self.device)
        dimensions = dict(checkpoint["dimensions"])
        self.policy = HierarchicalDecisionTransformer(**dimensions).to(self.device)
        self.policy.load_state_dict(checkpoint["decision_transformer"])
        self.policy.eval()
        self.world_model = ExogenousWorldModel(
            state_dim=int(dimensions["state_dim"]),
            latent_dim=int(checkpoint["world_latent_dim"]),
            n_actions=int(dimensions["n_actions"]),
            hidden_dim=int(dimensions["hidden_dim"]),
        ).to(self.device)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.world_model.eval()
        self.desired_return = float(checkpoint["desired_return"])
        self.probability_threshold = float(checkpoint["probability_threshold"])
        self.imagination_horizon = int(checkpoint.get("imagination_horizon", 12))
        self.world_model_rerank_weight = float(
            checkpoint.get("world_model_rerank_weight", 0.20)
        )

    @torch.no_grad()
    def predict_latest(
        self,
        feature_history: pd.DataFrame,
        *,
        symbol: str,
        previous_actions: Sequence[int] | None = None,
        previous_options: Sequence[int] | None = None,
        desired_return_remaining: float | None = None,
        world_model_weight: float | None = None,
    ) -> ResearchSignal:
        if feature_history.empty:
            raise ValueError("feature_history is empty")
        history = feature_history.sort_values("timestamp").copy()
        states = self.scaler.transform(history)
        length = len(history)
        actions = list(previous_actions or [])[-max(0, length - 1) :]
        options = list(previous_options or [])[-max(0, length - 1) :]
        previous_action_tokens = np.asarray(([0] * (length - len(actions))) + actions, dtype=np.int64)
        previous_option_tokens = np.asarray(([0] * (length - len(options))) + options, dtype=np.int64)
        remaining = self.desired_return if desired_return_remaining is None else desired_return_remaining
        returns_to_go = np.full((1, length, 1), float(remaining), dtype=np.float32)
        if "minute_index" in history:
            timesteps = history["minute_index"].to_numpy(dtype=np.int64)
        else:
            timesteps = np.arange(length, dtype=np.int64)
        outputs = self.policy(
            torch.from_numpy(states).unsqueeze(0).to(self.device),
            torch.from_numpy(returns_to_go).to(self.device),
            torch.from_numpy(previous_action_tokens).unsqueeze(0).to(self.device),
            torch.from_numpy(previous_option_tokens).unsqueeze(0).to(self.device),
            torch.from_numpy(timesteps).unsqueeze(0).to(self.device),
        )
        option_probabilities = torch.softmax(outputs["option_logits"][0, -1], dim=-1)
        option = int(option_probabilities.argmax().item())
        termination_probability = float(
            torch.sigmoid(outputs["all_termination_logits"][0, -1, option]).item()
        )
        action_logits = outputs["all_action_logits"][0, -1, option].clone()
        counterfactual = self.world_model.imagine_action_returns(
            torch.from_numpy(states[-1:]).to(self.device),
            horizon=self.imagination_horizon,
            discount=1.0,
        )[0]
        rerank_weight = (
            self.world_model_rerank_weight
            if world_model_weight is None
            else float(world_model_weight)
        )
        action_logits += rerank_weight * (
            counterfactual - counterfactual.mean()
        ) / counterfactual.std().clamp_min(1e-6)
        action_probabilities = torch.softmax(action_logits, dim=-1)
        action = int(action_probabilities.argmax().item())
        probability = float(action_probabilities[action].item())
        if action != 0 and probability < self.probability_threshold:
            action = 0
            option = 0
            probability = float(action_probabilities[0].item())
        latest = history.iloc[-1]
        return ResearchSignal(
            timestamp=str(latest["timestamp"]),
            symbol=symbol.upper(),
            action=ACTION_NAMES[action],
            option=OPTION_NAMES[option],
            action_probability=probability,
            option_probability=float(option_probabilities[option].item()),
            termination_probability=termination_probability,
            predicted_flat_reward=float(counterfactual[0].item()),
            predicted_long_reward=float(counterfactual[1].item()),
            predicted_short_reward=float(counterfactual[2].item()),
            evidence_score=float(latest.get("evidence_score", 0.0)),
            model_path=str(self.artifact_dir),
            execution_authorized=False,
        )
