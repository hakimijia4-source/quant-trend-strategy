from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HierarchicalDecisionTransformer(nn.Module):
    """Offline hierarchical Decision Transformer with option-conditioned actions.

    The high-level head selects a temporally extended option.  A separate action
    distribution is produced for every option, and a termination head determines
    whether the active option should end.  The causal sequence is conditioned on
    desired return-to-go, point-in-time state, and previous decisions.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        n_actions: int,
        n_options: int,
        attention_heads: int,
        transformer_layers: int,
        dropout: float,
        max_timestep: int = 512,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_actions = int(n_actions)
        self.n_options = int(n_options)
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.return_embedding = nn.Linear(1, hidden_dim)
        self.previous_action_embedding = nn.Embedding(n_actions, hidden_dim)
        self.previous_option_embedding = nn.Embedding(n_options, hidden_dim)
        self.time_embedding = nn.Embedding(max_timestep, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.option_head = nn.Linear(hidden_dim, n_options)
        self.intra_option_head = nn.Linear(hidden_dim, n_options * n_actions)
        self.termination_head = nn.Linear(hidden_dim, n_options)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        states: Tensor,
        returns_to_go: Tensor,
        previous_actions: Tensor,
        previous_options: Tensor,
        timesteps: Tensor,
        *,
        options_for_action: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        _, sequence_length, _ = states.shape
        time_ids = timesteps.clamp(0, self.time_embedding.num_embeddings - 1)
        tokens = (
            self.state_embedding(states)
            + self.return_embedding(returns_to_go)
            + self.previous_action_embedding(previous_actions)
            + self.previous_option_embedding(previous_options)
            + self.time_embedding(time_ids)
        )
        tokens = self.input_norm(tokens)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=states.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        hidden = self.output_norm(hidden)
        option_logits = self.option_head(hidden)
        all_action_logits = self.intra_option_head(hidden).view(
            *hidden.shape[:2], self.n_options, self.n_actions
        )
        all_termination_logits = self.termination_head(hidden)
        if options_for_action is None:
            selected_options = option_logits.argmax(dim=-1)
        else:
            selected_options = options_for_action
        option_index = selected_options[..., None, None].expand(-1, -1, 1, self.n_actions)
        action_logits = all_action_logits.gather(2, option_index).squeeze(2)
        termination_logits = all_termination_logits.gather(
            2, selected_options[..., None]
        ).squeeze(-1)
        return {
            "hidden": hidden,
            "option_logits": option_logits,
            "all_action_logits": all_action_logits,
            "action_logits": action_logits,
            "termination_logits": termination_logits,
            "all_termination_logits": all_termination_logits,
            "value": self.value_head(hidden).squeeze(-1),
            "selected_options": selected_options,
        }


class ExogenousWorldModel(nn.Module):
    """Latent market world model with action-independent market dynamics.

    A small trader's action does not change the consolidated market state.  The
    transition prior therefore predicts the next latent market state without an
    action input.  The portfolio reward head is action-conditioned.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        n_actions: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.n_actions = int(n_actions)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.transition = nn.GRUCell(latent_dim, latent_dim)
        self.prior_head = nn.Linear(latent_dim, latent_dim * 2)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.action_embedding = nn.Embedding(n_actions, latent_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _distribution(parameters: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = parameters.chunk(2, dim=-1)
        return mean, log_std.clamp(-6.0, 2.0)

    @staticmethod
    def _sample(mean: Tensor, log_std: Tensor, deterministic: bool) -> Tensor:
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * log_std.exp()

    def encode(self, state: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self._distribution(self.encoder(state))
        return self._sample(mean, log_std, deterministic), mean, log_std

    def prior(self, latent: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        recurrent = self.transition(latent, latent)
        mean, log_std = self._distribution(self.prior_head(recurrent))
        return self._sample(mean, log_std, deterministic), mean, log_std

    def reward(self, latent: Tensor, next_latent: Tensor, action: Tensor) -> Tensor:
        action_latent = self.action_embedding(action)
        return self.reward_head(torch.cat([latent, next_latent, action_latent], dim=-1)).squeeze(-1)

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> dict[str, Tensor]:
        latent, mean, log_std = self.encode(state, deterministic=deterministic)
        prior_latent, prior_mean, prior_log_std = self.prior(latent, deterministic=deterministic)
        output: dict[str, Tensor] = {
            "latent": latent,
            "mean": mean,
            "log_std": log_std,
            "prior_latent": prior_latent,
            "prior_mean": prior_mean,
            "prior_log_std": prior_log_std,
            "predicted_next_state": self.decoder(prior_latent),
            "predicted_reward": self.reward(latent, prior_latent, action),
            "done_logit": self.done_head(torch.cat([latent, prior_latent], dim=-1)).squeeze(-1),
        }
        if next_state is not None:
            posterior_latent, posterior_mean, posterior_log_std = self.encode(
                next_state, deterministic=deterministic
            )
            output.update(
                {
                    "posterior_latent": posterior_latent,
                    "posterior_mean": posterior_mean,
                    "posterior_log_std": posterior_log_std,
                    "reconstructed_next_state": self.decoder(posterior_latent),
                    "posterior_reward": self.reward(latent, posterior_latent, action),
                }
            )
        return output

    @torch.no_grad()
    def counterfactual_rewards(self, state: Tensor) -> Tensor:
        return self.imagine_action_returns(state, horizon=1, discount=1.0)

    @torch.no_grad()
    def imagine_action_returns(
        self, state: Tensor, *, horizon: int, discount: float = 1.0
    ) -> Tensor:
        latent, _, _ = self.encode(state, deterministic=True)
        totals = torch.zeros(*state.shape[:-1], self.n_actions, device=state.device)
        multiplier = 1.0
        current = latent
        for _ in range(max(1, int(horizon))):
            next_latent, _, _ = self.prior(current, deterministic=True)
            rewards = []
            for action_id in range(self.n_actions):
                action = torch.full(
                    state.shape[:-1], action_id, device=state.device, dtype=torch.long
                )
                rewards.append(self.reward(current, next_latent, action))
            totals = totals + multiplier * torch.stack(rewards, dim=-1)
            current = next_latent
            multiplier *= float(discount)
        return totals


def _normal_kl(
    posterior_mean: Tensor,
    posterior_log_std: Tensor,
    prior_mean: Tensor,
    prior_log_std: Tensor,
) -> Tensor:
    posterior_var = torch.exp(2.0 * posterior_log_std)
    prior_var = torch.exp(2.0 * prior_log_std)
    kl = (
        prior_log_std
        - posterior_log_std
        + (posterior_var + (posterior_mean - prior_mean).pow(2)) / (2.0 * prior_var)
        - 0.5
    )
    return kl.sum(dim=-1)


def hierarchical_dt_loss(
    outputs: dict[str, Tensor],
    *,
    actions: Tensor,
    options: Tensor,
    termination: Tensor,
    returns_to_go: Tensor,
    weights: Tensor,
    mask: Tensor,
    option_weight: float,
    action_weight: float,
    value_weight: float,
    termination_weight: float,
    label_smoothing: float,
) -> tuple[Tensor, dict[str, float]]:
    effective = weights * mask
    denominator = effective.sum().clamp_min(1e-6)
    option_loss = F.cross_entropy(
        outputs["option_logits"].reshape(-1, outputs["option_logits"].shape[-1]),
        options.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(options)
    action_loss = F.cross_entropy(
        outputs["action_logits"].reshape(-1, outputs["action_logits"].shape[-1]),
        actions.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(actions)
    termination_loss = F.binary_cross_entropy_with_logits(
        outputs["termination_logits"], termination, reduction="none"
    )
    value_loss = F.smooth_l1_loss(
        outputs["value"], returns_to_go.squeeze(-1), reduction="none"
    )
    reduced = {
        "option": (option_loss * effective).sum() / denominator,
        "action": (action_loss * effective).sum() / denominator,
        "termination": (termination_loss * effective).sum() / denominator,
        "value": (value_loss * effective).sum() / denominator,
    }
    total = (
        option_weight * reduced["option"]
        + action_weight * reduced["action"]
        + termination_weight * reduced["termination"]
        + value_weight * reduced["value"]
    )
    metrics = {name: float(value.detach().cpu()) for name, value in reduced.items()}
    metrics["total"] = float(total.detach().cpu())
    return total, metrics


def world_model_loss(
    outputs: dict[str, Tensor],
    *,
    next_state: Tensor,
    reward: Tensor,
    done: Tensor,
    kl_weight: float,
    free_nats: float,
) -> tuple[Tensor, dict[str, float]]:
    reconstruction = F.smooth_l1_loss(outputs["reconstructed_next_state"], next_state)
    prior_prediction = F.smooth_l1_loss(outputs["predicted_next_state"], next_state)
    reward_loss = F.smooth_l1_loss(outputs["posterior_reward"], reward)
    done_loss = F.binary_cross_entropy_with_logits(outputs["done_logit"], done)
    kl = _normal_kl(
        outputs["posterior_mean"],
        outputs["posterior_log_std"],
        outputs["prior_mean"],
        outputs["prior_log_std"],
    ).mean()
    kl_objective = torch.clamp(kl, min=float(free_nats))
    total = reconstruction + 0.5 * prior_prediction + reward_loss + 0.1 * done_loss + kl_weight * kl_objective
    metrics = {
        "total": float(total.detach().cpu()),
        "reconstruction": float(reconstruction.detach().cpu()),
        "prior_prediction": float(prior_prediction.detach().cpu()),
        "reward": float(reward_loss.detach().cpu()),
        "done": float(done_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
    }
    return total, metrics


@dataclass(frozen=True)
class ModelDimensions:
    state_dim: int
    hidden_dim: int
    n_actions: int
    n_options: int
    attention_heads: int
    transformer_layers: int
    dropout: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
