"""Structure-aware attack model for Interpose PUFs."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..common import (
    AgentTrainer as _BaseAgentTrainer,
    TorchPUFModel,
    _make_hidden_stack,
)

STAGE_NAME = "interpose_structure_aware"
LOGISTIC_STAGE_NAME = "interpose_structure_aware_logistic_regression"
STAGED_SPLIT_STAGE_NAME = "interpose_staged_split_logistic_regression"

_UPSTREAM_TEMPERATURE_START = 2.0
_UPSTREAM_TEMPERATURE_MIN = 0.25
_UPSTREAM_TEMPERATURE_DECAY = 0.97


def _att_torch(challenges: Any) -> Any:
    import torch

    return torch.flip(torch.cumprod(torch.flip(challenges, dims=[-1]), dim=-1), dims=[-1])


def _interpose_temperature(epoch: int) -> float:
    return max(
        _UPSTREAM_TEMPERATURE_MIN,
        _UPSTREAM_TEMPERATURE_START * _UPSTREAM_TEMPERATURE_DECAY**epoch,
    )


class InterposeCore:
    """Learn the upstream Arbiter and downstream XOR delay weights."""

    def __new__(
        cls,
        n: int,
        down_chain_count: int,
        interpose_pos: int,
        seed: int,
        hidden_layers: int = 1,
        width: int | None = None,
        response_bits: int = 1,
    ) -> Any:
        import torch
        from torch import nn
        from .xor_apuf import _ParityProductSTE

        if response_bits < 1 or down_chain_count < response_bits:
            raise ValueError(
                "The downstream chain count must be at least the positive response bit count."
            )
        base_group_size, remainder = divmod(down_chain_count, response_bits)
        group_sizes = tuple(
            base_group_size + (group_index < remainder)
            for group_index in range(response_bits)
        )
        torch.manual_seed(seed)
        hidden_width = max(16, 2 * n) if width is None else width

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up_hidden, up_hidden_size = _make_hidden_stack(
                    n, hidden_layers, hidden_width
                )
                self.down_hidden, down_hidden_size = _make_hidden_stack(
                    n + 1, hidden_layers, hidden_width
                )
                self.up_weights = nn.Linear(up_hidden_size, 1)
                self.down_weights = nn.Linear(down_hidden_size, down_chain_count)
                self.interpose_pos = interpose_pos
                self.group_sizes = group_sizes
                self.interpose_temperature = _interpose_temperature(0)

            def _score_downstream(
                self, challenges: Any, interpose_bit: Any
            ) -> Any:
                down_challenges = torch.cat((
                    challenges[:, :self.interpose_pos],
                    interpose_bit,
                    challenges[:, self.interpose_pos:],
                ), dim=1)
                down_features = _att_torch(down_challenges)
                chain_logits = self.down_weights(
                    self.down_hidden(down_features)
                )
                parity_scores = []
                start = 0
                for group_size in self.group_sizes:
                    group = chain_logits[:, start:start + group_size]
                    parity_scores.append(_ParityProductSTE.apply(group))
                    start += group_size
                return torch.tanh(torch.cat(parity_scores, dim=1))

            def forward(self, challenges: Any) -> Any:
                up_features = _att_torch(challenges)
                upstream_delay = self.up_weights(self.up_hidden(up_features))
                if not self.training:
                    interpose_bit = torch.where(
                        upstream_delay >= 0,
                        torch.ones_like(upstream_delay),
                        -torch.ones_like(upstream_delay),
                    )
                    return self._score_downstream(challenges, interpose_bit)

                positive_bits = torch.ones_like(upstream_delay)
                negative_bits = -positive_bits
                both_challenges = torch.cat((challenges, challenges), dim=0)
                both_bits = torch.cat((positive_bits, negative_bits), dim=0)
                both_scores = self._score_downstream(both_challenges, both_bits)
                positive_scores, negative_scores = both_scores.chunk(2, dim=0)
                upstream_probability = torch.sigmoid(
                    upstream_delay / self.interpose_temperature
                )
                return (
                    upstream_probability * positive_scores
                    + (1 - upstream_probability) * negative_scores
                )

        return _Module()


class AgentTrainer:
    """Train the Interpose up/down weights jointly from raw challenge-response data."""

    staged_training = False

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        hidden_layers: int,
        width: int,
        seed: int,
        batch_size: int,
        learning_rate: float,
        device: Any,
        down_chain_count: int,
        interpose_pos: int,
    ):
        import torch

        torch.manual_seed(seed)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(seed)
        self.model = InterposeCore(
            challenges.shape[1],
            down_chain_count,
            interpose_pos,
            seed,
            hidden_layers,
            width,
            response_bits,
        ).to(device)
        if self.staged_training:
            self.up_optimizer = torch.optim.Adam(
                self.model.up_weights.parameters(),
                lr=learning_rate,
                foreach=False,
            )
            self.down_optimizer = torch.optim.Adam(
                self.model.down_weights.parameters(),
                lr=learning_rate,
                foreach=False,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, foreach=False
            )
        response_matrix = np.asarray(responses).reshape(len(responses), response_bits)
        self.training_inputs = torch.as_tensor(challenges, dtype=torch.float32, device=device)
        self.training_labels = torch.as_tensor(
            (1 - response_matrix) / 2, dtype=torch.float32, device=device
        )
        self.loss_function = torch.nn.BCELoss()
        self.batch_size = batch_size
        self.epochs_trained = 0

    def _batch_loss(self, indices: Any) -> Any:
        probabilities = (1 - self.model(self.training_inputs[indices])) / 2
        return self.loss_function(
            probabilities.clamp(1e-7, 1 - 1e-7),
            self.training_labels[indices],
        )

    def train_epoch(self) -> None:
        import torch

        self.model.train()
        self.model.interpose_temperature = _interpose_temperature(
            self.epochs_trained
        )
        order = torch.randperm(len(self.training_inputs), device=self.training_inputs.device)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._batch_loss(indices)
            loss.backward()
            self.optimizer.step()
        self.epochs_trained += 1


class LogisticAgentTrainer(AgentTrainer):
    """Fit a zero-hidden-layer Interpose model with linear Arbiter scores."""

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        hidden_layers: int,
        width: int,
        seed: int,
        batch_size: int,
        learning_rate: float,
        device: Any,
        down_chain_count: int,
        interpose_pos: int,
    ):
        if hidden_layers != 0:
            raise ValueError("The Interpose logistic model must have zero hidden layers.")
        super().__init__(
            challenges,
            responses,
            response_bits,
            0,
            width,
            seed,
            batch_size,
            learning_rate,
            device,
            down_chain_count,
            interpose_pos,
        )


class StagedSplitAgentTrainer(LogisticAgentTrainer):
    """Alternate logistic fitting of the downstream and upstream PUF stages."""

    staged_training = True

    def _step_stage(
        self,
        indices: Any,
        optimizer: Any,
        frozen_module: Any,
    ) -> None:
        for parameter in frozen_module.parameters():
            parameter.requires_grad_(False)
        optimizer.zero_grad(set_to_none=True)
        loss = self._batch_loss(indices)
        loss.backward()
        optimizer.step()
        for parameter in frozen_module.parameters():
            parameter.requires_grad_(True)

    def train_epoch(self) -> None:
        import torch

        self.model.train()
        self.model.interpose_temperature = _interpose_temperature(
            self.epochs_trained
        )
        order = torch.randperm(len(self.training_inputs), device=self.training_inputs.device)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self._step_stage(
                indices,
                self.down_optimizer,
                self.model.up_weights,
            )
            self._step_stage(
                indices,
                self.up_optimizer,
                self.model.down_weights,
            )
        self.epochs_trained += 1


class InterposeTorchPUFModel(TorchPUFModel):
    """Use the shared bipolar-output adapter for Interpose predictions."""


__all__ = [
    "AgentTrainer",
    "InterposeCore",
    "InterposeTorchPUFModel",
    "LogisticAgentTrainer",
    "StagedSplitAgentTrainer",
    "STAGE_NAME",
    "LOGISTIC_STAGE_NAME",
    "STAGED_SPLIT_STAGE_NAME",
]