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


def _att_torch(challenges: Any) -> Any:
    import torch

    return torch.flip(torch.cumprod(torch.flip(challenges, dims=[-1]), dim=-1), dims=[-1])


class _SignSTE:
    @staticmethod
    def apply(values: Any) -> Any:
        import torch

        class _Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                ctx.save_for_backward(inputs)
                return torch.where(inputs >= 0, torch.ones_like(inputs), -torch.ones_like(inputs))

            @staticmethod
            def backward(ctx, gradient):
                (inputs,) = ctx.saved_tensors
                return gradient * (1 - torch.tanh(inputs) ** 2)

        return _Function.apply(values)


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
    ) -> Any:
        import torch
        from torch import nn

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

            def forward(self, challenges: Any) -> Any:
                up_features = _att_torch(challenges)
                interpose_bit = _SignSTE.apply(
                    self.up_weights(self.up_hidden(up_features))
                )
                down_challenges = torch.cat((
                    challenges[:, :self.interpose_pos],
                    interpose_bit,
                    challenges[:, self.interpose_pos:],
                ), dim=1)
                down_features = _att_torch(down_challenges)
                chain_logits = self.down_weights(
                    self.down_hidden(down_features)
                )
                parity_score = torch.prod(torch.tanh(chain_logits), dim=1, keepdim=True)
                return torch.tanh(parity_score)

        return _Module()


class AgentTrainer:
    """Train the Interpose up/down weights jointly from raw challenge-response data."""

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
        ).to(device)
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

    def train_epoch(self) -> None:
        import torch

        self.model.train()
        order = torch.randperm(len(self.training_inputs), device=self.training_inputs.device)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            probabilities = (1 - self.model(self.training_inputs[indices])) / 2
            loss = self.loss_function(
                probabilities.clamp(1e-7, 1 - 1e-7), self.training_labels[indices]
            )
            loss.backward()
            self.optimizer.step()


class InterposeTorchPUFModel(TorchPUFModel):
    """Use the shared bipolar-output adapter for Interpose predictions."""


__all__ = ["AgentTrainer", "InterposeCore", "InterposeTorchPUFModel", "STAGE_NAME"]