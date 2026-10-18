"""Structure-aware attack model for feed-forward Arbiter PUFs."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..common import AgentTrainer as _BaseAgentTrainer, _make_hidden_stack

STAGE_NAME = "ff_apuf_structure_aware"


def _att_torch(challenges: Any) -> Any:
    import torch

    return torch.flip(torch.cumprod(torch.flip(challenges, dims=[-1]), dim=-1), dims=[-1])


class _SignSTE:
    @staticmethod
    def apply(x: Any) -> Any:
        import torch

        class _Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.save_for_backward(value)
                return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

            @staticmethod
            def backward(ctx, grad_output):
                (value,) = ctx.saved_tensors
                return grad_output * (1 - torch.tanh(value) ** 2)

        return _Function.apply(x)


class _FeedForwardCore:
    """Learn the two delay segments and internal sign-gated feed-forward loop."""

    def __new__(
        cls,
        n: int,
        response_bits: int,
        seed: int,
        hidden_layers: int = 1,
        width: int | None = None,
    ) -> Any:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        hidden_width = max(16, 2 * n) if width is None else width
        arbiter_point = max(1, min(n - 1, n // 3))
        feed_point = max(arbiter_point + 1, min(n - 1, (2 * n) // 3))

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.arbiter_point = arbiter_point
                self.feed_point = feed_point
                self.hidden_a, hidden_a_size = _make_hidden_stack(
                    arbiter_point,
                    hidden_layers,
                    hidden_width,
                )
                self.hidden_b, hidden_b_size = _make_hidden_stack(
                    n - arbiter_point + 1,
                    hidden_layers,
                    hidden_width,
                )
                self.weight_a = nn.Linear(hidden_a_size, response_bits, bias=False)
                self.weight_b = nn.Linear(hidden_b_size, response_bits, bias=False)

            def forward(self, challenges: Any) -> Any:
                segment_a = _att_torch(challenges[:, :self.arbiter_point])
                delay_a = self.weight_a(self.hidden_a(segment_a))
                virtual_bit = _SignSTE.apply(delay_a)
                outputs = []
                for bit_index in range(delay_a.shape[1]):
                    segment_b_raw = torch.cat([
                        challenges[:, self.arbiter_point:self.feed_point],
                        virtual_bit[:, bit_index:bit_index + 1],
                        challenges[:, self.feed_point:],
                    ], dim=1)
                    gate = torch.prod(segment_b_raw, dim=1, keepdim=True)
                    down_features = self.hidden_b(_att_torch(segment_b_raw))
                    delay_b = self.weight_b(down_features)[:, bit_index:bit_index + 1]
                    outputs.append(delay_a[:, bit_index:bit_index + 1] * gate + delay_b)
                return torch.cat(outputs, dim=1)

        return _Module()


class AgentTrainer(_BaseAgentTrainer):
    """Train the structure-aware feed-forward model on raw challenges."""

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
    ):
        import torch

        torch.manual_seed(seed)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(seed)
        self.model = _FeedForwardCore(
            challenges.shape[1], response_bits, seed, hidden_layers, width
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, foreach=False
        )
        self.loss_function = self.build_loss_function()
        self.training_inputs = torch.as_tensor(challenges, dtype=torch.float32, device=device)
        self.training_labels = self.build_labels(responses, response_bits, device)
        self.batch_size = batch_size


__all__ = ["AgentTrainer", "STAGE_NAME"]