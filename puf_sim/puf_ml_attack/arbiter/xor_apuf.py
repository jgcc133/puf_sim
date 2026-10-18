"""Parity attack model and trainer for XOR Arbiter PUFs."""
from __future__ import annotations

from typing import Any, Callable, ClassVar

import numpy as np
from pypuf.simulation import ArbiterPUF

from ..common import TorchPUFModel, _make_hidden_stack

STAGE_NAME = "xor_arbiter_parity_ensemble"
LOGISTIC_STAGE_NAME = "xor_arbiter_per_bit_logistic_regression"
QUADRATIC_STAGE_NAME = "xor_arbiter_pairwise_quadratic"
RANK_TWO_STAGE_NAME = "xor_arbiter_rank_two_factorized"


def feature_map(challenges: np.ndarray) -> np.ndarray:
    """Apply the Arbiter Threshold Transform used by each XOR chain."""
    return ArbiterPUF.transform_atf(np.asarray(challenges), 1)[:, 0, :].copy()


class _ParityProductSTE:
    """Use soft parity scores with a sign-aware straight-through gradient."""

    @staticmethod
    def apply(chain_logits: Any) -> Any:
        import torch

        class _Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, logits: Any) -> Any:
                ctx.save_for_backward(logits)
                return torch.prod(torch.tanh(logits), dim=-1, keepdim=True)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> Any:
                (logits,) = ctx.saved_tensors
                chain_signs = torch.where(
                    logits >= 0,
                    torch.ones_like(logits),
                    -torch.ones_like(logits),
                )
                parity_sign = torch.prod(chain_signs, dim=-1, keepdim=True)
                other_chain_signs = parity_sign * chain_signs
                surrogate_slope = torch.clamp(
                    1 - torch.tanh(logits).square(),
                    min=0.05,
                )
                # Prevent the parity product from shrinking gradients across chains.
                return grad_output * other_chain_signs * surrogate_slope

        return _Function.apply(chain_logits)


class _XorParityModel:
    """Predict XOR outputs by grouping learned Arbiter-chain scores."""

    def __new__(
        cls,
        feature_count: int,
        chain_count: int,
        seed: int,
        response_bits: int = 1,
        hidden_layers: int = 1,
        width: int | None = None,
    ) -> Any:
        import torch
        from torch import nn

        if response_bits < 1 or chain_count < response_bits:
            raise ValueError("chain_count must be at least the positive response_bits.")
        hidden_width = max(16, 2 * feature_count) if width is None else width
        base_group_size, extra_chains = divmod(chain_count, response_bits)
        group_sizes = tuple(
            base_group_size + (group_index < extra_chains)
            for group_index in range(response_bits)
        )
        torch.manual_seed(seed)

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_stack, hidden_size = _make_hidden_stack(
                    feature_count,
                    hidden_layers,
                    hidden_width,
                )
                self.chain_logits = nn.Linear(hidden_size, chain_count)
                self.linear_skip = nn.Linear(feature_count, chain_count)
                nn.init.zeros_(self.linear_skip.weight)
                nn.init.zeros_(self.linear_skip.bias)
                self.group_sizes = group_sizes

            def forward(self, features: Any) -> Any:
                chain_logits = (
                    self.chain_logits(self.hidden_stack(features))
                    + self.linear_skip(features)
                )
                if len(set(self.group_sizes)) == 1:
                    grouped_logits = chain_logits.reshape(
                        len(features),
                        response_bits,
                        self.group_sizes[0],
                    )
                    return _ParityProductSTE.apply(grouped_logits).squeeze(-1)

                output_scores = []
                start = 0
                for group_size in self.group_sizes:
                    stop = start + group_size
                    group_logits = chain_logits[:, start:stop]
                    output_scores.append(
                        _ParityProductSTE.apply(group_logits).squeeze(-1)
                    )
                    start = stop
                return torch.stack(output_scores, dim=1)

        return _Module()


class _XorQuadraticModel:
    """Classify two-chain XOR outputs as quadratic functions of ATT features."""

    def __new__(
        cls,
        feature_count: int,
        response_bits: int,
        seed: int,
    ) -> Any:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        pair_rows, pair_columns = torch.triu_indices(
            feature_count,
            feature_count,
        )

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("pair_rows", pair_rows)
                self.register_buffer("pair_columns", pair_columns)
                quadratic_feature_count = feature_count + len(pair_rows)
                self.output = nn.Linear(quadratic_feature_count, response_bits)

            def forward(self, features: Any) -> Any:
                pairwise_features = (
                    features[:, self.pair_rows] * features[:, self.pair_columns]
                )
                polynomial_features = torch.cat(
                    (features, pairwise_features),
                    dim=1,
                )
                return self.output(polynomial_features)

        return _Module()


class _XorRankTwoModel:
    """Predict two-chain XOR responses from a pair of Arbiter delay scores."""

    def __new__(
        cls,
        feature_count: int,
        response_bits: int,
        seed: int,
    ) -> Any:
        import torch
        from torch import nn

        torch.manual_seed(seed)

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.chain_scores = nn.Linear(
                    feature_count,
                    2 * response_bits,
                )

            def forward(self, features: Any) -> Any:
                grouped_scores = self.chain_scores(features).reshape(
                    len(features),
                    response_bits,
                    2,
                )
                return torch.prod(grouped_scores, dim=-1)

        return _Module()


class AgentTrainer:
    """Train component Arbiter chains with a differentiable parity objective."""

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
        chain_count: int,
    ):
        import torch

        torch.manual_seed(seed)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(seed)
        features = np.asarray(feature_map(challenges), dtype=np.float32)
        self.model = _XorParityModel(
            features.shape[1],
            chain_count,
            seed,
            response_bits,
            hidden_layers,
            width,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, foreach=False
        )
        response_matrix = np.asarray(responses).reshape(len(responses), -1)
        if response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} XOR Arbiter PUF response bits, "
                f"received {response_matrix.shape[1]}."
            )
        self.training_inputs = torch.as_tensor(features, dtype=torch.float32, device=device)
        self.training_labels = torch.as_tensor(
            response_matrix, dtype=torch.float32, device=device
        )
        self.batch_size = batch_size

    def train_epoch(self) -> None:
        import torch

        self.model.train()
        order = torch.randperm(len(self.training_inputs), device=self.training_inputs.device)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            scores = self.model(self.training_inputs[indices])
            loss = 1.0 - torch.mean(self.training_labels[indices] * scores)
            loss.backward()
            self.optimizer.step()


class LogisticAgentTrainer:
    """Train one linear logistic classifier per XOR response bit."""

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        seed: int,
        batch_size: int,
        learning_rate: float,
        device: Any,
    ):
        import torch
        from torch import nn

        challenge_matrix = np.asarray(challenges)
        response_matrix = np.asarray(responses).reshape(len(challenge_matrix), -1)
        if challenge_matrix.ndim != 2 or challenge_matrix.shape[1] < 1:
            raise ValueError("Challenges must be a non-empty two-dimensional matrix.")
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} XOR Arbiter PUF response bits, "
                f"received {response_matrix.shape[1]}."
            )
        if not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("XOR Arbiter PUF responses must be bipolar (-1, +1).")
        if batch_size < 1 or learning_rate <= 0:
            raise ValueError("Batch size and learning rate must be positive.")

        torch.manual_seed(seed)
        if getattr(device, "type", None) == "xpu":
            torch.xpu.manual_seed_all(seed)
        features = np.asarray(feature_map(challenge_matrix), dtype=np.float32)
        self.model = nn.Linear(features.shape[1], response_bits).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            foreach=False,
        )
        self.loss_function = nn.BCEWithLogitsLoss()
        self.training_inputs = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=device,
        )
        self.training_labels = torch.as_tensor(
            (response_matrix + 1) / 2,
            dtype=torch.float32,
            device=device,
        )
        self.batch_size = batch_size

    def train_epoch(self) -> None:
        import torch

        self.model.train()
        order = torch.randperm(
            len(self.training_inputs),
            device=self.training_inputs.device,
        )
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.training_inputs[indices])
            loss = self.loss_function(logits, self.training_labels[indices])
            loss.backward()
            self.optimizer.step()


class _XorBCETrainer:
    """Shared binary-cross-entropy trainer for per-bit XOR classifiers."""

    model_factory: ClassVar[Callable[[int, int, int], Any]]

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        seed: int,
        batch_size: int,
        learning_rate: float,
        device: Any,
    ):
        import torch
        from torch import nn

        challenge_matrix = np.asarray(challenges)
        response_matrix = np.asarray(responses).reshape(len(challenge_matrix), -1)
        if challenge_matrix.ndim != 2 or challenge_matrix.shape[1] < 1:
            raise ValueError("Challenges must be a non-empty two-dimensional matrix.")
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} XOR Arbiter PUF response bits, "
                f"received {response_matrix.shape[1]}."
            )
        if not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("XOR Arbiter PUF responses must be bipolar (-1, +1).")
        if batch_size < 1 or learning_rate <= 0:
            raise ValueError("Batch size and learning rate must be positive.")

        torch.manual_seed(seed)
        if getattr(device, "type", None) == "xpu":
            torch.xpu.manual_seed_all(seed)
        features = np.asarray(feature_map(challenge_matrix), dtype=np.float32)
        self.model = self.model_factory(
            features.shape[1],
            response_bits,
            seed,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            foreach=False,
        )
        self.loss_function = nn.BCEWithLogitsLoss()
        self.training_inputs = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=device,
        )
        self.training_labels = torch.as_tensor(
            (response_matrix + 1) / 2,
            dtype=torch.float32,
            device=device,
        )
        self.batch_size = batch_size

    def train_epoch(self) -> None:
        import torch

        self.model.train()
        order = torch.randperm(
            len(self.training_inputs),
            device=self.training_inputs.device,
        )
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.training_inputs[indices])
            loss = self.loss_function(logits, self.training_labels[indices])
            loss.backward()
            self.optimizer.step()


class QuadraticAgentTrainer(_XorBCETrainer):
    """Train per-bit classifiers over degree-two ATT features."""

    model_factory = _XorQuadraticModel


class RankTwoAgentTrainer(_XorBCETrainer):
    """Train a pair of Arbiter delay scores for each XOR response bit."""

    model_factory = _XorRankTwoModel


class XorTorchPUFModel(TorchPUFModel):
    """Threshold differentiable XOR parity scores to bipolar responses."""

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        return super().eval(feature_map(challenges))


__all__ = [
    "AgentTrainer",
    "LogisticAgentTrainer",
    "QuadraticAgentTrainer",
    "RankTwoAgentTrainer",
    "XorTorchPUFModel",
    "STAGE_NAME",
    "LOGISTIC_STAGE_NAME",
    "QUADRATIC_STAGE_NAME",
    "RANK_TWO_STAGE_NAME",
    "feature_map",
]