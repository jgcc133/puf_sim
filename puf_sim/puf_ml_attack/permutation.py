"""Structure-aware parity attack for Permutation PUFs."""
from __future__ import annotations

from typing import Any

import numpy as np


STAGE_NAME = "permutation_chain_parity"


class _PermutationParitySTE:
    """Use a soft parity score with sign-aware gradients."""

    @staticmethod
    def apply(chain_logits: Any) -> Any:
        import torch

        class _Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, logits: Any) -> Any:
                ctx.save_for_backward(logits)
                return torch.prod(torch.tanh(logits), dim=-1)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> Any:
                (logits,) = ctx.saved_tensors
                chain_signs = torch.where(
                    logits >= 0,
                    torch.ones_like(logits),
                    -torch.ones_like(logits),
                )
                parity_sign = torch.prod(chain_signs, dim=-1, keepdim=True)
                surrogate_slope = torch.clamp(
                    1 - torch.tanh(logits).square(),
                    min=0.05,
                )
                return (
                    grad_output.unsqueeze(-1)
                    * parity_sign
                    * chain_signs
                    * surrogate_slope
                )

        return _Function.apply(chain_logits)


class _PermutationChainModel:
    """Learn a separate Arbiter score for each known challenge permutation."""

    def __new__(
        cls,
        n: int,
        permutation_table: np.ndarray,
        response_bits: int,
        seed: int,
    ) -> Any:
        import torch
        from torch import nn

        table = np.asarray(permutation_table, dtype=np.int64)
        if table.ndim != 2 or table.shape[1] != n or table.shape[0] < response_bits:
            raise ValueError(
                "permutation_table must have shape (chain_count, n), with at "
                "least one chain per response bit."
            )
        if any(not np.array_equal(np.sort(row), np.arange(n)) for row in table):
            raise ValueError("Each row of permutation_table must be a permutation.")
        if response_bits < 1:
            raise ValueError("response_bits must be positive.")

        chain_count = table.shape[0]
        base_group_size, extra_chains = divmod(chain_count, response_bits)
        group_sizes = tuple(
            base_group_size + (group_index < extra_chains)
            for group_index in range(response_bits)
        )
        torch.manual_seed(seed)

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "permutation_table",
                    torch.as_tensor(table, dtype=torch.long),
                )
                self.chain_weights = nn.Parameter(torch.empty(chain_count, n))
                self.chain_biases = nn.Parameter(torch.zeros(chain_count))
                nn.init.normal_(self.chain_weights, std=n ** -0.5)
                self.group_sizes = group_sizes

            def forward(self, challenges: Any) -> Any:
                chain_logits = self.chain_logits(challenges)

                if len(set(self.group_sizes)) == 1:
                    grouped_logits = chain_logits.reshape(
                        len(challenges),
                        response_bits,
                        self.group_sizes[0],
                    )
                    if self.group_sizes[0] == 2:
                        return torch.prod(grouped_logits, dim=2)
                    return _PermutationParitySTE.apply(grouped_logits)

                output_scores = []
                start = 0
                for group_size in self.group_sizes:
                    stop = start + group_size
                    output_scores.append(
                        _PermutationParitySTE.apply(chain_logits[:, start:stop])
                    )
                    start = stop
                return torch.stack(output_scores, dim=1)

            def chain_logits(self, challenges: Any) -> Any:
                batch_size = len(challenges)
                permutation_indices = self.permutation_table.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
                permuted = torch.gather(
                    challenges.unsqueeze(1).expand(-1, chain_count, -1),
                    2,
                    permutation_indices,
                )
                prefix_products = torch.cumprod(permuted, dim=2)
                exclusive_prefix = torch.cat(
                    (
                        torch.ones_like(permuted[:, :, :1]),
                        prefix_products[:, :, :-1],
                    ),
                    dim=2,
                )
                att_features = permuted.prod(dim=2, keepdim=True) * exclusive_prefix
                chain_logits = (
                    (att_features * self.chain_weights.unsqueeze(0)).sum(dim=2)
                    + self.chain_biases
                )
                return chain_logits

        return _Module()


def _spectral_rank_two_initialization(
    challenges: np.ndarray,
    responses: np.ndarray,
    permutation_table: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    challenge_matrix = np.asarray(challenges, dtype=np.float32)
    response_matrix = np.asarray(responses, dtype=np.float32)
    table = np.asarray(permutation_table, dtype=np.int64)
    response_bits = response_matrix.shape[1]
    challenge_length = challenge_matrix.shape[1]
    moments = np.zeros(
        (response_bits, challenge_length, challenge_length),
        dtype=np.float64,
    )

    for start in range(0, len(challenge_matrix), batch_size):
        stop = min(start + batch_size, len(challenge_matrix))
        permuted = np.take(challenge_matrix[start:stop], table, axis=1)
        att_features = np.flip(
            np.cumprod(np.flip(permuted, axis=2), axis=2),
            axis=2,
        )
        left_features = att_features[:, 0::2, :]
        right_features = att_features[:, 1::2, :]
        weighted_right = (
            right_features * response_matrix[start:stop, :, None]
        )
        moments += np.matmul(
            np.transpose(left_features, (1, 2, 0)),
            np.transpose(weighted_right, (1, 0, 2)),
        )

    left, _, right = np.linalg.svd(moments / len(challenge_matrix), full_matrices=False)
    initial_weights = np.empty(
        (2 * response_bits, challenge_length),
        dtype=np.float32,
    )
    initial_weights[0::2] = left[:, :, 0]
    initial_weights[1::2] = right[:, 0, :]
    return initial_weights


class AgentTrainer:
    """Fit per-chain Arbiter weights using the target's fixed permutations."""

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
        permutation_table: np.ndarray,
    ):
        import torch
        from torch import nn

        challenge_matrix = np.asarray(challenges)
        response_matrix = np.asarray(responses).reshape(len(challenge_matrix), -1)
        table = np.asarray(permutation_table)
        if challenge_matrix.ndim != 2 or challenge_matrix.shape[1] < 1:
            raise ValueError("Challenges must be a non-empty two-dimensional matrix.")
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} permutation PUF response bits, "
                f"received {response_matrix.shape[1]}."
            )
        if not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("Permutation PUF responses must be bipolar (-1, +1).")
        if batch_size < 1 or learning_rate <= 0:
            raise ValueError("Batch size and learning rate must be positive.")
        if table.ndim != 2 or table.shape[1] != challenge_matrix.shape[1]:
            raise ValueError(
                "permutation_table must have shape (chain_count, challenge_length)."
            )

        torch.manual_seed(seed)
        if getattr(device, "type", None) == "xpu":
            torch.xpu.manual_seed_all(seed)
        self.model = _PermutationChainModel(
            challenge_matrix.shape[1],
            table,
            response_bits,
            seed,
        ).to(device)
        self.rank_two = all(
            group_size == 2 for group_size in self.model.group_sizes
        )
        if self.rank_two:
            initial_weights = _spectral_rank_two_initialization(
                challenge_matrix,
                response_matrix,
                table,
                batch_size,
            )
            with torch.no_grad():
                self.model.chain_weights.copy_(
                    torch.as_tensor(
                        initial_weights,
                        dtype=torch.float32,
                        device=device,
                    )
                )
        self.loss_function = (
            nn.BCEWithLogitsLoss() if self.rank_two else None
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            foreach=False,
        )
        self.training_inputs = torch.as_tensor(
            challenge_matrix,
            dtype=torch.float32,
            device=device,
        )
        self.training_labels = torch.as_tensor(
            (response_matrix + 1) / 2 if self.rank_two else response_matrix,
            dtype=torch.float32,
            device=device,
        )
        self.training_responses = torch.as_tensor(
            response_matrix,
            dtype=torch.float32,
            device=device,
        )
        self.batch_size = batch_size
        self._epochs_trained = 0
        self._validation_ema: float | None = None
        self._validation_ema_alpha = 0.35
        self._best_validation_ema: float | None = None
        self._epochs_without_ema_improvement = 0
        self._lr_plateau_patience = 8
        self._lr_min_ema_improvement = 0.001
        self._lr_reduce_factor = 0.5
        self._lr_reduce_cooldown_epochs = 4
        self._epochs_since_lr_drop = 0
        self._lr_floor = 1e-4
        self._lr_events: list[dict[str, float]] = []
        if self.rank_two:
            self._pretrain_chain_factors()

    def update_validation_feedback(self, validation_accuracy: float) -> None:
        value = float(validation_accuracy)
        self._epochs_since_lr_drop += 1
        if self._validation_ema is None:
            self._validation_ema = value
            self._best_validation_ema = value
            return

        self._validation_ema = (
            self._validation_ema_alpha * value
            + (1.0 - self._validation_ema_alpha) * self._validation_ema
        )
        if (
            self._best_validation_ema is None
            or self._validation_ema
            > self._best_validation_ema + self._lr_min_ema_improvement
        ):
            self._best_validation_ema = self._validation_ema
            self._epochs_without_ema_improvement = 0
        else:
            self._epochs_without_ema_improvement += 1
        if (
            self._epochs_without_ema_improvement < self._lr_plateau_patience
            or self._epochs_since_lr_drop < self._lr_reduce_cooldown_epochs
        ):
            return

        reduced = False
        for parameter_group in self.optimizer.param_groups:
            current_lr = float(parameter_group["lr"])
            next_lr = max(self._lr_floor, current_lr * self._lr_reduce_factor)
            if next_lr < current_lr:
                parameter_group["lr"] = next_lr
                reduced = True
        if reduced:
            self._lr_events.append({
                "epoch": float(self._epochs_trained),
                "ema": float(self._validation_ema),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            })
            self._epochs_since_lr_drop = 0
            self._epochs_without_ema_improvement = 0

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def validation_ema(self) -> float | None:
        return self._validation_ema

    def consume_lr_events(self) -> list[dict[str, float]]:
        events = list(self._lr_events)
        self._lr_events.clear()
        return events

    def _pretrain_chain_factors(self) -> None:
        import torch
        from torch import nn

        for _ in range(5):
            for target_start in (1, 0):
                target_indices = slice(target_start, None, 2)
                other_indices = slice(1 - target_start, None, 2)
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=0.01,
                    foreach=False,
                )
                for _ in range(5):
                    order = torch.randperm(
                        len(self.training_inputs),
                        device=self.training_inputs.device,
                    )
                    for start in range(0, len(order), self.batch_size):
                        indices = order[start:start + self.batch_size]
                        self.model.train()
                        optimizer.zero_grad(set_to_none=True)
                        chain_logits = self.model.chain_logits(
                            self.training_inputs[indices]
                        )
                        other_signs = torch.where(
                            chain_logits[:, other_indices].detach() >= 0,
                            torch.ones_like(chain_logits[:, other_indices]),
                            -torch.ones_like(chain_logits[:, other_indices]),
                        )
                        pseudo_responses = (
                            self.training_responses[indices] * other_signs
                        )
                        pseudo_labels = (pseudo_responses + 1) / 2
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            chain_logits[:, target_indices],
                            pseudo_labels,
                        )
                        loss.backward()
                        optimizer.step()

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
            scores = self.model(self.training_inputs[indices])
            if self.rank_two:
                loss = self.loss_function(
                    scores,
                    self.training_labels[indices],
                )
            else:
                loss = 1.0 - torch.mean(
                    self.training_labels[indices] * scores
                )
            loss.backward()
            self.optimizer.step()
        self._epochs_trained += 1


__all__ = ["AgentTrainer", "STAGE_NAME", "_PermutationChainModel"]
