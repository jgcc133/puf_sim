"""Structure-aware deep neural attacks for Arbiter PUF variants."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

STAGE_NAME = "structure_aware_deep_neural"
SUPPORTED_FAMILIES = frozenset({
    "ff_apuf",
    "xor_apuf",
    "interpose",
    "permutation",
})


def _att_torch(challenges: Any) -> Any:
    import torch

    return torch.flip(
        torch.cumprod(torch.flip(challenges, dims=[-1]), dim=-1),
        dims=[-1],
    )


class _SignSTE:
    @staticmethod
    def apply(
        values: Any,
        temperature: float = 1.0,
        relaxed_forward: bool = False,
    ) -> Any:
        import torch

        class _Function(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any,
                inputs: Any,
                temperature: float,
                relaxed_forward: bool,
            ) -> Any:
                ctx.save_for_backward(inputs)
                ctx.temperature = temperature
                ctx.relaxed_forward = relaxed_forward
                if relaxed_forward:
                    return torch.tanh(inputs / temperature)
                return torch.where(inputs >= 0, 1.0, -1.0)

            @staticmethod
            def backward(ctx: Any, gradient: Any) -> Any:
                (inputs,) = ctx.saved_tensors
                if ctx.relaxed_forward:
                    temperature = ctx.temperature
                    scaled = torch.tanh(inputs / temperature)
                    slope = ((1 - scaled.square()) / temperature).clamp(min=0.05)
                else:
                    slope = 1 - torch.tanh(inputs).square()
                return gradient * slope, None, None

        return _Function.apply(values, temperature, relaxed_forward)


def _dense_stack(nn: Any, input_size: int, depth: int, width: int) -> Any:
    layers = []
    for _ in range(depth):
        layers.extend((nn.Linear(input_size, width), nn.Tanh()))
        input_size = width
    return nn.Sequential(*layers)


def _group_sizes(chain_count: int, response_bits: int) -> tuple[int, ...]:
    if response_bits < 1 or chain_count < response_bits:
        raise ValueError(
            "The chain count must be at least the positive response bit count."
        )
    base_size, remainder = divmod(chain_count, response_bits)
    return tuple(
        base_size + (index < remainder) for index in range(response_bits)
    )


def _grouped_parity(chain_logits: Any, group_sizes: Sequence[int], ste: Any) -> Any:
    import torch

    output_scores = []
    start = 0
    for group_size in group_sizes:
        group = chain_logits[:, start:start + group_size]
        if group_size == 1:
            output_scores.append(group[:, 0])
        else:
            output_scores.append(ste.apply(group).reshape(len(chain_logits)))
        start += group_size
    return torch.stack(output_scores, dim=1)


def _build_model(
    family_key: str,
    n: int,
    response_bits: int,
    depth: int,
    width: int,
    seed: int,
    chain_count: Optional[int],
    feed_forward_loops: Optional[Sequence[tuple[int, int]]],
    interpose_pos: Optional[int],
    down_chain_count: Optional[int],
    permutation_table: Optional[np.ndarray],
) -> Any:
    import torch
    from torch import nn

    if family_key not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"deep_neural does not support PUF family {family_key!r}."
        )
    if n < 1 or response_bits < 1 or depth < 1 or width < 1:
        raise ValueError("Challenge size, output size, depth, and width must be positive.")
    torch.manual_seed(seed)

    if family_key == "ff_apuf":
        loops = sorted(
            tuple(loop) for loop in (feed_forward_loops or ())
        )
        if not loops:
            raise ValueError("Feed-forward PUF attacks require at least one loop.")
        expanded_length = n + len(loops)
        feed_points = [feed_point for _, feed_point in loops]
        if (
            len(set(feed_points)) != len(feed_points)
            or any(
                arbiter_point < 0
                or arbiter_point >= feed_point
                or feed_point >= expanded_length
                for arbiter_point, feed_point in loops
            )
        ):
            raise ValueError("Feed-forward loop positions are invalid.")
        segment_lengths = []
        cursor = 0
        for arbiter_point, _ in loops:
            segment_lengths.append(arbiter_point - cursor)
            cursor = arbiter_point
        segment_lengths.append(expanded_length - cursor)

        class _SegmentNetwork(nn.Module):
            def __init__(self, segment_length: int) -> None:
                super().__init__()
                self.segment_length = segment_length
                if segment_length:
                    self.linear_weights = nn.Parameter(
                        torch.empty(response_bits, segment_length)
                    )
                    self.linear_biases = nn.Parameter(torch.zeros(response_bits))
                    nn.init.normal_(
                        self.linear_weights, std=segment_length ** -0.5
                    )
                    self.residual_hidden_weights = nn.ParameterList()
                    self.residual_hidden_biases = nn.ParameterList()
                    input_size = segment_length
                    for _ in range(depth):
                        layer_weights = nn.Parameter(
                            torch.empty(response_bits, input_size, width)
                        )
                        nn.init.xavier_uniform_(layer_weights)
                        self.residual_hidden_weights.append(layer_weights)
                        self.residual_hidden_biases.append(
                            nn.Parameter(torch.zeros(response_bits, width))
                        )
                        input_size = width
                    self.residual_weights = nn.Parameter(
                        torch.empty(response_bits, input_size)
                    )
                    self.residual_biases = nn.Parameter(
                        torch.zeros(response_bits)
                    )
                    nn.init.xavier_uniform_(self.residual_weights)
                    self.residual_scale = nn.Parameter(
                        torch.tensor(0.1, dtype=torch.float32)
                    )

            def forward(self, section: Any) -> Any:
                if not self.segment_length:
                    return section.new_zeros((section.shape[0], response_bits))
                features = _att_torch(section)
                linear_scores = (
                    features * self.linear_weights.unsqueeze(0)
                ).sum(dim=2) + self.linear_biases
                hidden = features
                for layer_weights, layer_biases in zip(
                    self.residual_hidden_weights, self.residual_hidden_biases
                ):
                    hidden = torch.tanh(
                        torch.einsum("bri,rij->brj", hidden, layer_weights)
                        + layer_biases.unsqueeze(0)
                    )
                residual_scores = (
                    hidden * self.residual_weights.unsqueeze(0)
                ).sum(dim=2) + self.residual_biases
                return linear_scores + self.residual_scale * residual_scores

        class _FeedForwardDNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loops = tuple(loops)
                self.feed_temperature = nn.Parameter(
                    torch.ones(response_bits, dtype=torch.float32)
                )
                self.segment_networks = nn.ModuleList(
                    _SegmentNetwork(length) for length in segment_lengths
                )

            def forward(self, challenges: Any) -> Any:
                batch_size = challenges.shape[0]
                zero_column = challenges.new_zeros((batch_size, response_bits))
                columns = []
                source_index = 0
                feed_point_set = set(feed_points)
                for expanded_index in range(expanded_length):
                    if expanded_index in feed_point_set:
                        columns.append(zero_column)
                    else:
                        columns.append(
                            challenges[:, source_index:source_index + 1].expand(
                                -1, response_bits
                            )
                        )
                        source_index += 1

                accumulated_delay = challenges.new_zeros(
                    (batch_size, response_bits)
                )
                cursor = 0
                for loop_index, (arbiter_point, feed_point) in enumerate(self.loops):
                    section_columns = columns[cursor:arbiter_point]
                    section = (
                        torch.stack(section_columns, dim=2)
                        if section_columns
                        else challenges.new_empty((batch_size, response_bits, 0))
                    )
                    gate = (
                        section.prod(dim=2)
                        if section.shape[2]
                        else challenges.new_ones((batch_size, response_bits))
                    )
                    accumulated_delay = (
                        accumulated_delay * gate
                        + self.segment_networks[loop_index](section)
                    )
                    temperature = torch.clamp(
                        self.feed_temperature, min=0.25, max=2.5
                    ).unsqueeze(0)
                    columns[feed_point] = _SignSTE.apply(
                        accumulated_delay / temperature
                    )
                    cursor = arbiter_point

                final_columns = columns[cursor:]
                final_section = (
                    torch.stack(final_columns, dim=2)
                    if final_columns
                    else challenges.new_empty((batch_size, response_bits, 0))
                )
                final_gate = (
                    final_section.prod(dim=2)
                    if final_section.shape[2]
                    else challenges.new_ones((batch_size, response_bits))
                )
                return (
                    accumulated_delay * final_gate
                    + self.segment_networks[-1](final_section)
                )

        return _FeedForwardDNN()

    if family_key == "xor_apuf":
        if chain_count is None:
            raise ValueError("XOR PUF attacks require a chain count.")
        group_sizes = _group_sizes(chain_count, response_bits)
        from .xor_apuf import _ParityProductSTE

        class _XorDNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = _dense_stack(nn, n, depth, width)
                self.chain_head = nn.Linear(width, chain_count)
                self.linear_skip = nn.Linear(n, chain_count)
                nn.init.zeros_(self.linear_skip.weight)
                nn.init.zeros_(self.linear_skip.bias)

            def forward(self, challenges: Any) -> Any:
                features = _att_torch(challenges)
                chain_logits = (
                    self.chain_head(self.body(features))
                    + self.linear_skip(features)
                )
                return _grouped_parity(
                    chain_logits,
                    group_sizes,
                    _ParityProductSTE,
                )

        return _XorDNN()

    if family_key == "interpose":
        if down_chain_count is None or down_chain_count < 1:
            raise ValueError("Interpose attacks require a positive downstream chain count.")
        if interpose_pos is None or not 0 <= interpose_pos <= n:
            raise ValueError("Interpose position must be between 0 and challenge length.")
        group_sizes = _group_sizes(down_chain_count, response_bits)
        from .xor_apuf import _ParityProductSTE

        class _InterposeDNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.interpose_temperature = 2.0
                self.up_body = _dense_stack(nn, n, depth, width)
                self.up_head = nn.Linear(width, 1)
                self.down_body = _dense_stack(nn, n + 1, depth, width)
                self.down_head = nn.Linear(width, down_chain_count)

            def _score_downstream(
                self, challenges: Any, interpose_bit: Any
            ) -> Any:
                downstream_challenges = torch.cat(
                    (
                        challenges[:, :interpose_pos],
                        interpose_bit,
                        challenges[:, interpose_pos:],
                    ),
                    dim=1,
                )
                chain_logits = self.down_head(
                    self.down_body(_att_torch(downstream_challenges))
                )
                return _grouped_parity(
                    chain_logits,
                    group_sizes,
                    _ParityProductSTE,
                )

            def forward(self, challenges: Any) -> Any:
                upstream_delay = self.up_head(
                    self.up_body(_att_torch(challenges))
                )
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
                mixed_probability = (
                    upstream_probability * torch.sigmoid(positive_scores)
                    + (1 - upstream_probability) * torch.sigmoid(negative_scores)
                )
                return torch.logit(mixed_probability.clamp(1e-6, 1 - 1e-6))

        return _InterposeDNN()

    if permutation_table is None:
        raise ValueError("Permutation PUF attacks require a permutation table.")
    table = np.asarray(permutation_table, dtype=np.int64)
    if table.ndim != 2 or table.shape[1] != n:
        raise ValueError(
            "permutation_table must have shape (chain_count, challenge_length)."
        )
    if any(not np.array_equal(np.sort(row), np.arange(n)) for row in table):
        raise ValueError("Each permutation table row must be a permutation.")
    chain_count = table.shape[0]
    group_sizes = _group_sizes(chain_count, response_bits)
    from ..permutation import _PermutationParitySTE

    class _PermutationDNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "permutation_table",
                torch.as_tensor(table, dtype=torch.long),
            )
            self.body = _dense_stack(nn, n, depth, width)
            self.chain_weights = nn.Parameter(
                torch.empty(chain_count, width)
            )
            self.chain_biases = nn.Parameter(torch.zeros(chain_count))
            nn.init.normal_(self.chain_weights, std=width ** -0.5)

        def forward(self, challenges: Any) -> Any:
            batch_size = len(challenges)
            permutation_indices = self.permutation_table.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            permuted = torch.gather(
                challenges.unsqueeze(1).expand(-1, chain_count, -1),
                2,
                permutation_indices,
            )
            features = _att_torch(permuted)
            hidden = self.body(features.reshape(-1, n)).reshape(
                batch_size, chain_count, width
            )
            chain_logits = (
                hidden * self.chain_weights.unsqueeze(0)
            ).sum(dim=2) + self.chain_biases
            return _grouped_parity(
                chain_logits,
                group_sizes,
                _PermutationParitySTE,
            )

    return _PermutationDNN()


class AgentTrainer:
    """Train a family-structured deep neural PUF model."""

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
        family_key: str,
        chain_count: Optional[int] = None,
        feed_forward_loops: Optional[Sequence[tuple[int, int]]] = None,
        interpose_pos: Optional[int] = None,
        down_chain_count: Optional[int] = None,
        permutation_table: Optional[np.ndarray] = None,
    ) -> None:
        import torch
        from torch import nn

        challenge_matrix = np.asarray(challenges, dtype=np.float32)
        response_matrix = np.asarray(responses).reshape(len(challenge_matrix), -1)
        if (
            challenge_matrix.ndim != 2
            or challenge_matrix.shape[1] < 1
            or len(challenge_matrix) < 1
        ):
            raise ValueError("Challenges must be a non-empty two-dimensional matrix.")
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} response columns; "
                f"received {response_matrix.shape[1]}."
            )
        if not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("DNN PUF responses must be bipolar (-1, +1).")
        if hidden_layers < 1 or width < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError(
                "DNN depth, width, batch size, and learning rate must be positive."
            )

        torch.manual_seed(seed)
        if getattr(device, "type", None) == "xpu":
            torch.xpu.manual_seed_all(seed)
        self.model = _build_model(
            family_key,
            challenge_matrix.shape[1],
            response_bits,
            hidden_layers,
            width,
            seed,
            chain_count,
            feed_forward_loops,
            interpose_pos,
            down_chain_count,
            permutation_table,
        ).to(device)
        self._family_key = family_key
        self._device_type = getattr(device, "type", None)
        self._ff_l2_weight = (
            1e-5
            if family_key == "ff_apuf" and self._device_type != "xpu"
            else 0.0
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            foreach=False,
        )
        self.loss_function = nn.BCEWithLogitsLoss(reduction="none")
        self.training_inputs = torch.as_tensor(
            challenge_matrix,
            dtype=torch.float32,
            device=device,
        )
        self.training_labels = torch.as_tensor(
            (response_matrix + 1) / 2,
            dtype=torch.float32,
            device=device,
        )
        self.training_sign_labels = torch.as_tensor(
            response_matrix,
            dtype=torch.float32,
            device=device,
        )
        self.use_joint_margin_loss = (
            family_key == "ff_apuf" and response_bits > 1
        )
        self._joint_margin_weight_min = (
            0.02 if self.use_joint_margin_loss else 0.0
        )
        self._joint_margin_weight_max = (
            0.08 if self.use_joint_margin_loss else 0.0
        )
        self._joint_margin_ramp_epochs = 120
        self._weak_margin_target = 0.8
        self._hard_sample_boost = 0.2
        self._epochs_trained = 0
        self._lr_plateau_enabled = family_key in SUPPORTED_FAMILIES
        self._validation_ema: Optional[float] = None
        self._validation_ema_alpha = 0.35
        self._best_validation_ema: Optional[float] = None
        self._epochs_without_ema_improvement = 0
        self._lr_plateau_patience = 8
        self._lr_min_ema_improvement = 0.001
        self._lr_reduce_factor = 0.5
        self._lr_reduce_cooldown_epochs = 4
        self._epochs_since_lr_drop = 0
        self._lr_floor = 1e-4
        self._lr_events: list[dict[str, float]] = []
        self.batch_size = batch_size

    def update_validation_feedback(self, validation_accuracy: float) -> None:
        if not self._lr_plateau_enabled:
            return
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
            latest_lr = float(self.optimizer.param_groups[0]["lr"])
            self._lr_events.append({
                "epoch": float(self._epochs_trained),
                "ema": float(self._validation_ema),
                "lr": latest_lr,
            })
            self._epochs_since_lr_drop = 0
            self._epochs_without_ema_improvement = 0

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def validation_ema(self) -> Optional[float]:
        return self._validation_ema

    def consume_lr_events(self) -> list[dict[str, float]]:
        events = list(self._lr_events)
        self._lr_events.clear()
        return events

    def train_epoch(self) -> None:
        import torch
        import torch.nn.functional as functional
        from torch.nn.utils import clip_grad_norm_

        self.model.train()
        if self._family_key == "interpose":
            from .interpose import _interpose_temperature

            self.model.interpose_temperature = _interpose_temperature(
                self._epochs_trained
            )
        order = torch.randperm(
            len(self.training_inputs),
            device=self.training_inputs.device,
        )
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            scores = self.model(self.training_inputs[indices])
            loss_matrix = self.loss_function(scores, self.training_labels[indices])
            base_loss_per_sample = loss_matrix.mean(dim=1)
            loss = base_loss_per_sample.mean()
            if self.use_joint_margin_loss:
                signed_margins = scores * self.training_sign_labels[indices]
                weakest_margin = signed_margins.min(dim=1).values
                hard_mask = (weakest_margin < self._weak_margin_target).to(
                    dtype=base_loss_per_sample.dtype
                )
                sample_weights = 1.0 + self._hard_sample_boost * hard_mask
                loss = (base_loss_per_sample * sample_weights).mean()
                joint_margin_loss = functional.softplus(-weakest_margin).mean()
                if self._joint_margin_ramp_epochs > 0:
                    ramp_progress = min(
                        1.0, self._epochs_trained / self._joint_margin_ramp_epochs
                    )
                else:
                    ramp_progress = 1.0
                margin_weight = (
                    self._joint_margin_weight_min
                    + (
                        self._joint_margin_weight_max
                        - self._joint_margin_weight_min
                    ) * ramp_progress
                )
                loss = loss + margin_weight * joint_margin_loss
            if self._ff_l2_weight > 0.0:
                l2_penalty = torch.zeros(
                    (), device=scores.device, dtype=scores.dtype
                )
                for parameter in self.model.parameters():
                    l2_penalty = l2_penalty + parameter.square().sum()
                loss = loss + self._ff_l2_weight * l2_penalty
            loss.backward()
            if self._family_key == "ff_apuf":
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        self._epochs_trained += 1


__all__ = ["AgentTrainer", "STAGE_NAME", "SUPPORTED_FAMILIES"]
