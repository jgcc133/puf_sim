"""Fully connected and bidirectional LSTM attack for nonlinear Arbiter variants."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pypuf.simulation import Simulation

STAGE_NAME = "fc_bilstm_sequence"
FF_APUF_STAGE_NAME = "ff_apuf_structure_aware_fc_bilstm"


def _build_model(
    challenge_length: int,
    response_bits: int,
    hidden_size: int,
    depth: int,
    seed: int,
    feed_forward_loop: tuple[int, int] | None = None,
) -> Any:
    import torch
    from torch import nn

    if feed_forward_loop is not None:
        return _build_feed_forward_model(
            challenge_length,
            response_bits,
            hidden_size,
            depth,
            seed,
            feed_forward_loop,
        )

    torch.manual_seed(seed)

    class _FCBiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(1, hidden_size)
            self.input_activation = nn.Tanh()
            self.recurrent = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=depth,
                batch_first=True,
                bidirectional=True,
            )
            self.output = nn.Linear(2 * hidden_size, response_bits)
            self.challenge_length = challenge_length

        def forward(self, challenge_sequence: Any) -> Any:
            projected = self.input_activation(
                self.input_projection(challenge_sequence)
            )
            _, (hidden_state, _) = self.recurrent(projected)
            final_state = torch.cat(
                (hidden_state[-2], hidden_state[-1]),
                dim=1,
            )
            return self.output(final_state)

    return _FCBiLSTM()


def _build_feed_forward_model(
    challenge_length: int,
    response_bits: int,
    hidden_size: int,
    depth: int,
    seed: int,
    feed_forward_loop: tuple[int, int],
) -> Any:
    import torch
    from torch import nn

    from .arbiter.feed_forward import _SignSTE, _att_torch

    loop_start, loop_stop = feed_forward_loop
    if not 0 < loop_start < loop_stop <= challenge_length:
        raise ValueError(
            "Feed-forward loop positions must satisfy "
            "0 < arbiter_position < feed_position <= challenge_length."
        )
    torch.manual_seed(seed)

    class _SequenceEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(1, hidden_size)
            self.recurrent = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=depth,
                batch_first=True,
                bidirectional=True,
            )

        def forward(self, sequence: Any) -> Any:
            projected = torch.tanh(self.projection(sequence.unsqueeze(-1)))
            _, (hidden_state, _) = self.recurrent(projected)
            return torch.cat(
                (hidden_state[-2], hidden_state[-1]),
                dim=1,
            )

    class _FeedForwardFCBiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loop_start = loop_start
            self.loop_stop = loop_stop
            self.encoder_a = _SequenceEncoder()
            self.encoder_b = _SequenceEncoder()
            self.delay_a = nn.Linear(2 * hidden_size, response_bits, bias=False)
            self.delay_b = nn.Linear(2 * hidden_size, response_bits, bias=False)

        def forward(self, challenge_sequence: Any) -> Any:
            challenges = challenge_sequence.squeeze(-1)
            upstream_features = _att_torch(challenges[:, :self.loop_start])
            delay_a = self.delay_a(self.encoder_a(upstream_features))
            virtual_bits = _SignSTE.apply(delay_a)
            outputs = []
            for bit_index in range(response_bits):
                downstream_challenges = torch.cat(
                    (
                        challenges[:, self.loop_start:self.loop_stop],
                        virtual_bits[:, bit_index:bit_index + 1],
                        challenges[:, self.loop_stop:],
                    ),
                    dim=1,
                )
                gate = torch.prod(
                    downstream_challenges,
                    dim=1,
                    keepdim=True,
                )
                downstream_features = _att_torch(downstream_challenges)
                delay_b = self.delay_b(
                    self.encoder_b(downstream_features)
                )[:, bit_index:bit_index + 1]
                outputs.append(
                    delay_a[:, bit_index:bit_index + 1] * gate + delay_b
                )
            return torch.cat(outputs, dim=1)

    return _FeedForwardFCBiLSTM()


class AgentTrainer:
    """Train a direct challenge-sequence classifier with BCE-with-logits loss."""

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
        feed_forward_loop: tuple[int, int] | None = None,
    ):
        import torch
        from torch import nn

        challenge_matrix = np.asarray(challenges, dtype=np.float32)
        response_matrix = np.asarray(responses).reshape(len(challenge_matrix), -1)
        if challenge_matrix.ndim != 2 or challenge_matrix.shape[1] < 1:
            raise ValueError("Challenges must be a non-empty two-dimensional matrix.")
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} response columns; "
                f"received {response_matrix.shape[1]}."
            )
        if not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("FC-BiLSTM responses must be bipolar (-1, +1).")
        if hidden_layers < 1 or width < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError(
                "LSTM depth, width, batch size, and learning rate must be positive."
            )

        torch.manual_seed(seed)
        if getattr(device, "type", None) == "xpu":
            torch.xpu.manual_seed_all(seed)
        self.model = _build_model(
            challenge_matrix.shape[1],
            response_bits,
            width,
            hidden_layers,
            seed,
            feed_forward_loop,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            foreach=False,
        )
        self.loss_function = nn.BCEWithLogitsLoss()
        self.training_inputs = torch.as_tensor(
            challenge_matrix[:, :, np.newaxis],
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


class TorchFCBiLSTMModel(Simulation):
    """Use an ensemble of FC-BiLSTM models to predict bipolar PUF responses."""

    def __init__(
        self,
        models: Sequence[Any],
        challenge_length: int,
        response_bits: int,
        device: Any,
        inference_batch_size: int = 64,
    ):
        super().__init__()
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive.")
        self.models = list(models)
        self._challenge_length = challenge_length
        self._response_bits = response_bits
        self.device = device
        self.inference_batch_size = inference_batch_size

    @property
    def challenge_length(self) -> int:
        return self._challenge_length

    @property
    def response_length(self) -> int:
        return self._response_bits

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        import torch

        challenge_matrix = np.asarray(challenges, dtype=np.float32)
        if (
            challenge_matrix.ndim != 2
            or challenge_matrix.shape[1] != self._challenge_length
        ):
            raise ValueError(
                f"Challenges must have shape (samples, {self._challenge_length})."
            )
        if len(challenge_matrix) == 0:
            return np.empty((0, self._response_bits), dtype=np.int8)
        for model in self.models:
            model.eval()
        response_batches = []
        for start in range(0, len(challenge_matrix), self.inference_batch_size):
            stop = min(start + self.inference_batch_size, len(challenge_matrix))
            inputs = torch.as_tensor(
                challenge_matrix[start:stop, :, np.newaxis],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                logits = torch.stack(
                    [model(inputs) for model in self.models]
                ).mean(dim=0)
                responses = torch.where(logits >= 0, 1, -1).to(torch.int8)
            response_batches.append(responses.cpu().numpy())
        return np.concatenate(response_batches, axis=0)


__all__ = ["AgentTrainer", "TorchFCBiLSTMModel", "STAGE_NAME"]
