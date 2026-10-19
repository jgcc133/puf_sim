"""Shared PyTorch model/trainer building blocks for PUF ML attacks.

Family-specific attacks (see the ``arbiter`` and ``optical`` submodules)
subclass ``TorchPUFModel`` and ``AgentTrainer`` to plug in a feature map,
loss function, and label encoding suited to that PUF family's response type.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from pypuf.io import ChallengeResponseSet
from pypuf.simulation import Simulation


def _make_hidden_stack(
    input_size: int, hidden_layers: int, width: int
) -> tuple[Any, int]:
    if hidden_layers < 0 or (hidden_layers > 0 and width < 1):
        raise ValueError("hidden_layers must be nonnegative and width must be positive.")
    from torch import nn

    layers: list[Any] = []
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(input_size, width), nn.Tanh()])
        input_size = width
    return nn.Sequential(*layers), input_size


class TorchPUFModel(Simulation):
    """Adapt one or more PyTorch classifiers to pypuf's simulation interface."""

    def __init__(
        self,
        models: Sequence[Any],
        n: int,
        response_bits: int,
        device: Any,
        response_thresholds: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.models = list(models)
        self._n = n
        self._response_bits = response_bits
        self.device = device
        if response_thresholds is None:
            self.response_thresholds = None
        else:
            threshold_vector = np.asarray(response_thresholds, dtype=np.float32).reshape(-1)
            if len(threshold_vector) != response_bits:
                raise ValueError(
                    "response_thresholds length must match response_bits."
                )
            self.response_thresholds = threshold_vector

    @property
    def challenge_length(self) -> int:
        return self._n

    @property
    def response_length(self) -> int:
        return self._response_bits

    def _ensemble_logits(self, challenges: np.ndarray) -> Any:
        import torch

        inputs = torch.as_tensor(challenges, dtype=torch.float32, device=self.device)
        for model in self.models:
            model.eval()
        with torch.no_grad():
            return torch.stack([model(inputs) for model in self.models]).mean(dim=0)

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        import torch

        logits = self._ensemble_logits(challenges)
        if self.response_thresholds is None:
            thresholds = torch.zeros(
                (1, self._response_bits),
                dtype=logits.dtype,
                device=logits.device,
            )
        else:
            thresholds = torch.as_tensor(
                self.response_thresholds,
                dtype=logits.dtype,
                device=logits.device,
            ).reshape(1, self._response_bits)
        return torch.where(logits >= thresholds, 1, -1).to(torch.int8).cpu().numpy()


class AgentTrainer:
    """Keep one model and optimizer alive while training it epoch by epoch."""

    @staticmethod
    def feature_map(challenges: np.ndarray) -> np.ndarray:
        return challenges

    @staticmethod
    def build_loss_function() -> Any:
        from torch import nn

        return nn.BCEWithLogitsLoss()

    @staticmethod
    def build_labels(responses: np.ndarray, response_bits: int, device: Any) -> Any:
        import torch

        response_matrix = np.asarray(responses).reshape(len(responses), response_bits)
        return torch.as_tensor((response_matrix + 1) / 2, dtype=torch.float32, device=device)

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
        from torch import nn

        torch.manual_seed(seed)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(seed)
        mapped_challenges = np.asarray(self.feature_map(challenges), dtype=np.float32)
        feature_count = mapped_challenges.shape[1]
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            input_size = feature_count if not layers else width
            layers.extend([nn.Linear(input_size, width), nn.Tanh()])
        input_size = width if hidden_layers else feature_count
        layers.append(nn.Linear(input_size, response_bits))
        self.model = nn.Sequential(*layers).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, foreach=False
        )
        self.loss_function = self.build_loss_function()
        self.training_inputs = torch.as_tensor(
            mapped_challenges, dtype=torch.float32, device=device
        )
        self.training_labels = self.build_labels(responses, response_bits, device)
        self.batch_size = batch_size

    def train_epoch(self) -> None:
        import torch

        self.model.train()
        order = torch.randperm(len(self.training_inputs), device=self.training_inputs.device)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_function(
                self.model(self.training_inputs[indices]), self.training_labels[indices]
            )
            loss.backward()
            self.optimizer.step()


def resolve_device(backend: str, device: Optional[int]) -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to train PUF attack models.") from error

    normalized = backend.strip().lower()
    if normalized not in {"auto", "cpu", "torch_xpu"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'torch_xpu'.")
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    if normalized == "auto":
        normalized = "torch_xpu" if xpu_available else "cpu"
    if normalized == "torch_xpu":
        if not xpu_available:
            raise RuntimeError("torch_xpu requested, but a usable Intel XPU is unavailable.")
        selected_device = 0 if device is None else device
        if selected_device < 0 or selected_device >= torch.xpu.device_count():
            raise ValueError(f"XPU device index {selected_device} is unavailable.")
        return torch.device(f"xpu:{selected_device}")
    if device is not None:
        raise ValueError("device can only be set when backend='torch_xpu'.")
    return torch.device("cpu")


def pair_accuracy(model: Simulation, dataset: ChallengeResponseSet) -> float:
    predicted = np.asarray(model.eval(dataset.challenges))
    expected = np.asarray(dataset.responses).reshape(len(dataset.challenges), -1)
    return float(np.mean(np.all(predicted == expected, axis=1)))


__all__ = ["AgentTrainer", "TorchPUFModel", "pair_accuracy", "resolve_device"]
