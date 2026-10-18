"""Feature-mapped support-vector models for PUF response prediction."""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from pypuf.simulation import Simulation

STAGE_NAME = "rbf_feature_mapped_svm"
XOR_STAGE_NAME = "xor_parity_hinge_svm"
RBF_FEATURE_COUNT = 512
XOR_SVM_WEIGHT_DECAY = 1e-3


def _map_challenges(
    challenges: np.ndarray,
    challenge_transform: Optional[Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    mapped = (
        challenges
        if challenge_transform is None
        else challenge_transform(challenges)
    )
    mapped = np.asarray(mapped, dtype=np.float32)
    if (
        mapped.ndim != 2
        or mapped.shape[0] != len(challenges)
        or mapped.shape[1] < 1
    ):
        raise ValueError(
            "Challenge transforms must return a non-empty feature matrix "
            "with one row per challenge."
        )
    return mapped


class FeatureMappedSVM(Simulation):
    """Fit one support-vector model per PUF response bit."""

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        seed: int,
        regression: bool = False,
        challenge_transform: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = None,
    ):
        try:
            from sklearn.kernel_approximation import RBFSampler
            from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
            from sklearn.pipeline import FeatureUnion, Pipeline
            from sklearn.svm import LinearSVC, LinearSVR
        except ImportError as error:
            raise RuntimeError(
                "The SVM attack requires scikit-learn. Install project requirements "
                "before selecting attack_model='svm'."
            ) from error

        challenges = np.asarray(challenges, dtype=np.float32)
        response_matrix = np.asarray(responses).reshape(len(challenges), -1)
        if challenges.ndim != 2 or challenges.shape[1] < 1:
            raise ValueError("SVM challenges must be a non-empty feature matrix.")
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} response columns; "
                f"received {response_matrix.shape[1]}."
            )
        if not regression and not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("Classification SVM responses must be bipolar (-1, +1).")

        mapped_challenges = _map_challenges(challenges, challenge_transform)
        feature_map = FeatureUnion(
            [
                ("mapped_challenges", "passthrough"),
                (
                    "rbf_random_fourier",
                    RBFSampler(
                        gamma=1.0 / mapped_challenges.shape[1],
                        n_components=RBF_FEATURE_COUNT,
                        random_state=seed,
                    ),
                ),
            ]
        )
        use_dual = len(challenges) < mapped_challenges.shape[1] + RBF_FEATURE_COUNT
        if regression:
            output_model: Any = MultiOutputRegressor(
                LinearSVR(
                    C=1.0,
                    epsilon=0.0,
                    dual=use_dual,
                    max_iter=10000,
                    random_state=seed,
                )
            )
        else:
            output_model = MultiOutputClassifier(
                LinearSVC(
                    C=1.0,
                    dual=use_dual,
                    max_iter=10000,
                    random_state=seed,
                )
            )

        self._model = Pipeline(
            [("feature_map", feature_map), ("output_model", output_model)]
        )
        self._n = challenges.shape[1]
        self._response_bits = response_bits
        self._regression = regression
        self._challenge_transform = challenge_transform
        self._model.fit(mapped_challenges, response_matrix)

    @property
    def challenge_length(self) -> int:
        return self._n

    @property
    def response_length(self) -> int:
        return self._response_bits

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        challenges = np.asarray(challenges, dtype=np.float32)
        mapped_challenges = _map_challenges(challenges, self._challenge_transform)
        predictions = np.asarray(self._model.predict(mapped_challenges))
        if self._regression:
            return predictions.reshape(len(challenges), self._response_bits)
        return predictions.reshape(len(challenges), self._response_bits).astype(np.int8)


class TorchFeatureMappedSVM(Simulation):
    """Fit a feature-mapped linear SVM using PyTorch on the selected device."""

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        seed: int,
        device: Any,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        regression: bool = False,
        challenge_transform: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = None,
    ):
        import torch
        from torch import nn

        challenges = np.asarray(challenges, dtype=np.float32)
        response_matrix = np.asarray(responses).reshape(len(challenges), -1)
        if challenges.ndim != 2 or challenges.shape[1] < 1:
            raise ValueError("SVM challenges must be a non-empty feature matrix.")
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} response columns; "
                f"received {response_matrix.shape[1]}."
            )
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive.")
        if not regression and not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("Classification SVM responses must be bipolar (-1, +1).")

        mapped_challenges = _map_challenges(challenges, challenge_transform)
        rng = np.random.default_rng(seed)
        feature_count = mapped_challenges.shape[1]
        self._rff_weights = torch.as_tensor(
            rng.normal(
                scale=np.sqrt(2.0 / feature_count),
                size=(feature_count, RBF_FEATURE_COUNT),
            ),
            dtype=torch.float32,
            device=device,
        )
        self._rff_phases = torch.as_tensor(
            rng.uniform(0.0, 2.0 * np.pi, size=RBF_FEATURE_COUNT),
            dtype=torch.float32,
            device=device,
        )
        self._rff_scale = np.sqrt(2.0 / RBF_FEATURE_COUNT)
        self._n = challenges.shape[1]
        self._response_bits = response_bits
        self._regression = regression
        self._challenge_transform = challenge_transform
        self.device = device
        self._model = nn.Linear(
            feature_count + RBF_FEATURE_COUNT, response_bits
        ).to(device)
        nn.init.zeros_(self._model.weight)
        nn.init.zeros_(self._model.bias)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=learning_rate, foreach=False
        )

        training_inputs = torch.as_tensor(
            mapped_challenges, dtype=torch.float32, device=device
        )
        training_labels = torch.as_tensor(
            response_matrix, dtype=torch.float32, device=device
        )
        for _ in range(epochs):
            order = torch.randperm(len(training_inputs), device=device)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                features = self._feature_map(training_inputs[indices])
                scores = self._model(features)
                if regression:
                    errors = torch.abs(scores - training_labels[indices])
                    per_output_loss = (
                        0.5 * self._model.weight.square().sum(dim=1)
                        + errors.mean(dim=0)
                    )
                else:
                    margins = 1.0 - training_labels[indices] * scores
                    per_output_loss = (
                        0.5 * self._model.weight.square().sum(dim=1)
                        + torch.clamp(margins, min=0).mean(dim=0)
                    )
                loss = per_output_loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self._model.eval()

    def _feature_map(self, challenges: Any) -> Any:
        import torch

        rbf_features = torch.cos(
            challenges @ self._rff_weights + self._rff_phases
        ) * self._rff_scale
        return torch.cat((challenges, rbf_features), dim=1)

    @property
    def challenge_length(self) -> int:
        return self._n

    @property
    def response_length(self) -> int:
        return self._response_bits

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        import torch

        mapped_challenges = _map_challenges(
            np.asarray(challenges, dtype=np.float32),
            self._challenge_transform,
        )
        inputs = torch.as_tensor(
            mapped_challenges,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            scores = self._model(self._feature_map(inputs))
            if self._regression:
                predictions = scores
            else:
                predictions = torch.where(scores >= 0, 1, -1)
        return predictions.to(torch.float32 if self._regression else torch.int8).cpu().numpy()


class TorchXorParitySVM(Simulation):
    """Fit linear chain margins and combine them with an XOR hinge objective."""

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        chain_count: int,
        seed: int,
        device: Any,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        import torch
        from torch import nn

        from .arbiter.xor_apuf import feature_map

        challenges = np.asarray(challenges, dtype=np.float32)
        response_matrix = np.asarray(responses).reshape(len(challenges), -1)
        if challenges.ndim != 2 or challenges.shape[1] < 1:
            raise ValueError("SVM challenges must be a non-empty feature matrix.")
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} response columns; "
                f"received {response_matrix.shape[1]}."
            )
        if chain_count < response_bits:
            raise ValueError("chain_count must be at least response_bits.")
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive.")
        if not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("Classification SVM responses must be bipolar (-1, +1).")

        torch.manual_seed(seed)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(seed)
        mapped_challenges = np.asarray(feature_map(challenges), dtype=np.float32)
        group_size, extra_chains = divmod(chain_count, response_bits)
        group_sizes = tuple(
            group_size + (output_index < extra_chains)
            for output_index in range(response_bits)
        )
        self._group_sizes = group_sizes
        self._n = challenges.shape[1]
        self._response_bits = response_bits
        self.device = device
        self._chain_model = nn.Linear(
            mapped_challenges.shape[1], chain_count
        ).to(device)
        optimizer = torch.optim.Adam(
            self._chain_model.parameters(), lr=learning_rate, foreach=False
        )

        training_inputs = torch.as_tensor(
            mapped_challenges, dtype=torch.float32, device=device
        )
        training_labels = torch.as_tensor(
            response_matrix, dtype=torch.float32, device=device
        )
        for _ in range(epochs):
            order = torch.randperm(len(training_inputs), device=device)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                scores = self._parity_scores(training_inputs[indices])
                margins = 1.0 - training_labels[indices] * scores
                regularization = []
                chain_start = 0
                for size in group_sizes:
                    chain_stop = chain_start + size
                    regularization.append(
                        0.5
                        * XOR_SVM_WEIGHT_DECAY
                        * self._chain_model.weight[chain_start:chain_stop]
                        .square()
                        .sum()
                    )
                    chain_start = chain_stop
                per_output_loss = (
                    torch.stack(regularization)
                    + torch.clamp(margins, min=0).mean(dim=0)
                )
                loss = per_output_loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self._chain_model.eval()

    def _parity_scores(self, features: Any) -> Any:
        import torch

        chain_scores = torch.tanh(self._chain_model(features))
        output_scores = []
        chain_start = 0
        for size in self._group_sizes:
            chain_stop = chain_start + size
            output_scores.append(
                torch.prod(chain_scores[:, chain_start:chain_stop], dim=1)
            )
            chain_start = chain_stop
        return torch.stack(output_scores, dim=1)

    @property
    def challenge_length(self) -> int:
        return self._n

    @property
    def response_length(self) -> int:
        return self._response_bits

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        import torch

        from .arbiter.xor_apuf import feature_map

        mapped_challenges = torch.as_tensor(
            np.asarray(feature_map(np.asarray(challenges, dtype=np.float32))),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            scores = self._parity_scores(mapped_challenges)
            predictions = torch.where(scores >= 0, 1, -1)
        return predictions.to(torch.int8).cpu().numpy()


__all__ = [
    "FeatureMappedSVM",
    "RBF_FEATURE_COUNT",
    "STAGE_NAME",
    "TorchFeatureMappedSVM",
    "TorchXorParitySVM",
    "XOR_STAGE_NAME",
]
