"""Logistic-regression attack for Arbiter PUFs."""
from __future__ import annotations

from typing import Any

import numpy as np
from pypuf.simulation import ArbiterPUF, Simulation


STAGE_NAME = "arbiter_logistic_regression"


class ArbiterLogisticRegression(Simulation):
    """Fit one logistic classifier per response bit on ATT-transformed challenges."""

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        seed: int,
    ):
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as error:
            raise RuntimeError(
                "The Arbiter logistic-regression attack requires scikit-learn."
            ) from error

        challenge_matrix = np.asarray(challenges)
        response_matrix = np.asarray(responses).reshape(len(challenge_matrix), -1)
        if challenge_matrix.ndim != 2 or challenge_matrix.shape[1] < 1:
            raise ValueError("Challenges must be a non-empty two-dimensional matrix.")
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} response columns; "
                f"received {response_matrix.shape[1]}."
            )
        if not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("Arbiter responses must be bipolar (-1, +1).")

        self._challenge_length = challenge_matrix.shape[1]
        self._response_bits = response_bits
        self._models: list[Any] = []
        self._constant_responses: list[int | None] = []
        features = ArbiterPUF.transform_atf(challenge_matrix, 1)[:, 0, :]
        for response_bit in range(response_bits):
            labels = response_matrix[:, response_bit]
            classes = np.unique(labels)
            if len(classes) == 1:
                self._models.append(None)
                self._constant_responses.append(int(classes[0]))
            else:
                model = LogisticRegression(
                    max_iter=1000,
                    random_state=seed + response_bit,
                )
                model.fit(features, labels)
                self._models.append(model)
                self._constant_responses.append(None)

    @property
    def challenge_length(self) -> int:
        return self._challenge_length

    @property
    def response_length(self) -> int:
        return self._response_bits

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        challenge_matrix = np.asarray(challenges)
        if challenge_matrix.ndim != 2 or challenge_matrix.shape[1] != self._challenge_length:
            raise ValueError(
                f"Challenges must have shape (samples, {self._challenge_length})."
            )
        features = ArbiterPUF.transform_atf(challenge_matrix, 1)[:, 0, :]
        predictions = np.empty(
            (len(challenge_matrix), self._response_bits), dtype=np.int8
        )
        for response_bit, (model, constant_response) in enumerate(
            zip(self._models, self._constant_responses)
        ):
            if model is None:
                predictions[:, response_bit] = constant_response
            else:
                predictions[:, response_bit] = model.predict(features).astype(np.int8)
        return predictions


__all__ = ["ArbiterLogisticRegression", "STAGE_NAME"]
