"""Structure-aware attack for unhashed SRAM PUF responses."""
from __future__ import annotations

from typing import Any

import numpy as np
from pypuf.simulation import Simulation

from puf_sim.puf_implementations.analytical import hash_sram_responses

STAGE_NAME = "sram_direct_logistic_regression"
HASH_BRUTE_FORCE_STAGE_NAME = "sram_hash_random_fingerprint_search"
HASH_CANDIDATE_LIMIT = 100


class SRAMLogisticRegression(Simulation):
    """Fit one logistic classifier per direct challenge/fingerprint response bit."""

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        seed: int,
    ) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as error:
            raise RuntimeError(
                "The SRAM logistic-regression attack requires scikit-learn."
            ) from error

        challenge_matrix = np.asarray(challenges)
        response_matrix = np.asarray(responses).reshape(len(challenge_matrix), -1)
        if challenge_matrix.ndim != 2 or challenge_matrix.shape[1] < response_bits:
            raise ValueError(
                "Unhashed SRAM response width cannot exceed the challenge width."
            )
        if response_bits < 1 or response_matrix.shape[1] != response_bits:
            raise ValueError(
                f"Expected {response_bits} response columns; "
                f"received {response_matrix.shape[1]}."
            )
        if not np.isin(response_matrix, (-1, 1)).all():
            raise ValueError("SRAM responses must be bipolar (-1, +1).")

        self._challenge_length = challenge_matrix.shape[1]
        self._response_bits = response_bits
        self._models: list[Any] = []
        self._constant_responses: list[int | None] = []
        binary_challenges = (challenge_matrix[:, :response_bits] > 0).astype(np.float64)
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
                model.fit(binary_challenges[:, response_bit:response_bit + 1], labels)
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
        if (
            challenge_matrix.ndim != 2
            or challenge_matrix.shape[1] != self._challenge_length
        ):
            raise ValueError(
                f"Challenges must have shape (samples, {self._challenge_length})."
            )
        binary_challenges = (challenge_matrix[:, :self._response_bits] > 0).astype(
            np.float64
        )
        predictions = np.empty(
            (len(challenge_matrix), self._response_bits), dtype=np.int8
        )
        for response_bit, (model, constant_response) in enumerate(
            zip(self._models, self._constant_responses)
        ):
            if model is None:
                predictions[:, response_bit] = constant_response
            else:
                predictions[:, response_bit] = model.predict(
                    binary_challenges[:, response_bit:response_bit + 1]
                ).astype(np.int8)
        return predictions


class SRAMHashBruteForceAttack(Simulation):
    """Search random fixed fingerprints against hashed SRAM training CRPs."""

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        seed: int,
    ) -> None:
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
            raise ValueError("SRAM responses must be bipolar (-1, +1).")
        self._challenge_length = challenge_matrix.shape[1]
        self._response_bits = response_bits
        self.candidate_limit = min(
            HASH_CANDIDATE_LIMIT,
            1 << self._challenge_length,
        )
        binary_challenges = challenge_matrix > 0
        rng = np.random.default_rng(seed)
        candidate_fingerprints: list[np.ndarray] = []
        seen_candidates: set[bytes] = set()
        while len(candidate_fingerprints) < self.candidate_limit:
            fingerprint = rng.integers(
                0, 2, size=self._challenge_length, dtype=np.uint8
            ).astype(np.bool_)
            packed_fingerprint = np.packbits(fingerprint, bitorder="big").tobytes()
            if packed_fingerprint in seen_candidates:
                continue
            seen_candidates.add(packed_fingerprint)
            candidate_fingerprints.append(fingerprint)

        expected_rows = response_matrix
        best_score = (-1, -1)
        self.fingerprint_guess = candidate_fingerprints[0]
        for fingerprint in candidate_fingerprints:
            predicted = hash_sram_responses(
                np.logical_xor(binary_challenges, fingerprint),
                self._response_bits,
            )
            bit_matches = predicted == expected_rows
            row_matches = np.all(bit_matches, axis=1)
            score = (int(np.count_nonzero(row_matches)), int(np.count_nonzero(bit_matches)))
            if score > best_score:
                best_score = score
                self.fingerprint_guess = fingerprint
        self.training_match_count = best_score[0]
        self.training_element_match_count = best_score[1]
        self.training_response_count = len(response_matrix)
        self.training_element_count = response_matrix.size

    @property
    def challenge_length(self) -> int:
        return self._challenge_length

    @property
    def response_length(self) -> int:
        return self._response_bits

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        challenge_matrix = np.asarray(challenges)
        if (
            challenge_matrix.ndim != 2
            or challenge_matrix.shape[1] != self._challenge_length
        ):
            raise ValueError(
                f"Challenges must have shape (samples, {self._challenge_length})."
            )
        combined_bits = np.logical_xor(
            challenge_matrix > 0,
            self.fingerprint_guess,
        )
        return hash_sram_responses(combined_bits, self._response_bits)
