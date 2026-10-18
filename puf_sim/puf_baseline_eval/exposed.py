"""Consolidated calls to the public :mod:`pypuf.metrics` API."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from pypuf.io import ChallengeResponseSet, random_inputs
from pypuf.metrics import (
    accuracy,
    bias,
    correlation,
    influence,
    noise_sensitivity,
    reliability,
    similarity,
    total_influence,
    uniqueness,
)
from pypuf.simulation import Simulation

from puf_sim.puf_backend import Backend


class NormalizedResponseSimulation(Simulation):
    """Wrap a continuous-response simulation to expose bipolar ``{-1, 1}`` responses.

    Every metric in :mod:`pypuf.metrics` assumes responses in ``{-1, 1}``.
    Simulations with unbounded, non-negative outputs (e.g. the Optical PUF's
    intensity values) break those formulas silently instead of raising an
    error. This wrapper calibrates a per-response-bit median threshold from a
    sample of challenges, then binarizes every later ``eval`` call against it.
    """

    def __init__(self, instance: Simulation, seed: int = 0, calibration_samples: int = 1000) -> None:
        super().__init__()
        self._instance = instance
        calibration_challenges = random_inputs(
            n=instance.challenge_length, N=calibration_samples, seed=seed
        )
        calibration_responses = np.asarray(instance.eval(calibration_challenges), dtype=float)
        self._threshold = np.median(calibration_responses, axis=0)

    @property
    def challenge_length(self) -> int:
        return self._instance.challenge_length

    @property
    def response_length(self) -> int:
        return self._instance.response_length

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        responses = np.asarray(self._instance.eval(challenges), dtype=float)
        return np.where(responses >= self._threshold, 1, -1).astype(np.int8)

    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        responses = backend.eval_array(self._instance, challenges)
        threshold = backend.asarray(self._threshold, dtype=backend.xp.float32)
        return backend.astype(
            backend.xp.where(responses >= threshold, 1, -1),
            backend.xp.int8,
        )


class BipolarResponseSimulation(Simulation):
    """Expose a simulation's discrete responses as bipolar values for metrics."""

    def __init__(self, instance: Simulation) -> None:
        super().__init__()
        self._instance = instance

    @property
    def challenge_length(self) -> int:
        return self._instance.challenge_length

    @property
    def response_length(self) -> int:
        return self._instance.response_length

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        responses = np.asarray(self._instance.eval(challenges))
        return np.where(responses > 0, 1, -1).astype(np.int8)

    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        responses = backend.eval_array(self._instance, challenges)
        return backend.astype(
            backend.xp.where(responses > 0, 1, -1),
            backend.xp.int8,
        )


def normalize_to_bipolar(
    instance: Simulation, seed: int = 0, calibration_samples: int = 1000
) -> Simulation:
    """Return a bipolar-response wrapper for continuous-output simulations like the Optical PUF.

    Pass the result to :func:`evaluate_exposed_api` (or
    :func:`puf_sim.puf_baseline_eval.report.evaluate_exposed_metrics`) instead
    of the raw instance so every metric operates on ``{-1, 1}`` responses.
    """
    return NormalizedResponseSimulation(instance, seed=seed, calibration_samples=calibration_samples)


def evaluate_exposed_api(
    instance: Simulation,
    instances: Optional[Iterable[Simulation]] = None,
    test_set: Optional[ChallengeResponseSet] = None,
    seed: int = 0,
    samples: int = 1000,
    repetitions: int = 17,
    input_noise: float = 0.01,
    precomputed_uniqueness: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Call every applicable metric exposed by ``pypuf.metrics``.

    Returned keys correspond directly to the public API names: ``bias``,
    ``reliability``, ``uniqueness``, ``similarity``, ``accuracy``,
    ``correlation``, ``influence``, ``total_influence``, and
    ``noise_sensitivity``. Metrics requiring a comparison object return
    ``None`` when that object is not supplied.
    """
    population = list(instances or [instance])
    if not population:
        raise ValueError("At least one PUF instance is required.")

    result: Dict[str, Any] = {
        "bias": np.asarray(bias(instance, seed=seed + 1, N=samples)),
        "reliability": np.asarray(
            reliability(instance, seed=seed + 3, N=samples, r=repetitions)
        ),
        "uniqueness": np.array([], dtype=float),
        "similarity": None,
        "accuracy": None,
        "correlation": None,
    }
    if len(population) >= 2:
        result["uniqueness"] = np.asarray(
            precomputed_uniqueness
            if precomputed_uniqueness is not None
            else uniqueness(population, seed=seed + 2, N=samples)
        )
        result["similarity"] = np.asarray(
            similarity(population[0], population[1], seed=seed + 6, N=samples)
        )

    influence_values = [
        influence(instance, i=index, seed=seed + 10 + index, N=samples)
        for index in range(instance.challenge_length)
    ]
    result["influence"] = np.asarray(influence_values)
    result["total_influence"] = float(
        total_influence(instance, seed=seed + 4, N=samples)
    )
    result["noise_sensitivity"] = float(
        noise_sensitivity(
            instance,
            eps=input_noise,
            seed=seed + 5,
            N=samples,
        )
    )

    if test_set is not None:
        result["accuracy"] = np.asarray(accuracy(instance, test_set))
        result["correlation"] = np.asarray(correlation(instance, test_set))

    return result


__all__ = [
    "BipolarResponseSimulation",
    "evaluate_exposed_api",
    "NormalizedResponseSimulation",
    "normalize_to_bipolar",
]
