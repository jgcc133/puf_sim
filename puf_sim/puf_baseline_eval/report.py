"""Combined baseline report for official and derived PUF metrics."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from pypuf.io import ChallengeResponseSet, random_inputs
from pypuf.simulation import Simulation

from puf_sim.puf_backend import get_backend
from puf_sim.puf_implementations.analytical import SiliconPhotonicPUF

from .derived import (
    bit_aliasing,
    cross_challenge_randomness,
    hamming_distance_distribution,
    majority_vote_repetitions,
    probability_of_misidentification,
    response_matrix,
    uniqueness_from_responses,
)
from .exposed import BipolarResponseSimulation, evaluate_exposed_api


def _silicon_photonic_population(
    population: list[Simulation],
) -> Optional[list[SiliconPhotonicPUF]]:
    raw_instances = [
        puf._instance if isinstance(puf, BipolarResponseSimulation) else puf
        for puf in population
    ]
    if all(isinstance(puf, SiliconPhotonicPUF) for puf in raw_instances):
        return raw_instances
    return None


def _evaluate_silicon_population(
    population: list[SiliconPhotonicPUF],
    challenges: np.ndarray,
    backend: Any,
    batch_size: int,
) -> Any:
    batches = [
        SiliconPhotonicPUF.eval_population_backend(
            population[start:start + batch_size],
            challenges,
            backend,
            batch_size=batch_size,
        )
        for start in range(0, len(population), batch_size)
    ]
    return backend.xp.cat(batches, dim=0)


def evaluate_exposed_metrics(
    instance: Simulation,
    instances: Optional[Iterable[Simulation]] = None,
    test_set: Optional[ChallengeResponseSet] = None,
    seed: int = 0,
    samples: int = 1000,
    repetitions: int = 17,
    input_noise: float = 0.01,
    backend: str = "numpy",
    device: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Return one report containing official and derived evaluation metrics."""
    population = list(instances or [instance])
    if not population:
        raise ValueError("At least one PUF instance is required.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least one or None.")
    if repetitions < 1:
        raise ValueError("repetitions must be at least one.")

    selected_backend = get_backend(backend, device=device)
    xp = selected_backend.xp
    challenges = random_inputs(instance.challenge_length, samples, seed)
    step = batch_size or len(population)
    silicon_population = (
        _silicon_photonic_population(population)
        if selected_backend.name == "torch_xpu"
        else None
    )
    if silicon_population is not None:
        population_responses = _evaluate_silicon_population(
            silicon_population,
            challenges,
            selected_backend,
            step,
        )
    else:
        population_responses = []
        for start in range(0, len(population), step):
            population_responses.extend(
                selected_backend.eval_array(puf, challenges)
                for puf in population[start:start + step]
            )
        population_responses = xp.stack(
            [
                response_matrix(response, selected_backend)
                for response in population_responses
            ],
            axis=0,
        )
    tie_rng = np.random.default_rng(seed)
    if silicon_population is not None:
        positive_counts = np.zeros(
            (len(population), samples, instance.response_length),
            dtype=np.int16,
        )
        for start in range(0, len(silicon_population), step):
            population_batch = silicon_population[start:start + step]
            for _ in range(repetitions):
                repeated_batch = SiliconPhotonicPUF.eval_population_backend(
                    population_batch,
                    challenges,
                    selected_backend,
                    batch_size=step,
                )
                positive_counts[start:start + len(population_batch)] += (
                    selected_backend.asnumpy(repeated_batch > 0)
                )
        ties = positive_counts * 2 == repetitions
        votes_one = positive_counts * 2 > repetitions
        tie_count = int(np.count_nonzero(ties))
        if tie_count:
            votes_one[ties] = tie_rng.integers(0, 2, size=tie_count).astype(bool)
        voted_population_responses = np.where(
            votes_one, 1, -1
        ).astype(np.int8)
    else:
        voted_population_responses = np.empty(
            (len(population), samples, instance.response_length),
            dtype=np.int8,
        )
        for device_index, puf in enumerate(population):
            repeated_responses = np.stack(
                [
                    selected_backend.asnumpy(
                        response_matrix(
                            selected_backend.eval_array(puf, challenges),
                            selected_backend,
                        )
                    )
                    for _ in range(repetitions)
                ],
                axis=-1,
            )
            expected_shape = (samples, instance.response_length, repetitions)
            if repeated_responses.shape != expected_shape:
                raise ValueError(
                    f"Expected repeated responses with shape {expected_shape}, "
                    f"got {repeated_responses.shape}."
                )
            voted_population_responses[device_index] = majority_vote_repetitions(
                repeated_responses, tie_rng
            )
    repeated_bit_aliasing = bit_aliasing(voted_population_responses)

    precomputed_uniqueness = None
    if silicon_population is not None and len(population) >= 2:
        uniqueness_challenges = random_inputs(
            instance.challenge_length,
            samples,
            seed + 2,
        )
        uniqueness_responses = _evaluate_silicon_population(
            silicon_population,
            uniqueness_challenges,
            selected_backend,
            step,
        )
        precomputed_uniqueness = uniqueness_from_responses(
            uniqueness_responses,
            selected_backend,
        )

    result = evaluate_exposed_api(
        instance,
        instances=population,
        test_set=test_set,
        seed=seed,
        samples=samples,
        repetitions=repetitions,
        input_noise=input_noise,
        precomputed_uniqueness=precomputed_uniqueness,
    )
    result.update({
        "uniformity": float(np.mean(voted_population_responses > 0)),
        "randomness": cross_challenge_randomness(
            voted_population_responses,
            selected_backend,
        ),
        "bit_aliasing": repeated_bit_aliasing,
        "probability_of_misidentification": probability_of_misidentification(
            population_responses,
            selected_backend,
        ),
        "hamming_distance_distribution": hamming_distance_distribution(
            population_responses,
            selected_backend,
        ),
        "backend": selected_backend.name,
        "device": (
            None
            if selected_backend.device is None
            else str(selected_backend.device)
        ),
        "batch_size": step,
    })
    result["steadiness"] = result["reliability"]
    result["diffuseness"] = round(
        result["total_influence"] / max(1, instance.challenge_length),
        6,
    )
    return result


__all__ = ["evaluate_exposed_metrics"]
