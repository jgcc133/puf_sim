"""Population-level baseline evaluation entry points."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from pypuf.io import ChallengeResponseSet
from pypuf.simulation import Simulation

from puf_sim.puf_implementations import create_puf
from puf_sim.puf_implementations.analytical import StaticRandomAccessMemoryPUF
from .exposed import normalize_to_bipolar
from .report import evaluate_exposed_metrics
from .writer import save_baseline_report

DEFAULT_PUF_FAMILIES = (
    "optical",
    "arbiter",
    "ff_apuf",
    "xor_apuf",
    "interpose",
    "permutation",
    "ring_oscillator",
    "sram",
    "memristive",
    "silicon_photonic",
)
DEFAULT_INSTANCE_COUNT = 100


def _effective_noisiness(family: str, noisiness: Optional[float]) -> float:
    if noisiness is not None:
        return noisiness
    family_key = family.strip().lower().replace("-", "_").replace(" ", "_")
    return 0.01 if family_key in {"sram", "static_random_access_memory"} else 0.0


def _prepare_population_for_metrics(
    population: list[Simulation],
    family: str,
    calibration_seed: int,
    calibration_samples: int = 1000,
) -> list[Simulation]:
    family_key = family.strip().lower().replace("-", "_").replace(" ", "_")
    if family_key != "optical":
        return population

    return [
        normalize_to_bipolar(
            puf,
            seed=calibration_seed + index,
            calibration_samples=calibration_samples,
        )
        for index, puf in enumerate(population)
    ]

def _evaluate_family(
    family_index: int,
    family: str,
    instances_per_family: int,
    n: int,
    samples: int,
    repetitions: int,
    input_noise: float,
    seed: int,
    k: Optional[int],
    response_bits: int,
    noisiness: Optional[float],
    backend: str,
    device: Optional[int],
    batch_size: Optional[int],
    bch_correction_capacity: int,
) -> tuple[str, Dict[str, Any]]:
    """Create and evaluate one family; suitable for a worker thread."""
    print(f"Evaluating {instances_per_family} instances of {family!r}...")
    family_key = family.strip().lower().replace("-", "_").replace(" ", "_")
    effective_k = k
    if effective_k is None:
        effective_k = (
            n
            if family_key in {"xor_arbiter", "xor_apuf", "interpose", "permutation"}
            else 2
        )
    population = [
        create_puf(
            family,
            n=n,
            seed=seed + family_index * instances_per_family + device_index,
            k=effective_k,
            response_bits=response_bits,
            noisiness=_effective_noisiness(family, noisiness),
            bch_correction_capacity=bch_correction_capacity,
        )
        for device_index in range(instances_per_family)
    ]
    population = _prepare_population_for_metrics(
        population,
        family,
        calibration_seed=seed + family_index * instances_per_family,
    )

    report = evaluate_exposed_metrics(
        population[0],
        instances=population,
        samples=samples,
        repetitions=repetitions,
        input_noise=input_noise,
        seed=seed + family_index,
        backend=backend,
        device=device,
        batch_size=batch_size,
    )
    if isinstance(population[0], StaticRandomAccessMemoryPUF):
        report["bch_total_reads"] = sum(
            puf.bch_total_reads for puf in population
            if isinstance(puf, StaticRandomAccessMemoryPUF)
        )
        report["bch_uncorrectable_reads"] = sum(
            puf.bch_uncorrectable_reads for puf in population
            if isinstance(puf, StaticRandomAccessMemoryPUF)
        )
    return family, report


def evaluate_puf_population(
    instances: Iterable[Simulation],
    test_set: Optional[ChallengeResponseSet] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate a supplied PUF population with the baseline metric report."""
    population = list(instances)
    if not population:
        raise ValueError("At least one PUF instance is required.")
    return evaluate_exposed_metrics(
        population[0], instances=population, test_set=test_set, **kwargs
    )


def run_baseline_experiment(
    families: Optional[Sequence[str]] = None,
    instances_per_family: int = DEFAULT_INSTANCE_COUNT,
    n: int = 64,
    samples: int = 1000,
    repetitions: int = 17,
    input_noise: float = 0.01,
    seed: int = 20260921,
    k: Optional[int] = None,
    response_bits: int = 1,
    noisiness: Optional[float] = None,
    report_root: Optional[Path | str] = None,
    save_report: bool = True,
    workers: Optional[int] = None,
    backend: str = "numpy",
    device: Optional[int] = None,
    batch_size: Optional[int] = None,
    bch_correction_capacity: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate the baseline metrics for a population of each PUF family.

    By default this creates and evaluates 100 independently seeded instances
    for each of the ten canonical families in ``DEFAULT_PUF_FAMILIES``.
    """
    if instances_per_family < 1:
        raise ValueError("instances_per_family must be at least one.")
    selected_families = tuple(families or DEFAULT_PUF_FAMILIES)
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least one or None.")
    max_workers = workers or min(len(selected_families), os.cpu_count() or 1)
    results: Dict[str, Dict[str, Any]] = {}
    print(f"Running baseline evaluation for {len(selected_families)} families...")
    print(f"Using {max_workers} family worker(s).")
    print(f"Expected number of computations: {len(selected_families) * instances_per_family}*{samples}*{repetitions} = {len(selected_families) * instances_per_family * samples * repetitions}")

    worker_arguments = [
        (
            family_index,
            family,
            instances_per_family,
            n,
            samples,
            repetitions,
            input_noise,
            seed,
            k,
            response_bits,
            noisiness,
            backend,
            device,
            batch_size,
            bch_correction_capacity,
        )
        for family_index, family in enumerate(selected_families)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        completed = executor.map(lambda args: _evaluate_family(*args), worker_arguments)
        for family, report in completed:
            results[family] = report

    if save_report:
        parameters = {
            "instances_per_family": instances_per_family,
            "n": n,
            "samples": samples,
            "repetitions": repetitions,
            "input_noise": input_noise,
            "seed": seed,
            "k": k,
            "response_bits": response_bits,
            "noisiness": noisiness,
            "effective_noisiness_by_family": {
                family: _effective_noisiness(family, noisiness)
                for family in selected_families
            },
            "bch_correction_capacity": bch_correction_capacity,
            "workers": max_workers,
            "backend": backend,
            "device": device,
            "batch_size": batch_size,
            "default_families": DEFAULT_PUF_FAMILIES,
        }
        report_directory = save_baseline_report(
            results,
            parameters=parameters,
            families=selected_families,
            report_root=report_root,
        )
        print(f"Saved baseline report to: {report_directory}")

    return results

__all__ = [
    "DEFAULT_INSTANCE_COUNT",
    "DEFAULT_PUF_FAMILIES",
    "evaluate_puf_population",
    "run_baseline_experiment",
]
