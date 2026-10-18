"""Baseline evaluation metrics for PUF simulations."""
from .derived import (
    binary_entropy,
    bit_aliasing,
    cross_challenge_randomness,
    hamming_distance_distribution,
    majority_vote_repetitions,
    probability_of_misidentification,
    response_matrix,
    uniqueness_from_responses,
)
from .exposed import evaluate_exposed_api
from .puf_test import (
    DEFAULT_INSTANCE_COUNT,
    DEFAULT_PUF_FAMILIES,
    evaluate_puf_population,
    run_baseline_experiment,
)
from .report import evaluate_exposed_metrics
from .writer import DEFAULT_REPORT_ROOT, save_baseline_report

__all__ = [
    "binary_entropy",
    "bit_aliasing",
    "cross_challenge_randomness",
    "DEFAULT_INSTANCE_COUNT",
    "DEFAULT_PUF_FAMILIES",
    "DEFAULT_REPORT_ROOT",
    "evaluate_exposed_api",
    "evaluate_exposed_metrics",
    "evaluate_puf_population",
    "hamming_distance_distribution",
    "majority_vote_repetitions",
    "probability_of_misidentification",
    "response_matrix",
    "run_baseline_experiment",
    "save_baseline_report",
    "uniqueness_from_responses",
]
