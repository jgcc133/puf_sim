import numpy as np

from pypuf.simulation import Simulation
from puf_sim import evaluate_puf_population
from puf_sim.puf_baseline_eval import report as baseline_report
from puf_sim.puf_baseline_eval import (
    binary_entropy,
    evaluate_exposed_metrics,
    hamming_distance_distribution,
    majority_vote_repetitions,
    probability_of_misidentification,
    response_matrix,
)
from puf_sim.puf_baseline_eval.puf_test import _prepare_population_for_metrics


class ConstantPUF(Simulation):
    def __init__(self, value: int) -> None:
        self.value = value

    @property
    def challenge_length(self) -> int:
        return 4

    @property
    def response_length(self) -> int:
        return 1

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        return np.full((len(challenges), 1), self.value, dtype=np.int8)


def test_response_data_helpers():
    responses = np.array([[-1], [1]])
    assert binary_entropy(responses) == 0.0
    np.testing.assert_allclose(hamming_distance_distribution(responses), [1.0])
    assert probability_of_misidentification(responses.reshape(2, 1, 1)) == 0.0


def test_majority_vote_repetitions_votes_per_sample_and_bit():
    responses = np.array([
        [[1, 1, -1], [-1, -1, 1]],
        [[-1, 1, -1], [1, -1, 1]],
    ])

    votes = majority_vote_repetitions(responses, np.random.default_rng(10))

    np.testing.assert_array_equal(votes, [[1, -1], [-1, 1]])


def test_majority_vote_repetitions_resolves_ties_reproducibly_and_fairly():
    responses = np.array([[[1, -1]] * 2000])

    first = majority_vote_repetitions(responses, np.random.default_rng(10))
    second = majority_vote_repetitions(responses, np.random.default_rng(10))

    np.testing.assert_array_equal(first, second)
    zero_fraction = np.mean(first == -1)
    assert 0.45 < zero_fraction < 0.55


def test_baseline_bit_aliasing_uses_majority_votes_across_repetitions(monkeypatch):
    class RepeatingPUF(ConstantPUF):
        def __init__(self):
            super().__init__(1)
            self.responses = iter((1, -1, 1))

        def eval(self, challenges: np.ndarray) -> np.ndarray:
            value = next(self.responses)
            return np.full((len(challenges), 1), value, dtype=np.int8)

    monkeypatch.setattr(
        baseline_report,
        "evaluate_exposed_api",
        lambda *args, **kwargs: {
            "bias": np.array([0.0]),
            "reliability": np.ones((1, 1)),
            "total_influence": 0.0,
        },
    )
    seed = 10
    expected_tie_vote = int(np.random.default_rng(seed).integers(0, 2))

    result = evaluate_exposed_metrics(
        RepeatingPUF(),
        samples=1,
        repetitions=2,
        seed=seed,
    )

    np.testing.assert_array_equal(result["bit_aliasing"], [expected_tie_vote])


def test_baseline_report_rounds_diffuseness_for_reporting(monkeypatch):
    class LongChallengePUF(ConstantPUF):
        @property
        def challenge_length(self) -> int:
            return 128

    monkeypatch.setattr(
        baseline_report,
        "evaluate_exposed_api",
        lambda *args, **kwargs: {
            "bias": np.array([0.0]),
            "reliability": np.ones((1, 1)),
            "total_influence": 61.19,
        },
    )

    result = evaluate_exposed_metrics(
        LongChallengePUF(1),
        samples=1,
        repetitions=1,
    )

    assert result["diffuseness"] == 0.478047


def test_binary_entropy_measures_balance_within_each_response():
    responses = np.array([
        [1, 1, -1, -1],
        [1, 1, 1, -1],
    ])
    entropy_for_three_ones = -(
        0.75 * np.log2(0.75) + 0.25 * np.log2(0.25)
    )
    expected = (1.0 + entropy_for_three_ones) / 2
    np.testing.assert_allclose(binary_entropy(responses), expected)


def test_binary_entropy_preserves_response_bits_for_population_input():
    responses = np.array([
        [[1, 1, -1, -1], [1, 1, 1, 1]],
        [[-1, -1, -1, -1], [-1, 1, 1, 1]],
    ])
    expected_per_response = np.array([
        1.0,
        0.0,
        0.0,
        -(
            0.75 * np.log2(0.75) + 0.25 * np.log2(0.25)
        ),
    ])
    np.testing.assert_allclose(
        binary_entropy(responses),
        expected_per_response.mean(),
    )


def test_binary_entropy_handles_zero_one_bit_responses():
    entropy_for_three_ones = -(
        0.75 * np.log2(0.75) + 0.25 * np.log2(0.25)
    )
    expected = (1.0 + entropy_for_three_ones) / 2
    numeric_bits = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    ])
    np.testing.assert_allclose(binary_entropy(numeric_bits), expected)
    np.testing.assert_array_equal(
        response_matrix(numeric_bits),
        np.array([[-1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.int8),
    )


def test_population_evaluation_exposes_official_and_derived_metrics():
    result = evaluate_puf_population(
        [ConstantPUF(-1), ConstantPUF(1)], samples=4, repetitions=3, seed=10
    )

    expected = {
        "uniqueness", "steadiness", "reliability", "uniformity", "randomness",
        "bit_aliasing", "diffuseness", "probability_of_misidentification",
        "similarity", "accuracy", "correlation", "influence", "total_influence",
        "noise_sensitivity",
    }
    assert expected.issubset(result)
    assert result["steadiness"].shape == (4, 1)
    assert result["reliability"].shape == (4, 1)
    assert result["diffuseness"] == 0.0
    assert result["randomness"] == 0.0
    assert result["probability_of_misidentification"] == 0.0


def test_single_device_population_has_empty_pair_metrics():
    result = evaluate_puf_population([ConstantPUF(1)], samples=2, repetitions=2)
    assert result["uniqueness"].size == 0
    assert result["similarity"] is None
    assert result["probability_of_misidentification"] == 0.0


def test_baseline_uses_bipolar_responses_without_changing_puf_outputs():
    zero_puf = ConstantPUF(0)
    one_puf = ConstantPUF(1)
    metric_population = _prepare_population_for_metrics(
        [zero_puf, one_puf],
        family="arbiter",
        calibration_seed=10,
    )

    challenges = np.ones((2, zero_puf.challenge_length), dtype=np.int8)
    np.testing.assert_array_equal(zero_puf.eval(challenges), [[0], [0]])
    np.testing.assert_array_equal(one_puf.eval(challenges), [[1], [1]])
    np.testing.assert_array_equal(metric_population[0].eval(challenges), [[-1], [-1]])
    np.testing.assert_array_equal(metric_population[1].eval(challenges), [[1], [1]])

    result = evaluate_exposed_metrics(
        metric_population[0],
        instances=metric_population,
        samples=4,
        repetitions=3,
        seed=10,
    )
    np.testing.assert_array_equal(result["bias"], [-1.0])
    np.testing.assert_allclose(result["reliability"], 1.0)
