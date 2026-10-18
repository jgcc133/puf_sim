import numpy as np

from pypuf.simulation import Simulation
from puf_sim import evaluate_puf_population
from puf_sim.puf_backend import Backend
from puf_sim.puf_baseline_eval.exposed import (
    BipolarResponseSimulation,
    normalize_to_bipolar,
)
from puf_sim.puf_baseline_eval import report as baseline_report
from puf_sim.puf_baseline_eval import (
    binary_entropy,
    cross_challenge_randomness,
    evaluate_exposed_metrics,
    hamming_distance_distribution,
    majority_vote_repetitions,
    probability_of_misidentification,
    response_matrix,
    uniqueness_from_responses,
)
from puf_sim.puf_baseline_eval.puf_test import (
    _collect_population_responses,
    _prepare_population_for_metrics,
)


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


class FakeXPUBackend(Backend):
    def __init__(self) -> None:
        super().__init__("torch_xpu", np)

    def asarray(self, value, dtype=None):
        return np.asarray(value, dtype=dtype)

    def asnumpy(self, value):
        return np.asarray(value)

    def astype(self, value, dtype):
        return np.asarray(value).astype(dtype)


class BackendOnlyPUF(Simulation):
    def __init__(self) -> None:
        self.backend_evaluations = 0

    @property
    def challenge_length(self) -> int:
        return 1

    @property
    def response_length(self) -> int:
        return 1

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        raise AssertionError("The CPU evaluation path should not be used.")

    def eval_backend(self, challenges, backend):
        self.backend_evaluations += 1
        return backend.asarray(challenges[:, 0], dtype=backend.xp.int8)


def test_bipolar_response_wrapper_forwards_backend_evaluation():
    puf = BackendOnlyPUF()
    backend = FakeXPUBackend()
    challenges = np.array([[1], [0], [-1]], dtype=np.int8)

    responses = backend.eval_array(BipolarResponseSimulation(puf), challenges)

    np.testing.assert_array_equal(responses, [1, -1, -1])
    assert puf.backend_evaluations == 1


def test_normalized_response_wrapper_forwards_backend_evaluation():
    class ContinuousBackendPUF(BackendOnlyPUF):
        def eval(self, challenges: np.ndarray) -> np.ndarray:
            return ((challenges[:, :1] + 1) / 2).astype(float)

        def eval_backend(self, challenges, backend):
            self.backend_evaluations += 1
            values = backend.asarray(challenges[:, :1], dtype=backend.xp.float32)
            return (values + 1) / 2

    puf = ContinuousBackendPUF()
    wrapped = normalize_to_bipolar(puf, seed=10, calibration_samples=8)
    backend = FakeXPUBackend()
    challenges = np.array([[-1], [1]], dtype=np.int8)

    responses = backend.eval_array(wrapped, challenges)

    np.testing.assert_array_equal(
        responses,
        wrapped.eval(challenges),
    )
    assert puf.backend_evaluations == 1


def test_response_collection_uses_backend_for_supported_pufs():
    puf = BackendOnlyPUF()
    responses = next(
        _collect_population_responses(
            [puf],
            samples=3,
            repetitions=2,
            seed=5,
            backend=FakeXPUBackend(),
        )
    )

    assert responses.shape == (6, 1)
    assert set(np.unique(responses)).issubset({0, 1})
    assert puf.backend_evaluations == 1


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
    assert result["uniformity"] == expected_tie_vote


def test_cross_challenge_randomness_scores_uncorrelated_challenges_as_one():
    responses = np.array([
        [[-1], [-1]],
        [[-1], [1]],
        [[1], [-1]],
        [[1], [1]],
    ])

    assert cross_challenge_randomness(responses) == 1.0


def test_cross_challenge_randomness_scores_correlated_challenges_as_zero():
    responses = np.array([
        [[-1], [-1]],
        [[-1], [-1]],
        [[1], [1]],
        [[1], [1]],
    ])

    assert cross_challenge_randomness(responses) == 0.0


def test_cross_challenge_randomness_returns_zero_without_variable_pairs():
    constant_responses = np.ones((3, 2, 1), dtype=np.int8)
    one_challenge = np.array([[[-1]], [[1]]])

    assert cross_challenge_randomness(constant_responses) == 0.0
    assert cross_challenge_randomness(one_challenge) == 0.0


def test_uniqueness_from_responses_matches_pypuf_metric():
    from pypuf.metrics import uniqueness_data

    responses = np.array([
        [[-1], [-1], [1]],
        [[-1], [1], [1]],
        [[1], [-1], [1]],
    ])

    np.testing.assert_allclose(
        uniqueness_from_responses(responses),
        uniqueness_data(responses),
    )


def test_report_scores_uniformity_and_randomness_after_majority_vote(monkeypatch):
    class NoisyTablePUF(ConstantPUF):
        def __init__(self, responses):
            super().__init__(1)
            self.responses = np.asarray(responses, dtype=np.int8)
            self.evaluations = 0

        def eval(self, challenges: np.ndarray) -> np.ndarray:
            evaluation = self.evaluations
            self.evaluations += 1
            responses = self.responses
            if evaluation == 2:
                responses = -responses
            return responses.reshape(len(challenges), 1)

    monkeypatch.setattr(
        baseline_report,
        "evaluate_exposed_api",
        lambda *args, **kwargs: {
            "bias": np.array([0.0]),
            "reliability": np.ones((2, 1)),
            "total_influence": 0.0,
        },
    )
    population = [
        NoisyTablePUF([-1, -1]),
        NoisyTablePUF([-1, 1]),
        NoisyTablePUF([1, -1]),
        NoisyTablePUF([1, 1]),
    ]

    result = evaluate_exposed_metrics(
        population[0],
        instances=population,
        samples=2,
        repetitions=3,
        seed=10,
    )

    assert result["uniformity"] == 0.5
    assert result["randomness"] == 1.0


def test_report_majority_votes_one_dimensional_single_bit_responses(monkeypatch):
    class OneDimensionalPUF(Simulation):
        @property
        def challenge_length(self) -> int:
            return 4

        @property
        def response_length(self) -> int:
            return 1

        def eval(self, challenges: np.ndarray) -> np.ndarray:
            return np.where(challenges[:, 0] > 0, 1, -1).astype(np.int8)

    monkeypatch.setattr(
        baseline_report,
        "evaluate_exposed_api",
        lambda *args, **kwargs: {
            "bias": np.array([0.0]),
            "reliability": np.ones((4, 1)),
            "total_influence": 0.0,
        },
    )
    puf = OneDimensionalPUF()

    result = evaluate_exposed_metrics(puf, samples=4, repetitions=3, seed=10)

    assert result["bit_aliasing"].shape == (1,)
    assert 0.0 <= result["uniformity"] <= 1.0


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
