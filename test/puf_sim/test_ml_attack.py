import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import puf_sim.puf_ml_attack.experiment as experiment
import puf_sim.puf_ml_attack.unknown as unknown_attack
from puf_sim.puf_ml_attack.unknown import (
    _split_indices,
    generate_crps,
    run_attack_unknown_puf,
)
from puf_sim.puf_ml_attack.experiment import _pair_accuracy
from puf_sim.puf_ml_attack.arbiter.xor_apuf import (
    AgentTrainer as XorAgentTrainer,
    _XorQuadraticModel,
    _XorRankTwoModel,
    XorTorchPUFModel,
    _XorParityModel,
    feature_map as xor_feature_map,
)
from puf_sim.puf_ml_attack.arbiter.feed_forward import (
    AgentTrainer as FeedForwardAgentTrainer,
)
from puf_sim.puf_ml_attack.arbiter.deep_neural import (
    AgentTrainer as DeepNeuralAgentTrainer,
)
from puf_sim.puf_ml_attack.arbiter.interpose import (
    AgentTrainer as InterposeAgentTrainer,
    InterposeCore,
    InterposeTorchPUFModel,
    LogisticAgentTrainer as InterposeLogisticAgentTrainer,
    StagedSplitAgentTrainer as InterposeStagedSplitAgentTrainer,
)
from puf_sim.puf_ml_attack.svm import (
    FeatureMappedSVM,
    TorchFeatureMappedSVM,
    TorchXorParitySVM,
)
from puf_sim.puf_ml_attack.fc_lstm import AgentTrainer as FCBiLSTMTrainer
from puf_sim.puf_ml_attack.fc_lstm import TorchFCBiLSTMModel
from puf_sim.puf_ml_attack.logistic_regression import ArbiterLogisticRegression
from puf_sim.puf_ml_attack.ring_oscillator import (
    RingOscillatorLogisticRegression,
)
from puf_sim.puf_ml_attack.sram import SRAMHashBruteForceAttack
from puf_sim.puf_implementations import create_puf
from puf_sim.puf_implementations.analytical import hash_sram_responses
from puf_sim.puf_ml_attack.writer import save_ml_attack_report


class EchoModel:
    def eval(self, challenges):
        return challenges[:, :1]


class FakeTorchModel:
    def __init__(self, depth):
        self.depth = depth

    def cpu(self):
        return self


def test_pair_accuracy_flattens_responses():
    challenges = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    dataset = SimpleNamespace(
        challenges=challenges,
        responses=np.array([[[1]], [[-1]]], dtype=np.int8),
    )

    assert _pair_accuracy(EchoModel(), dataset) == 1.0


def test_feature_mapped_svm_fits_multiple_response_bits():
    challenges = np.random.default_rng(7).choice(
        (-1, 1), size=(48, 6)
    ).astype(np.int8)
    responses = challenges[:, :2]
    model = FeatureMappedSVM(
        challenges, responses, response_bits=2, seed=11
    )

    assert model.challenge_length == 6
    assert model.response_length == 2
    np.testing.assert_array_equal(model.eval(challenges), responses)


def test_feature_mapped_svm_applies_challenge_transform():
    challenges = np.random.default_rng(10).choice(
        (-1, 1), size=(64, 6)
    ).astype(np.int8)

    def parity_feature_map(values):
        return np.column_stack((values, np.prod(values, axis=1)))

    responses = np.prod(challenges, axis=1).reshape(-1, 1)
    model = FeatureMappedSVM(
        challenges,
        responses,
        response_bits=1,
        seed=14,
        challenge_transform=parity_feature_map,
    )

    assert model.challenge_length == challenges.shape[1]
    np.testing.assert_array_equal(model.eval(challenges), responses)


def test_feature_mapped_svm_supports_continuous_regression():
    challenges = np.random.default_rng(8).choice(
        (-1, 1), size=(64, 6)
    ).astype(np.int8)
    responses = (challenges[:, 0] + 2 * challenges[:, 1]).reshape(-1, 1)
    model = FeatureMappedSVM(
        challenges, responses, response_bits=1, seed=12, regression=True
    )

    assert np.mean(np.abs(model.eval(challenges) - responses)) < 0.2


def test_torch_feature_mapped_svm_trains_on_selected_device():
    import torch

    challenges = np.random.default_rng(9).choice(
        (-1, 1), size=(48, 6)
    ).astype(np.int8)
    responses = challenges[:, :2]
    model = TorchFeatureMappedSVM(
        challenges,
        responses,
        response_bits=2,
        seed=13,
        device=torch.device("cpu"),
        epochs=40,
        batch_size=24,
        learning_rate=0.03,
    )
    predictions = model.eval(challenges)

    assert predictions.shape == responses.shape
    assert set(np.unique(predictions)).issubset({-1, 1})
    assert np.mean(predictions == responses) > 0.8


def test_torch_feature_mapped_svm_applies_challenge_transform():
    import torch

    challenges = np.random.default_rng(11).choice(
        (-1, 1), size=(48, 6)
    ).astype(np.int8)

    def parity_feature_map(values):
        return np.column_stack((values, values[:, 0] * values[:, 1]))

    responses = (challenges[:, 0] * challenges[:, 1]).reshape(-1, 1)
    model = TorchFeatureMappedSVM(
        challenges,
        responses,
        response_bits=1,
        seed=15,
        device=torch.device("cpu"),
        epochs=40,
        batch_size=24,
        learning_rate=0.03,
        challenge_transform=parity_feature_map,
    )

    assert model.challenge_length == challenges.shape[1]
    np.testing.assert_array_equal(model.eval(challenges), responses)


def test_torch_xor_parity_svm_learns_chain_parity():
    import torch

    rng = np.random.default_rng(45)
    challenges = rng.choice((-1, 1), size=(1024, 8)).astype(np.int8)
    test_challenges = rng.choice((-1, 1), size=(512, 8)).astype(np.int8)
    mapped = xor_feature_map(challenges)
    mapped_test = xor_feature_map(test_challenges)
    chain_weights = rng.normal(size=(2, mapped.shape[1]))
    chain_biases = rng.normal(size=2)
    responses = np.prod(
        np.where(mapped @ chain_weights.T + chain_biases >= 0, 1, -1),
        axis=1,
    ).astype(np.int8).reshape(-1, 1)
    test_responses = np.prod(
        np.where(mapped_test @ chain_weights.T + chain_biases >= 0, 1, -1),
        axis=1,
    ).astype(np.int8).reshape(-1, 1)
    model = TorchXorParitySVM(
        challenges,
        responses,
        response_bits=1,
        chain_count=2,
        seed=14,
        device=torch.device("cpu"),
        epochs=500,
        batch_size=128,
        learning_rate=0.01,
    )

    assert np.mean(model.eval(test_challenges) == test_responses) > 0.9


def test_svm_xpu_backend_selects_torch_implementation(monkeypatch):
    fake_xpu = SimpleNamespace(type="xpu")

    class FakeTorchXpuSVM:
        def __init__(
            self,
            challenges,
            responses,
            response_bits,
            seed,
            device,
            epochs,
            batch_size,
            learning_rate,
            regression=False,
            challenge_transform=None,
        ):
            assert device is fake_xpu
            self.challenge_transform = challenge_transform
            self.model = FeatureMappedSVM(
                challenges,
                responses,
                response_bits,
                seed,
                regression=regression,
                challenge_transform=challenge_transform,
            )

        @property
        def challenge_length(self):
            return self.model.challenge_length

        @property
        def response_length(self):
            return self.model.response_length

        def eval(self, challenges):
            return self.model.eval(challenges)

    monkeypatch.setattr(
        experiment, "_resolve_device", lambda backend, device: fake_xpu
    )
    monkeypatch.setattr(
        experiment._svm, "TorchFeatureMappedSVM", FakeTorchXpuSVM
    )
    result = experiment.run_ml_attack_experiment(
        families=["arbiter"],
        n=8,
        k=1,
        response_bits=2,
        training_samples=32,
        validation_samples=16,
        test_samples=16,
        seed=23,
        success_threshold=1.0,
        max_depth=1,
        agents=1,
        epochs=1,
        batch_size=8,
        backend="torch_xpu",
        device=0,
        attack_model="svm",
        save_report=False,
        puf_instances=1,
    )["arbiter"]["instances"][0]

    assert result["model_architecture"] == "svm"
    assert result["activation_types"]["estimator"] == (
        "PyTorch hinge-loss SVM per response bit"
    )
    assert result["epochs_trained"] == 1
    assert result["agent_epochs_trained"] == 1
    assert result["stages"][0]["epochs"] == 1


def test_xor_svm_uses_arbiter_threshold_transform(monkeypatch):
    fake_xpu = SimpleNamespace(type="xpu")
    received = {}

    class FakeXorParitySVM:
        def __init__(
            self,
            challenges,
            responses,
            response_bits,
            chain_count,
            seed,
            device,
            epochs,
            batch_size,
            learning_rate,
        ):
            received["chain_count"] = chain_count
            received["epochs"] = epochs
            self._n = challenges.shape[1]
            self._response_bits = response_bits

        @property
        def challenge_length(self):
            return self._n

        @property
        def response_length(self):
            return self._response_bits

        def eval(self, challenges):
            return np.ones(
                (len(challenges), self._response_bits), dtype=np.int8
            )

    monkeypatch.setattr(
        experiment, "_resolve_device", lambda backend, device: fake_xpu
    )
    monkeypatch.setattr(
        experiment._svm, "TorchXorParitySVM", FakeXorParitySVM
    )
    result = experiment.run_ml_attack_experiment(
        families=["xor_apuf"],
        n=8,
        k=4,
        response_bits=2,
        training_samples=32,
        validation_samples=16,
        test_samples=16,
        seed=24,
        success_threshold=1.0,
        max_depth=1,
        agents=1,
        epochs=2,
        batch_size=8,
        backend="torch_xpu",
        device=0,
        attack_model="svm",
        save_report=False,
        puf_instances=1,
    )["xor_apuf"]["instances"][0]

    assert received["chain_count"] == 4
    assert received["epochs"] == 2
    assert result["activation_types"]["feature_mapping"] == [
        "XOR Arbiter threshold transform"
    ]
    assert result["activation_types"]["estimator"] == (
        "PyTorch parity-composed hinge-loss SVM over learned chain scores"
    )
    assert result["epochs_trained"] == 2
    assert result["stages"][0]["epochs"] == 2


def test_arbiter_multi_response_shape_and_attack(monkeypatch, tmp_path):
    import torch

    puf = create_puf("arbiter", n=8, response_bits=3, seed=17)
    challenges = np.random.default_rng(18).choice(
        (-1, 1), size=(12, 8)
    ).astype(np.int8)
    responses = puf.eval(challenges)

    assert puf.response_length == 3
    assert responses.shape == (12, 3)
    assert set(np.unique(responses)).issubset({-1, 1})

    class DatasetFactory:
        @staticmethod
        def from_simulation(target, N, seed):
            return SimpleNamespace(
                challenges=challenges[:N],
                responses=target.eval(challenges[:N]),
                seed=seed,
            )

    monkeypatch.setattr(experiment, "ChallengeResponseSet", DatasetFactory)
    monkeypatch.setattr(experiment, "_resolve_device", lambda backend, device: torch.device("cpu"))
    result = experiment.run_ml_attack_experiment(
        families=["arbiter"],
        n=8,
        training_samples=12,
        validation_samples=12,
        test_samples=12,
        seed=20,
        response_bits=3,
        success_threshold=0.95,
        max_depth=1,
        agents=1,
        epochs=1,
        batch_size=12,
        backend="cpu",
        save_report=True,
        report_root=tmp_path,
        puf_instances=1,
        attack_model="neural",
    )["arbiter"]["instances"][0]

    assert result["activation_types"] == {
        "hidden_layers": ["Tanh"],
        "internal_layers": [],
        "output_layer": "Linear logits",
        "prediction": "Bipolar threshold",
    }
    assert result["test_response_count"] == 12
    assert result["matched_test_responses"] == result["test_response_count"] * result[
        "test_accuracy"
    ]
    response_file = next((tmp_path / "general").iterdir()) / "responses.csv"
    with response_file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [
        "label",
        "bit_0",
        "bit_1",
        "bit_2",
        "ml_bit_0",
        "ml_bit_1",
        "ml_bit_2",
        "match",
    ]
    assert len(rows) == 13
    assert sum(int(row[-1]) for row in rows[1:]) == result["matched_test_responses"]
    report_directory = next((tmp_path / "general").iterdir())
    with (report_directory / "results.json").open(encoding="utf-8") as handle:
        report_results = json.load(handle)
    saved_instance = report_results["arbiter"]["instances"][0]
    assert saved_instance["activation_types"] == result["activation_types"]
    assert saved_instance["stages"][0]["activation_types"] == result["activation_types"]


def test_xor_inference_applies_training_feature_map():
    import torch

    puf = create_puf("xor_apuf", n=8, seed=17, k=2)
    challenges = np.array([
        [1, -1, 1, -1, 1, -1, 1, -1],
        [-1, 1, -1, 1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1],
    ], dtype=np.int8)
    model = _XorParityModel(8, puf.k, seed=0, hidden_layers=0)
    with torch.no_grad():
        model.chain_logits.weight.copy_(
            torch.as_tensor(puf.weight_array[:, :-1], dtype=torch.float32)
        )
        model.chain_logits.bias.copy_(
            torch.as_tensor(puf.weight_array[:, -1], dtype=torch.float32)
        )
    attack_model = XorTorchPUFModel([model], 8, 1, torch.device("cpu"))

    assert np.array_equal(
        attack_model.eval(challenges).reshape(-1), puf.eval(challenges).reshape(-1)
    )


def test_two_chain_quadratic_model_matches_xor_puf_outputs():
    import torch

    puf = create_puf("xor_apuf", n=8, k=2, seed=16)
    challenges = np.random.default_rng(17).choice(
        (-1, 1), size=(256, 8)
    ).astype(np.int8)
    model = _XorQuadraticModel(feature_count=8, response_bits=1, seed=18)
    first_chain, second_chain = torch.as_tensor(
        puf.weight_array, dtype=torch.float32
    )

    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias[0] = first_chain[-1] * second_chain[-1]
        model.output.weight[0, :8] = (
            first_chain[:-1] * second_chain[-1]
            + second_chain[:-1] * first_chain[-1]
        )
        rows = model.pair_rows
        columns = model.pair_columns
        pair_weights = first_chain[rows] * second_chain[columns]
        off_diagonal = rows != columns
        pair_weights[off_diagonal] += (
            first_chain[columns[off_diagonal]]
            * second_chain[rows[off_diagonal]]
        )
        model.output.weight[0, 8:] = pair_weights

    predicted = model(
        torch.as_tensor(xor_feature_map(challenges), dtype=torch.float32)
    )
    predicted = torch.where(predicted >= 0, 1, -1).numpy()
    np.testing.assert_array_equal(predicted.reshape(-1), puf.eval(challenges))


def test_two_chain_rank_two_model_matches_xor_puf_outputs():
    import torch

    puf = create_puf("xor_apuf", n=8, k=2, seed=19)
    challenges = np.random.default_rng(20).choice(
        (-1, 1), size=(256, 8)
    ).astype(np.int8)
    model = _XorRankTwoModel(feature_count=8, response_bits=1, seed=21)

    with torch.no_grad():
        model.chain_scores.weight.copy_(
            torch.as_tensor(puf.weight_array[:, :-1], dtype=torch.float32)
        )
        model.chain_scores.bias.copy_(
            torch.as_tensor(puf.weight_array[:, -1], dtype=torch.float32)
        )

    predictions = model(
        torch.as_tensor(xor_feature_map(challenges), dtype=torch.float32)
    )
    predictions = torch.where(predictions >= 0, 1, -1).numpy()
    np.testing.assert_array_equal(predictions.reshape(-1), puf.eval(challenges))


def test_xor_multi_response_attack_matches_grouped_chain_outputs():
    import torch

    response_bits = 3
    puf = create_puf(
        "xor_apuf", n=8, k=7, response_bits=response_bits, seed=17
    )
    challenges = np.random.default_rng(18).choice(
        (-1, 1), size=(16, 8)
    ).astype(np.int8)
    responses = puf.eval(challenges)
    model = _XorParityModel(
        8, puf.k, seed=0, response_bits=response_bits, hidden_layers=0
    )
    with torch.no_grad():
        model.chain_logits.weight.copy_(
            torch.as_tensor(puf.weight_array[:, :-1], dtype=torch.float32)
        )
        model.chain_logits.bias.copy_(
            torch.as_tensor(puf.weight_array[:, -1], dtype=torch.float32)
        )
    attack_model = XorTorchPUFModel(
        [model], 8, response_bits, torch.device("cpu")
    )
    np.testing.assert_array_equal(attack_model.eval(challenges), responses)

    trainer = XorAgentTrainer(
        challenges,
        responses,
        response_bits,
        hidden_layers=2,
        width=16,
        seed=0,
        batch_size=8,
        learning_rate=0.001,
        device=torch.device("cpu"),
        chain_count=puf.k,
    )
    assert sum(
        isinstance(layer, torch.nn.Linear)
        for layer in trainer.model.hidden_stack
    ) == 2
    trainer.train_epoch()
    assert trainer.model(
        torch.as_tensor(xor_feature_map(challenges), dtype=torch.float32)
    ).shape == (len(challenges), response_bits)


def test_xor_parity_surrogate_prevents_chain_gradient_collapse():
    import torch

    model = _XorParityModel(
        feature_count=4,
        chain_count=4,
        seed=19,
        response_bits=1,
        hidden_layers=0,
    )
    with torch.no_grad():
        model.chain_logits.weight.zero_()
        model.chain_logits.bias.fill_(0.01)

    scores = model(torch.zeros(2, 4))
    scores.sum().backward()

    assert scores.abs().max().item() < 1e-6
    assert model.chain_logits.bias.grad is not None
    assert torch.all(model.chain_logits.bias.grad.abs() > 0.5)


def test_xor_model_has_vectorized_chain_outputs_and_independent_residuals():
    import torch

    feature_count = 8
    model = _XorParityModel(
        feature_count,
        chain_count=7,
        seed=19,
        response_bits=3,
        hidden_layers=2,
        width=12,
    )
    features = torch.randn(5, feature_count)
    assert model.chain_logits.out_features == 7
    assert model.hidden_stack[0].in_features == feature_count

    with torch.no_grad():
        model.chain_logits.weight.zero_()
        model.chain_logits.bias.fill_(0.2)
    before = model(features).detach()
    with torch.no_grad():
        model.linear_skip.bias[0] = 1
    after = model(features).detach()

    assert not torch.equal(before[:, 0], after[:, 0])
    torch.testing.assert_close(before[:, 1:], after[:, 1:])


def test_structured_attack_trainers_use_requested_hidden_layers():
    import torch

    challenges = np.random.default_rng(21).choice(
        (-1, 1), size=(16, 12)
    ).astype(np.int8)

    feed_forward_puf = create_puf("ff_apuf", n=12, seed=22)
    feed_forward = FeedForwardAgentTrainer(
        challenges,
        feed_forward_puf.eval(challenges),
        response_bits=1,
        hidden_layers=2,
        width=10,
        seed=23,
        batch_size=8,
        learning_rate=0.001,
        device=torch.device("cpu"),
    )
    assert sum(
        isinstance(layer, torch.nn.Linear)
        for layer in feed_forward.model.hidden_a
    ) == 2
    assert sum(
        isinstance(layer, torch.nn.Linear)
        for layer in feed_forward.model.hidden_b
    ) == 2
    feed_forward.train_epoch()

    interpose_puf = create_puf("interpose", n=12, k=3, seed=24)
    interpose = InterposeAgentTrainer(
        challenges,
        interpose_puf.eval(challenges),
        response_bits=1,
        hidden_layers=2,
        width=10,
        seed=25,
        batch_size=8,
        learning_rate=0.001,
        device=torch.device("cpu"),
        down_chain_count=interpose_puf.down.k,
        interpose_pos=interpose_puf.interpose_pos,
    )
    assert sum(
        isinstance(layer, torch.nn.Linear)
        for layer in interpose.model.up_hidden
    ) == 2
    assert sum(
        isinstance(layer, torch.nn.Linear)
        for layer in interpose.model.down_hidden
    ) == 2
    interpose.train_epoch()


def test_xor_experiment_tries_all_depths(monkeypatch):
    import torch

    monkeypatch.setattr(
        experiment,
        "_resolve_device",
        lambda backend, device: torch.device("cpu"),
    )
    monkeypatch.setattr(experiment, "_pair_accuracy", lambda model, dataset: 0.5)
    result = experiment.run_ml_attack_experiment(
        families=["xor_apuf"],
        n=8,
        training_samples=8,
        validation_samples=8,
        test_samples=8,
        seed=26,
        k=4,
        max_depth=2,
        agents=1,
        width=8,
        epochs=1,
        batch_size=8,
        backend="cpu",
        attack_model="neural",
        save_report=False,
        puf_instances=1,
    )["xor_apuf"]["instances"][0]

    assert [stage["depth"] for stage in result["stages"]] == [1, 2]


def test_xor_family_auto_routes_to_quadratic_model(monkeypatch):
    recorded_models = []

    def fake_attack(*args, **kwargs):
        recorded_models.append(args[-1])
        return {
            "attack": "xor_arbiter_pairwise_quadratic",
            "model_family": "xor_apuf",
            "model_architecture": args[-1],
            "activation_types": {},
            "depth": 0,
            "agents": 1,
            "validation_accuracy": 1.0,
            "validation_element_accuracy": 1.0,
            "test_accuracy": 1.0,
            "test_element_accuracy": 1.0,
            "matched_test_responses": 1,
            "test_response_count": 1,
            "success_threshold": 0.95,
            "success": True,
            "threshold_reached": True,
            "early_stopped": False,
            "early_stop_reason": None,
            "epochs_to_threshold": 0,
            "time_to_threshold_seconds": 0.0,
            "epochs_trained": 0,
            "agent_epochs_trained": 0,
            "epochs_per_depth_limit": 1,
            "training_samples": 1,
            "validation_samples": 1,
            "test_samples": 1,
            "fit_elapsed_seconds": 0.0,
            "stages": [],
            "_response_rows": (
                np.ones((1, 1), dtype=np.uint8),
                np.ones((1, 1), dtype=np.uint8),
                np.array([True]),
            ),
        }

    monkeypatch.setattr(experiment, "_resolve_device", lambda backend, device: "cpu")
    monkeypatch.setattr(experiment, "_attack_one_puf", fake_attack)

    result = experiment.run_ml_attack_experiment(
        families=["xor-apuf"],
        n=8,
        training_samples=1,
        validation_samples=1,
        test_samples=1,
        epochs=1,
        batch_size=1,
        save_report=False,
        puf_instances=1,
    )["xor-apuf"]["instances"][0]

    assert recorded_models == ["xor_rank_two"]
    assert result["model_architecture"] == "xor_rank_two"


def test_auto_routes_arbiter_families_by_architecture():
    assert experiment._resolve_attack_model_for_family("arbiter", "auto") == (
        "logistic_regression"
    )
    assert experiment._resolve_attack_model_for_family("ff_apuf", "auto") == (
        "deep_neural"
    )
    assert experiment._resolve_attack_model_for_family("xor_apuf", "auto") == (
        "xor_rank_two"
    )
    assert experiment._resolve_attack_model_for_family(
        "xor_apuf", "auto", n=64, k=128, response_bits=64
    ) == "xor_rank_two"
    assert experiment._resolve_attack_model_for_family(
        "xor_apuf", "auto", n=64, k=4, response_bits=1
    ) == "neural"
    assert experiment._resolve_attack_model_for_family("interpose", "auto") == (
        "interpose_logistic_regression"
    )
    assert experiment._resolve_attack_model_for_family(
        "interpose", "staged_split"
    ) == "staged_split"
    assert experiment._resolve_attack_model_for_family(
        "permutation", "auto", n=128, k=256, response_bits=128
    ) == "permutation_parity"
    assert experiment._resolve_attack_model_for_family(
        "xor_apuf", "auto", n=64, k=2, response_bits=64
    ) == "xor_logistic_regression"
    assert experiment._resolve_attack_model_for_family("arbiter", "neural") == (
        "neural"
    )
    assert experiment._resolve_attack_model_for_family("xor_apuf", "svm") == "svm"


def test_auto_routes_sram_based_on_hash_setting():
    assert experiment._resolve_attack_model_for_family(
        "sram", "auto", hash_responses=True
    ) == "sram_hash_bruteforce"
    assert experiment._resolve_attack_model_for_family(
        "sram", "auto", hash_responses=False
    ) == "sram_logistic_regression"


def test_ring_oscillator_logistic_regression_learns_raw_challenge_bits():
    rng = np.random.default_rng(902)
    challenges = rng.choice((-1, 1), size=(256, 8)).astype(np.int8)
    responses = np.column_stack((
        challenges[:, 0],
        challenges[:, 1],
        np.where(challenges[:, :3].sum(axis=1) >= 0, 1, -1),
    )).astype(np.int8)
    model = RingOscillatorLogisticRegression(
        challenges,
        responses,
        response_bits=3,
        seed=903,
    )

    assert model.challenge_length == 8
    assert model.response_length == 3
    np.testing.assert_array_equal(model.eval(challenges), responses)


def test_sram_hash_bruteforce_search_finds_candidate_fingerprint():
    seed = 331
    rng = np.random.default_rng(seed)
    fingerprint = rng.integers(0, 2, size=8, dtype=np.uint8).astype(bool)
    training_bits = rng.integers(0, 2, size=(128, 8), dtype=np.uint8).astype(bool)
    challenges = np.where(training_bits, 1, -1).astype(np.int8)
    responses = hash_sram_responses(
        np.logical_xor(training_bits, fingerprint),
        response_bits=8,
    )

    model = SRAMHashBruteForceAttack(
        challenges,
        responses,
        response_bits=8,
        seed=seed,
    )

    assert model.candidate_limit == 100
    assert model.training_match_count == len(challenges)
    np.testing.assert_array_equal(model.eval(challenges), responses)


def test_auto_uses_bounded_bruteforce_for_hashed_sram():
    instance = experiment.run_ml_attack_experiment(
        families=["sram"],
        n=8,
        training_samples=16,
        validation_samples=8,
        test_samples=8,
        seed=332,
        response_bits=8,
        noisiness=0.0,
        backend="cpu",
        save_report=False,
        puf_instances=1,
        attack_model="auto",
        hash_responses=True,
    )["sram"]["instances"][0]

    assert instance["model_architecture"] == "sram_hash_bruteforce"
    assert instance["attack"] == "sram_hash_random_fingerprint_search"
    assert instance["candidate_fingerprints_tested"] == 100
    assert instance["training_response_count"] == 16


def test_auto_uses_sram_logistic_model_for_unhashed_responses(tmp_path):
    result = experiment.run_ml_attack_experiment(
        families=["sram"],
        n=16,
        training_samples=256,
        validation_samples=64,
        test_samples=64,
        seed=20261019,
        response_bits=16,
        noisiness=0.0,
        success_threshold=1.0,
        backend="cpu",
        scenario="unhashed sram test",
        report_root=tmp_path,
        save_report=True,
        puf_instances=1,
        attack_model="auto",
        hash_responses=False,
    )["sram"]
    instance = result["instances"][0]

    assert instance["model_architecture"] == "sram_logistic_regression"
    assert instance["attack"] == "sram_direct_logistic_regression"
    assert instance["validation_accuracy"] == 1.0
    assert instance["test_accuracy"] == 1.0
    assert instance["test_element_accuracy"] == 1.0
    assert result["mean_test_element_accuracy"] == 1.0
    parameters_path = next(tmp_path.rglob("parameters.json"))
    with parameters_path.open(encoding="utf-8") as parameters_file:
        parameters = json.load(parameters_file)
    assert parameters["hash_responses"] is False


def test_auto_uses_arbiter_logistic_for_single_chain_xor_outputs(capsys):
    result = experiment.run_ml_attack_experiment(
        families=["xor_apuf"],
        n=8,
        training_samples=64,
        validation_samples=16,
        test_samples=16,
        seed=327,
        k=2,
        response_bits=8,
        success_threshold=1.0,
        max_depth=1,
        agents=1,
        width=8,
        epochs=1,
        batch_size=16,
        backend="cpu",
        save_report=False,
        puf_instances=1,
        attack_model="auto",
        print_progress=True,
    )["xor_apuf"]["instances"][0]

    output = capsys.readouterr().out
    assert (
        experiment.CLEAR_LINE
        + "Depth 0, trainers 1, epoch 1/1, validation accuracy "
        in output
    )
    assert (
        " | maximum epochs reached" in output
    )
    assert result["model_architecture"] == "xor_logistic_regression"
    assert result["attack"] == "xor_arbiter_per_bit_logistic_regression"
    assert result["activation_types"]["estimator"] == (
        "PyTorch logistic regression per response bit"
    )


def test_auto_ff_apuf_experiment_uses_structure_aware_deep_neural(tmp_path):
    result = experiment.run_ml_attack_experiment(
        families=["ff_apuf"],
        n=8,
        training_samples=16,
        validation_samples=8,
        test_samples=8,
        seed=328,
        max_depth=1,
        agents=1,
        width=8,
        epochs=1,
        batch_size=8,
        backend="cpu",
        report_root=str(tmp_path),
        save_report=True,
        puf_instances=1,
        attack_model="auto",
    )["ff_apuf"]["instances"][0]

    assert result["model_architecture"] == "deep_neural"
    assert result["attack"] == "structure_aware_deep_neural_ff_apuf"
    epoch = result["stages"][0]["epoch_history"][0]
    assert epoch["epoch"] == 1
    assert 0.0 <= epoch["validation_accuracy"] <= 1.0
    assert 0.0 <= epoch["validation_element_accuracy"] <= 1.0
    assert epoch["validation_emas"] == [epoch["validation_accuracy"]]
    assert epoch["learning_rates"] == [0.001]
    assert epoch["lr_events"] == []
    report_path = next(tmp_path.rglob("results.json"))
    with report_path.open(encoding="utf-8") as report_file:
        saved_result = json.load(report_file)
    saved_epoch = saved_result["ff_apuf"]["instances"][0]["stages"][0][
        "epoch_history"
    ][0]
    assert saved_epoch == epoch


def test_auto_interpose_uses_structure_aware_logistic_model(capsys):
    result = experiment.run_ml_attack_experiment(
        families=["interpose"],
        n=8,
        training_samples=32,
        validation_samples=8,
        test_samples=8,
        seed=329,
        k=2,
        response_bits=1,
        success_threshold=1.0,
        max_depth=1,
        agents=1,
        width=8,
        epochs=1,
        batch_size=8,
        backend="cpu",
        save_report=False,
        puf_instances=1,
        attack_model="auto",
        print_progress=True,
    )["interpose"]["instances"][0]

    output = capsys.readouterr().out
    assert (
        experiment.CLEAR_LINE
        + "Depth 0, trainers 1, epoch 1/1, validation accuracy "
        in output
    )
    assert " | maximum epochs reached" in output
    assert result["model_architecture"] == "interpose_logistic_regression"
    assert result["attack"] == "interpose_structure_aware_logistic_regression"
    assert result["activation_types"]["hidden_layers"] == []


def test_staged_split_interpose_runs_alternating_logistic_fit():
    result = experiment.run_ml_attack_experiment(
        families=["interpose"],
        n=8,
        training_samples=32,
        validation_samples=8,
        test_samples=8,
        seed=330,
        k=2,
        response_bits=1,
        success_threshold=1.0,
        max_depth=1,
        agents=1,
        width=8,
        epochs=1,
        batch_size=8,
        backend="cpu",
        save_report=False,
        puf_instances=1,
        attack_model="staged_split",
    )["interpose"]["instances"][0]

    assert result["model_architecture"] == "staged_split"
    assert result["attack"] == "interpose_staged_split_logistic_regression"
    assert "Alternating downstream and upstream logistic fitting" in (
        result["activation_types"]["internal_layers"]
    )


def test_deep_neural_xor_experiment_records_learning_rate_history(tmp_path):
    result = experiment.run_ml_attack_experiment(
        families=["xor_apuf"],
        n=8,
        training_samples=16,
        validation_samples=8,
        test_samples=8,
        seed=333,
        k=2,
        response_bits=1,
        max_depth=1,
        agents=1,
        width=8,
        epochs=1,
        batch_size=8,
        backend="cpu",
        report_root=str(tmp_path),
        save_report=True,
        puf_instances=1,
        attack_model="deep_neural",
    )["xor_apuf"]["instances"][0]

    assert result["model_architecture"] == "deep_neural"
    epoch = result["stages"][0]["epoch_history"][0]
    assert epoch["epoch"] == 1
    assert 0.0 <= epoch["validation_accuracy"] <= 1.0
    assert epoch["validation_emas"] == [epoch["validation_accuracy"]]
    assert epoch["learning_rates"] == [0.001]
    assert epoch["lr_events"] == []
    report_path = next(tmp_path.rglob("results.json"))
    with report_path.open(encoding="utf-8") as report_file:
        saved_result = json.load(report_file)
    saved_epoch = saved_result["xor_apuf"]["instances"][0]["stages"][0][
        "epoch_history"
    ][0]
    assert saved_epoch == epoch


def test_ff_learning_rate_reduces_on_plateau_to_floor():
    import torch

    challenges = np.array([[-1, 1], [1, -1]], dtype=np.int8)
    responses = np.array([[-1, 1], [1, -1]], dtype=np.int8)
    trainer = DeepNeuralAgentTrainer(
        challenges,
        responses,
        response_bits=2,
        hidden_layers=1,
        width=4,
        seed=1,
        batch_size=2,
        learning_rate=4e-4,
        device=torch.device("cpu"),
        family_key="ff_apuf",
        feed_forward_loops=[(0, 1)],
    )

    for epoch in range(1, 18):
        trainer._epochs_trained = epoch
        trainer.update_validation_feedback(0.8)

    assert trainer.learning_rate == 1e-4
    assert [event["epoch"] for event in trainer.consume_lr_events()] == [
        9.0,
        17.0,
    ]


def test_ff_learning_rate_does_not_reduce_during_validation_improvement():
    import torch

    challenges = np.array([[-1, 1], [1, -1]], dtype=np.int8)
    responses = np.array([[-1, 1], [1, -1]], dtype=np.int8)
    trainer = DeepNeuralAgentTrainer(
        challenges,
        responses,
        response_bits=2,
        hidden_layers=1,
        width=4,
        seed=2,
        batch_size=2,
        learning_rate=4e-4,
        device=torch.device("cpu"),
        family_key="ff_apuf",
        feed_forward_loops=[(0, 1)],
    )

    for epoch in range(1, 21):
        trainer._epochs_trained = epoch
        trainer.update_validation_feedback(0.8 + 0.002 * (epoch - 1))

    assert trainer.learning_rate == 4e-4
    assert trainer.consume_lr_events() == []


def test_xor_learning_rate_reduces_on_plateau_to_floor():
    import torch

    challenges = np.array([[-1, 1], [1, -1]], dtype=np.int8)
    responses = np.array([[-1], [1]], dtype=np.int8)
    trainer = DeepNeuralAgentTrainer(
        challenges,
        responses,
        response_bits=1,
        hidden_layers=1,
        width=4,
        seed=3,
        batch_size=2,
        learning_rate=4e-4,
        device=torch.device("cpu"),
        family_key="xor_apuf",
        chain_count=2,
    )

    for epoch in range(1, 18):
        trainer._epochs_trained = epoch
        trainer.update_validation_feedback(0.8)

    assert trainer.learning_rate == 1e-4
    assert [event["epoch"] for event in trainer.consume_lr_events()] == [
        9.0,
        17.0,
    ]


def test_permutation_learning_rate_reduces_on_plateau_to_floor():
    import torch

    challenges = np.array([[-1, 1], [1, -1]], dtype=np.int8)
    responses = np.array([[-1], [1]], dtype=np.int8)
    trainer = experiment._permutation.AgentTrainer(
        challenges,
        responses,
        response_bits=1,
        hidden_layers=1,
        width=4,
        seed=4,
        batch_size=2,
        learning_rate=4e-4,
        device=torch.device("cpu"),
        permutation_table=np.array([[0, 1]]),
    )

    for epoch in range(1, 18):
        trainer._epochs_trained = epoch
        trainer.update_validation_feedback(0.8)

    assert trainer.learning_rate == 1e-4
    assert [event["epoch"] for event in trainer.consume_lr_events()] == [
        9.0,
        17.0,
    ]


def test_auto_permutation_experiment_records_learning_rate_history(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        experiment,
        "_pair_accuracy",
        lambda model, dataset: 0.75,
    )
    result = experiment.run_ml_attack_experiment(
        families=["permutation"],
        n=16,
        k=1,
        response_bits=1,
        training_samples=8,
        validation_samples=8,
        test_samples=8,
        seed=446,
        success_threshold=1.0,
        max_depth=1,
        agents=1,
        width=8,
        epochs=9,
        batch_size=4,
        learning_rate=0.001,
        backend="cpu",
        report_root=str(tmp_path),
        save_report=True,
        puf_instances=1,
        attack_model="auto",
        print_progress=False,
    )["permutation"]["instances"][0]

    assert result["model_architecture"] == "permutation_parity"
    history = result["stages"][0]["epoch_history"]
    assert len(history) == 9
    epoch = history[-1]
    assert epoch["epoch"] == 9
    assert epoch["validation_emas"] == [0.75]
    assert epoch["learning_rates"] == [0.0005]
    assert epoch["lr_events"] == [{
        "epoch": 9.0,
        "ema": 0.75,
        "lr": 0.0005,
    }]
    report_path = next(tmp_path.rglob("results.json"))
    with report_path.open(encoding="utf-8") as report_file:
        saved_result = json.load(report_file)
    saved_epoch = saved_result["permutation"]["instances"][0]["stages"][0][
        "epoch_history"
    ]
    assert saved_epoch == history


def test_feed_forward_aware_fc_bilstm_trains_and_predicts():
    import torch

    puf = create_puf("ff_apuf", n=8, seed=329)
    rng = np.random.default_rng(330)
    challenges = rng.choice((-1, 1), size=(16, 8)).astype(np.int8)
    responses = puf.eval(challenges)
    trainer = FCBiLSTMTrainer(
        challenges,
        responses,
        response_bits=1,
        hidden_layers=1,
        width=8,
        seed=331,
        batch_size=8,
        learning_rate=0.001,
        device=torch.device("cpu"),
        feed_forward_loop=tuple(puf.ff[0]),
    )

    trainer.train_epoch()
    predictions = TorchFCBiLSTMModel(
        [trainer.model],
        challenge_length=8,
        response_bits=1,
        device=torch.device("cpu"),
        inference_batch_size=8,
    ).eval(challenges)

    assert predictions.shape == (16, 1)
    assert set(np.unique(predictions)).issubset({-1, 1})


def test_arbiter_logistic_regression_learns_att_response():
    puf = create_puf("arbiter", n=8, seed=321)
    rng = np.random.default_rng(322)
    training_challenges = rng.choice((-1, 1), size=(1024, 8)).astype(np.int8)
    test_challenges = rng.choice((-1, 1), size=(256, 8)).astype(np.int8)
    model = ArbiterLogisticRegression(
        training_challenges,
        puf.eval(training_challenges),
        response_bits=1,
        seed=323,
    )

    predictions = model.eval(test_challenges)

    assert predictions.shape == (256, 1)
    assert set(np.unique(predictions)).issubset({-1, 1})
    assert np.mean(
        predictions.reshape(-1) == puf.eval(test_challenges).reshape(-1)
    ) > 0.95


def test_auto_arbiter_experiment_uses_logistic_regression():
    result = experiment.run_ml_attack_experiment(
        families=["arbiter"],
        n=8,
        training_samples=1024,
        validation_samples=128,
        test_samples=128,
        seed=330,
        success_threshold=0.6,
        max_depth=1,
        agents=1,
        epochs=1,
        batch_size=32,
        backend="cpu",
        save_report=False,
        puf_instances=1,
    )["arbiter"]["instances"][0]

    assert result["model_architecture"] == "logistic_regression"
    assert result["attack"] == "arbiter_logistic_regression"
    assert result["epochs_trained"] == 0
    assert result["test_accuracy"] > 0.95


def test_fc_bilstm_trains_on_challenge_sequences():
    import torch

    puf = create_puf("xor_apuf", n=8, k=2, seed=324)
    rng = np.random.default_rng(325)
    challenges = rng.choice((-1, 1), size=(24, 8)).astype(np.int8)
    trainer = FCBiLSTMTrainer(
        challenges,
        puf.eval(challenges),
        response_bits=1,
        hidden_layers=1,
        width=8,
        seed=326,
        batch_size=8,
        learning_rate=0.01,
        device=torch.device("cpu"),
    )
    trainer.train_epoch()
    model = TorchFCBiLSTMModel(
        [trainer.model],
        challenge_length=8,
        response_bits=1,
        device=torch.device("cpu"),
        inference_batch_size=5,
    )

    predictions = model.eval(challenges)

    assert predictions.shape == (24, 1)
    assert set(np.unique(predictions)).issubset({-1, 1})


def test_fc_bilstm_inference_batch_matches_configured_batch_size(monkeypatch):
    import torch

    inference_batch_sizes = []
    model_type = experiment._fc_lstm.TorchFCBiLSTMModel

    class RecordingFCBiLSTMModel(model_type):
        def __init__(self, *args, **kwargs):
            inference_batch_sizes.append(kwargs["inference_batch_size"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        experiment._fc_lstm,
        "TorchFCBiLSTMModel",
        RecordingFCBiLSTMModel,
    )
    experiment._attack_one_puf(
        target=create_puf("ff_apuf", n=4, seed=330),
        n=4,
        training_samples=8,
        validation_samples=4,
        test_samples=4,
        seed=331,
        success_threshold=1.0,
        max_depth=1,
        agents=1,
        width=4,
        epochs=1,
        batch_size=3,
        learning_rate=0.001,
        device=torch.device("cpu"),
        family="ff_apuf",
        attack_model="fc_lstm",
    )

    assert inference_batch_sizes
    assert set(inference_batch_sizes) == {3}


def test_auto_xor_experiment_trains_rank_two_model():
    for attack_model in ("auto", "neural"):
        result = experiment.run_ml_attack_experiment(
            families=["xor_apuf"],
            n=8,
            k=2,
            training_samples=32,
            validation_samples=16,
            test_samples=16,
            seed=331,
            max_depth=1,
            agents=1,
            width=8,
            epochs=1,
            batch_size=8,
            backend="cpu",
            save_report=False,
            puf_instances=1,
            attack_model=attack_model,
        )["xor_apuf"]["instances"][0]

        assert result["model_architecture"] == "xor_rank_two"
        assert result["attack"] == "xor_arbiter_rank_two_factorized"
        assert result["activation_types"]["internal_layers"] == [
            "Two learned Arbiter delay scores per response bit",
            "Product of the paired chain scores",
        ]
        assert result["epochs_trained"] == 1


def test_permutation_chain_model_matches_permuted_arbiter_chains():
    import torch

    puf = create_puf(
        "permutation",
        n=16,
        k=4,
        response_bits=2,
        seed=441,
        noisiness=0.0,
    )
    challenges = np.random.default_rng(442).choice(
        (-1, 1), size=(64, 16)
    ).astype(np.int8)
    model = experiment._permutation._PermutationChainModel(
        16,
        puf._permutation_table,
        puf.response_length,
        seed=443,
    )
    with torch.no_grad():
        model.chain_weights.copy_(
            torch.as_tensor(puf.weight_array[:, :-1], dtype=torch.float32)
        )
        model.chain_biases.copy_(
            torch.as_tensor(puf.weight_array[:, -1], dtype=torch.float32)
        )
    attack_model = experiment._TorchPUFModel(
        [model], 16, puf.response_length, torch.device("cpu")
    )

    np.testing.assert_array_equal(attack_model.eval(challenges), puf.eval(challenges))


def test_auto_permutation_experiment_trains_structured_model():
    result = experiment.run_ml_attack_experiment(
        families=["permutation"],
        n=16,
        k=4,
        response_bits=2,
        training_samples=5000,
        validation_samples=1000,
        test_samples=1000,
        seed=445,
        success_threshold=0.9,
        max_depth=1,
        agents=1,
        width=8,
        epochs=200,
        batch_size=256,
        learning_rate=0.01,
        backend="cpu",
        save_report=False,
        puf_instances=1,
        attack_model="auto",
    )["permutation"]["instances"][0]

    assert result["model_architecture"] == "permutation_parity"
    assert result["attack"] == "permutation_chain_parity"
    assert result["activation_types"]["feature_mapping"] == [
        "Per-chain fixed challenge permutation",
        "Arbiter Threshold Transform per chain",
    ]
    assert result["validation_accuracy"] >= 0.85
    assert result["test_accuracy"] >= 0.85


def test_xor_model_selection_uses_element_accuracy_to_break_row_ties(monkeypatch):
    element_accuracies = iter((0.6, 0.8))
    monkeypatch.setattr(experiment, "_pair_accuracy", lambda model, dataset: 0.0)
    monkeypatch.setattr(
        experiment,
        "_response_element_accuracy",
        lambda model, dataset, family: next(element_accuracies),
    )

    result = experiment.run_ml_attack_experiment(
        families=["xor_apuf"],
        n=8,
        k=2,
        training_samples=32,
        validation_samples=16,
        test_samples=16,
        seed=332,
        success_threshold=1.0,
        agents=1,
        epochs=2,
        batch_size=8,
        backend="cpu",
        save_report=False,
        puf_instances=1,
        attack_model="auto",
    )["xor_apuf"]["instances"][0]

    assert result["validation_accuracy"] == 0.0
    assert result["validation_element_accuracy"] == 0.8
    assert result["stages"][0]["validation_element_accuracy"] == 0.8
    assert result["epochs_trained"] == 2


def test_xor_agent_selection_uses_element_accuracy_to_break_row_ties(monkeypatch):
    element_accuracies = iter((0.4, 0.6, 0.8))
    monkeypatch.setattr(experiment, "_pair_accuracy", lambda model, dataset: 0.0)
    monkeypatch.setattr(
        experiment,
        "_response_element_accuracy",
        lambda model, dataset, family: next(element_accuracies),
    )

    result = experiment.run_ml_attack_experiment(
        families=["xor_apuf"],
        n=8,
        k=2,
        training_samples=32,
        validation_samples=16,
        test_samples=16,
        seed=333,
        success_threshold=1.0,
        agents=2,
        epochs=1,
        batch_size=8,
        backend="cpu",
        save_report=False,
        puf_instances=1,
        attack_model="auto",
    )["xor_apuf"]["instances"][0]

    assert result["validation_accuracy"] == 0.0
    assert result["validation_element_accuracy"] == 0.8
    assert result["agents"] == 1


def test_optical_experiment_uses_closed_form_linear_regression(monkeypatch):
    import torch

    bootstrap = experiment._optical.bootstrap_linear_model
    bootstrap_calls = []

    def record_bootstrap(*args, **kwargs):
        bootstrap_calls.append((args, kwargs))
        return bootstrap(*args, **kwargs)

    def trainer_must_not_run(*args, **kwargs):
        raise AssertionError("Optical attacks should use closed-form regression.")

    def svm_must_not_run(*args, **kwargs):
        raise AssertionError("Optical attacks should use their structural model.")

    monkeypatch.setattr(experiment._optical, "bootstrap_linear_model", record_bootstrap)
    monkeypatch.setattr(experiment._optical, "AgentTrainer", trainer_must_not_run)
    monkeypatch.setattr(experiment._svm, "FeatureMappedSVM", svm_must_not_run)
    monkeypatch.setattr(
        experiment,
        "_resolve_device",
        lambda backend, device: torch.device("cpu"),
    )
    result = experiment.run_ml_attack_experiment(
        families=["optical"],
        n=8,
        training_samples=128,
        validation_samples=32,
        test_samples=32,
        seed=27,
        max_depth=2,
        agents=1,
        width=8,
        epochs=1,
        batch_size=8,
        backend="cpu",
        save_report=False,
        puf_instances=1,
        attack_model="svm",
    )["optical"]["instances"][0]

    assert len(bootstrap_calls) == 1
    assert result["attack"] == "optical_linear_regression"
    assert result["model_family"] == "optical"
    assert result["model_architecture"] == "optical_regression"
    assert result["activation_types"] == {
        "hidden_layers": [],
        "internal_layers": [],
        "output_layer": "Linear regression",
        "prediction": "Continuous output",
    }
    assert result["depth"] == 0
    assert result["epochs_trained"] == 0
    assert result["threshold_reached"] is True
    assert result["validation_accuracy"] == 1.0
    assert result["test_accuracy"] == 1.0
    assert result["epochs_to_threshold"] == 0
    assert [stage["depth"] for stage in result["stages"]] == [0]


def test_optical_regression_scales_to_high_dimensional_challenges(monkeypatch):
    import torch

    puf = create_puf("optical", n=128, seed=28)
    rng = np.random.default_rng(29)
    training_challenges = rng.choice(
        (-1, 1), size=(20_000, 128)
    ).astype(np.int8)
    validation_challenges = rng.choice(
        (-1, 1), size=(1024, 128)
    ).astype(np.int8)
    training_responses = puf.eval(training_challenges)
    validation_responses = puf.eval(validation_challenges)

    def pseudoinverse_must_not_run(*args, **kwargs):
        raise AssertionError("Optical fitting must not build a dense pseudoinverse.")

    monkeypatch.setattr(np.linalg, "pinv", pseudoinverse_must_not_run)
    linear_model = experiment._optical.bootstrap_linear_model(
        training_challenges,
        training_responses,
        response_bits=1,
        device=torch.device("cpu"),
    )
    model = experiment._optical.TorchPUFModel(
        [linear_model], 128, 1, torch.device("cpu")
    )

    predicted = model.eval(validation_challenges)
    assert np.isclose(
        predicted, validation_responses, rtol=0.10, atol=0.10
    ).all()


def test_interpose_model_matches_simulator_with_target_weights():
    import torch

    puf = create_puf("interpose", n=12, seed=27, k=2, interpose_pos=5)
    challenges = np.random.default_rng(28).choice((-1, 1), size=(32, 12)).astype(np.int8)
    model = InterposeCore(
        12, puf.down.k, puf.interpose_pos, seed=0, hidden_layers=0
    )
    with torch.no_grad():
        model.up_weights.weight.copy_(
            torch.as_tensor(puf.up.weight_array[:, :-1], dtype=torch.float32)
        )
        model.up_weights.bias.copy_(
            torch.as_tensor(puf.up.weight_array[:, -1], dtype=torch.float32)
        )
        model.down_weights.weight.copy_(
            torch.as_tensor(puf.down.weight_array[:, :-1], dtype=torch.float32)
        )
        model.down_weights.bias.copy_(
            torch.as_tensor(puf.down.weight_array[:, -1], dtype=torch.float32)
        )
    attack_model = InterposeTorchPUFModel([model], 12, 1, torch.device("cpu"))

    assert np.array_equal(
        attack_model.eval(challenges).reshape(-1), puf.eval(challenges).reshape(-1)
    )


def test_interpose_parity_gradient_does_not_vanish_for_large_groups():
    import torch

    model = InterposeCore(
        n=64,
        down_chain_count=64,
        interpose_pos=32,
        seed=30,
        hidden_layers=0,
        response_bits=10,
    )
    with torch.no_grad():
        model.down_weights.weight.zero_()
        model.down_weights.bias.zero_()

    model(torch.zeros((2, 64), dtype=torch.float32)).sum().backward()

    assert model.down_weights.bias.grad is not None
    assert torch.isfinite(model.down_weights.bias.grad).all()
    assert torch.all(model.down_weights.bias.grad.abs() > 0)


def test_interpose_training_marginalizes_upstream_signs_and_anneals_gate():
    import torch

    model = InterposeCore(
        n=2,
        down_chain_count=1,
        interpose_pos=1,
        seed=30,
        hidden_layers=0,
    )
    with torch.no_grad():
        model.up_weights.weight.zero_()
        model.up_weights.bias.fill_(1.0)
        model.down_weights.weight.zero_()
        model.down_weights.weight[0, 1] = 1.0
        model.down_weights.bias.zero_()

    challenges = torch.ones((1, 2), dtype=torch.float32)
    model.train()
    mixture_prediction = model(challenges)
    mixture_prediction.sum().backward()
    assert model.up_weights.bias.grad is not None
    assert torch.all(model.up_weights.bias.grad.abs() > 0)
    model.eval()
    hard_prediction = model(challenges)

    assert not torch.allclose(mixture_prediction, hard_prediction)

    puf = create_puf("interpose", n=8, seed=30, k=2)
    training_challenges = np.random.default_rng(31).choice(
        (-1, 1), size=(8, puf.challenge_length)
    ).astype(np.int8)
    trainer = InterposeLogisticAgentTrainer(
        training_challenges,
        puf.eval(training_challenges),
        response_bits=1,
        hidden_layers=0,
        width=8,
        seed=32,
        batch_size=4,
        learning_rate=0.001,
        device=torch.device("cpu"),
        down_chain_count=puf.down.k,
        interpose_pos=puf.interpose_pos,
    )
    initial_temperature = trainer.model.interpose_temperature
    trainer.train_epoch()
    trainer.train_epoch()

    assert trainer.model.interpose_temperature < initial_temperature


def test_interpose_multibit_checkpoint_selection_prioritizes_element_accuracy():
    interpose_key = experiment._validation_selection_key(
        "interpose",
        response_bits=10,
        exact_accuracy=0.80,
        element_accuracy=0.75,
    )
    competing_interpose_key = experiment._validation_selection_key(
        "interpose",
        response_bits=10,
        exact_accuracy=0.85,
        element_accuracy=0.70,
    )
    arbiter_key = experiment._validation_selection_key(
        "arbiter",
        response_bits=10,
        exact_accuracy=0.80,
        element_accuracy=0.75,
    )
    competing_arbiter_key = experiment._validation_selection_key(
        "arbiter",
        response_bits=10,
        exact_accuracy=0.85,
        element_accuracy=0.70,
    )

    assert interpose_key > competing_interpose_key
    assert arbiter_key < competing_arbiter_key


def test_interpose_neural_attacks_train_with_ten_response_bits():
    import torch

    response_bits = 10
    puf = create_puf(
        "interpose",
        n=12,
        k=20,
        response_bits=response_bits,
        interpose_pos=5,
        seed=31,
    )
    challenges = np.random.default_rng(32).choice(
        (-1, 1), size=(8, puf.challenge_length)
    ).astype(np.int8)
    responses = puf.eval(challenges)
    assert responses.shape == (len(challenges), response_bits)

    exact_model = InterposeCore(
        puf.challenge_length,
        puf.down.k,
        puf.interpose_pos,
        seed=30,
        hidden_layers=0,
        response_bits=response_bits,
    )
    with torch.no_grad():
        exact_model.up_weights.weight.copy_(
            torch.as_tensor(puf.up.weight_array[:, :-1], dtype=torch.float32)
        )
        exact_model.up_weights.bias.copy_(
            torch.as_tensor(puf.up.weight_array[:, -1], dtype=torch.float32)
        )
        exact_model.down_weights.weight.copy_(
            torch.as_tensor(puf.down.weight_array[:, :-1], dtype=torch.float32)
        )
        exact_model.down_weights.bias.copy_(
            torch.as_tensor(puf.down.weight_array[:, -1], dtype=torch.float32)
        )
    exact_adapter = InterposeTorchPUFModel(
        [exact_model],
        puf.challenge_length,
        response_bits,
        torch.device("cpu"),
    )
    np.testing.assert_array_equal(exact_adapter.eval(challenges), responses)

    legacy_trainer = InterposeAgentTrainer(
        challenges,
        responses,
        response_bits=response_bits,
        hidden_layers=1,
        width=8,
        seed=33,
        batch_size=4,
        learning_rate=0.001,
        device=torch.device("cpu"),
        down_chain_count=puf.down.k,
        interpose_pos=puf.interpose_pos,
    )
    assert legacy_trainer.model(
        torch.as_tensor(challenges, dtype=torch.float32)
    ).shape == (len(challenges), response_bits)
    legacy_trainer.train_epoch()

    for trainer_type in (
        InterposeLogisticAgentTrainer,
        InterposeStagedSplitAgentTrainer,
    ):
        logistic_trainer = trainer_type(
            challenges,
            responses,
            response_bits=response_bits,
            hidden_layers=0,
            width=8,
            seed=35,
            batch_size=4,
            learning_rate=0.001,
            device=torch.device("cpu"),
            down_chain_count=puf.down.k,
            interpose_pos=puf.interpose_pos,
        )
        assert logistic_trainer.model(
            torch.as_tensor(challenges, dtype=torch.float32)
        ).shape == (len(challenges), response_bits)
        initial_temperature = logistic_trainer.model.interpose_temperature
        logistic_trainer.train_epoch()
        logistic_trainer.train_epoch()
        assert logistic_trainer.model.interpose_temperature < initial_temperature

    deep_trainer = DeepNeuralAgentTrainer(
        challenges,
        responses,
        response_bits=response_bits,
        hidden_layers=1,
        width=8,
        seed=34,
        batch_size=4,
        learning_rate=0.001,
        device=torch.device("cpu"),
        family_key="interpose",
        down_chain_count=puf.down.k,
        interpose_pos=puf.interpose_pos,
    )
    assert deep_trainer.model(
        torch.as_tensor(challenges, dtype=torch.float32)
    ).shape == (len(challenges), response_bits)
    initial_temperature = deep_trainer.model.interpose_temperature
    deep_trainer.train_epoch()
    deep_trainer.train_epoch()
    assert deep_trainer.model.interpose_temperature < initial_temperature


def test_train_until_threshold_reports_effort(monkeypatch, capsys):
    class DatasetFactory:
        @staticmethod
        def from_simulation(target, N, seed):
            return SimpleNamespace(
                challenges=np.zeros((N, 1)),
                responses=np.ones((N, 1)),
                seed=seed,
            )

    class AgentTrainer:
        def __init__(self, *args, **kwargs):
            self.model = FakeTorchModel(args[3])

        def train_epoch(self):
            pass

    class AttackModel:
        def __init__(self, models, n, response_bits, device):
            self.depth = models[0].depth

        def eval(self, challenges):
            predictions = np.ones((len(challenges), 1), dtype=np.int8)
            predictions[0] = -1
            return predictions

    validation_scores = iter([0.4, 0.8, 0.95])
    monkeypatch.setattr(experiment, "ChallengeResponseSet", DatasetFactory)
    monkeypatch.setattr(experiment, "_AgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_TorchPUFModel", AttackModel)
    monkeypatch.setattr(experiment, "_ArbiterAgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_ArbiterTorchPUFModel", AttackModel)
    monkeypatch.setattr(
        experiment,
        "_pair_accuracy",
        lambda model, dataset: next(validation_scores, 0.95)
        if dataset.seed == 11 else 0.98,
    )

    result = experiment._attack_one_puf(
        SimpleNamespace(response_length=1),
        n=1,
        training_samples=1,
        validation_samples=1,
        test_samples=50,
        seed=10,
        family="arbiter",
        success_threshold=0.95,
        max_depth=1,
        agents=1,
        width=None,
        epochs=3,
        batch_size=1,
        learning_rate=0.001,
        device=SimpleNamespace(type="cpu"),
        print_progress=True,
    )

    output = capsys.readouterr().out
    assert output.count(experiment.CLEAR_LINE) == 3
    assert (
        experiment.CLEAR_LINE
        + "Depth 1, trainers 1, epoch 3/3, validation accuracy 95.0000% "
        "| success threshold reached"
    ) in output
    assert result["threshold_reached"] is True
    assert result["epochs_to_threshold"] == 3
    assert result["epochs_trained"] == 3
    assert result["time_to_threshold_seconds"] >= 0
    assert result["test_accuracy"] == 0.98
    assert result["matched_test_responses"] == 49
    assert result["test_response_count"] == 50
    actual, predicted, matches = result["_response_rows"]
    assert actual.shape == predicted.shape == (50, 1)
    assert int(np.count_nonzero(matches)) == result["matched_test_responses"]
    assert "epoch 1/3, validation accuracy 40.0000%" in output
    assert "epoch 3/3, validation accuracy 95.0000%" in output


def test_epoch_limit_applies_at_each_depth(monkeypatch, capsys):
    class DatasetFactory:
        @staticmethod
        def from_simulation(target, N, seed):
            return SimpleNamespace(
                challenges=np.zeros((N, 1)),
                responses=np.ones((N, 1)),
                seed=seed,
            )

    class AgentTrainer:
        def __init__(self, *args, **kwargs):
            self.model = FakeTorchModel(args[3])

        def train_epoch(self):
            pass

    class AttackModel:
        def __init__(self, models, n, response_bits, device):
            self.depth = models[0].depth

        def eval(self, challenges):
            return np.ones((len(challenges), 1), dtype=np.int8)

    monkeypatch.setattr(experiment, "ChallengeResponseSet", DatasetFactory)
    monkeypatch.setattr(experiment, "_ArbiterAgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_ArbiterTorchPUFModel", AttackModel)
    monkeypatch.setattr(experiment, "_pair_accuracy", lambda model, dataset: 0.5)

    result = experiment._attack_one_puf(
        SimpleNamespace(response_length=1),
        n=1,
        training_samples=1,
        validation_samples=1,
        test_samples=1,
        seed=10,
        family="arbiter",
        success_threshold=1.0,
        max_depth=2,
        agents=2,
        width=None,
        epochs=3,
        batch_size=1,
        learning_rate=0.001,
        device=SimpleNamespace(type="cpu"),
        attack_model="neural",
        print_progress=True,
    )

    output = capsys.readouterr().out
    assert (
        experiment.CLEAR_LINE
        + "Depth 1, trainers 2, epoch 3/3, validation accuracy 50.0000% "
        "| maximum epochs reached"
    ) in output
    assert (
        experiment.CLEAR_LINE
        + "Depth 2, trainers 2, epoch 3/3, validation accuracy 50.0000% "
        "| maximum epochs reached"
    ) in output
    assert [stage["depth"] for stage in result["stages"]] == [1, 2]
    assert [stage["epochs"] for stage in result["stages"]] == [3, 3]
    assert result["epochs_trained"] == 6
    assert result["epochs_per_depth_limit"] == 3


def test_progressive_checkpoint_epochs_scale_with_epoch_limit():
    assert experiment._progressive_validation_checkpoints(200) == {
        30: (30, 0.0),
        50: (50, 0.5),
        80: (80, 0.9),
        100: (100, 0.92),
    }
    assert experiment._progressive_validation_checkpoints(150) == {
        30: (30, 0.0),
        50: (50, 0.5),
        80: (80, 0.9),
        100: (100, 0.92),
    }
    assert experiment._progressive_validation_checkpoints(500) == {
        75: (30, 0.0),
        125: (50, 0.5),
        200: (80, 0.9),
        250: (100, 0.92),
    }
    assert experiment._progressive_validation_checkpoints(1) == {}


@pytest.mark.parametrize(
    (
        "depth_one_accuracy",
        "expected_epoch",
        "expected_gate",
        "expected_epochs_trained",
    ),
    [
        (0.5, 80, 0.9, 180),
        (0.0, 30, 0.0, 130),
    ],
)
def test_progressive_screening_excludes_only_the_failed_depth(
    monkeypatch,
    depth_one_accuracy,
    expected_epoch,
    expected_gate,
    expected_epochs_trained,
):
    class DatasetFactory:
        @staticmethod
        def from_simulation(target, N, seed):
            return SimpleNamespace(
                challenges=np.zeros((N, 1)),
                responses=np.ones((N, 1)),
                seed=seed,
            )

    class AgentTrainer:
        def __init__(self, *args, **kwargs):
            self.model = FakeTorchModel(args[3])

        def train_epoch(self):
            pass

    class AttackModel:
        def __init__(self, models, n, response_bits, device):
            self.depth = models[0].depth

        def eval(self, challenges):
            return np.ones((len(challenges), 1), dtype=np.int8)

    monkeypatch.setattr(experiment, "ChallengeResponseSet", DatasetFactory)
    monkeypatch.setattr(experiment, "_AgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_TorchPUFModel", AttackModel)
    monkeypatch.setattr(
        experiment,
        "_pair_accuracy",
        lambda model, dataset: (
            depth_one_accuracy if model.depth == 1 else 0.95
        ),
    )

    result = experiment._attack_one_puf(
        SimpleNamespace(response_length=1),
        n=1,
        training_samples=1,
        validation_samples=1,
        test_samples=1,
        seed=10,
        family="sram",
        success_threshold=0.99,
        max_depth=2,
        agents=1,
        width=None,
        epochs=100,
        batch_size=1,
        learning_rate=0.001,
        device=SimpleNamespace(type="cpu"),
        progressive_screening=True,
    )

    assert [stage["status"] for stage in result["stages"]] == [
        "excluded",
        "complete",
    ]
    assert result["excluded_depths"] == [{
        "depth": 1,
        "epoch": expected_epoch,
        "reference_gate_epoch": expected_epoch,
        "validation_accuracy": depth_one_accuracy,
        "minimum_validation_accuracy": expected_gate,
    }]
    assert result["depth"] == 2
    assert result["candidate_excluded"] is False
    assert result["epochs_trained"] == expected_epochs_trained


def test_progressive_screening_stops_at_user_success_threshold(monkeypatch):
    class DatasetFactory:
        @staticmethod
        def from_simulation(target, N, seed):
            return SimpleNamespace(
                challenges=np.zeros((N, 1)),
                responses=np.ones((N, 1)),
                seed=seed,
            )

    class AgentTrainer:
        def __init__(self, *args, **kwargs):
            self.model = FakeTorchModel(args[3])

        def train_epoch(self):
            pass

    class AttackModel:
        def __init__(self, models, n, response_bits, device):
            self.depth = models[0].depth

        def eval(self, challenges):
            return np.ones((len(challenges), 1), dtype=np.int8)

    monkeypatch.setattr(experiment, "ChallengeResponseSet", DatasetFactory)
    monkeypatch.setattr(experiment, "_AgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_TorchPUFModel", AttackModel)
    monkeypatch.setattr(
        experiment,
        "_pair_accuracy",
        lambda model, dataset: 0.99,
    )

    result = experiment._attack_one_puf(
        SimpleNamespace(response_length=1),
        n=1,
        training_samples=1,
        validation_samples=1,
        test_samples=1,
        seed=10,
        family="sram",
        success_threshold=0.99,
        max_depth=2,
        agents=1,
        width=None,
        epochs=300,
        batch_size=1,
        learning_rate=0.001,
        device=SimpleNamespace(type="cpu"),
        progressive_screening=True,
    )

    assert result["threshold_reached"] is True
    assert result["validation_accuracy"] == 0.99
    assert result["depth"] == 1
    assert result["epochs_trained"] == 1
    assert [stage["depth"] for stage in result["stages"]] == [1]
    assert result["epochs_per_depth_limit"] == 300


def test_low_element_accuracy_stops_model_after_ten_epochs(monkeypatch):
    class DatasetFactory:
        @staticmethod
        def from_simulation(target, N, seed):
            return SimpleNamespace(
                challenges=np.ones((N, 1), dtype=np.int8),
                responses=-np.ones((N, 1), dtype=np.int8),
                seed=seed,
            )

    class AgentTrainer:
        def __init__(self, *args, **kwargs):
            self.model = FakeTorchModel(args[3])

        def train_epoch(self):
            pass

    class AttackModel:
        def __init__(self, models, n, response_bits, device):
            self.depth = models[0].depth

        def eval(self, challenges):
            return np.ones((len(challenges), 1), dtype=np.int8)

    monkeypatch.setattr(experiment, "ChallengeResponseSet", DatasetFactory)
    monkeypatch.setattr(experiment, "_AgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_TorchPUFModel", AttackModel)
    monkeypatch.setattr(experiment, "_pair_accuracy", lambda model, dataset: 0.25)

    result = experiment._attack_one_puf(
        SimpleNamespace(response_length=1),
        n=1,
        training_samples=1,
        validation_samples=1,
        test_samples=1,
        seed=30,
        family="sram",
        success_threshold=0.95,
        max_depth=2,
        agents=1,
        width=None,
        epochs=20,
        batch_size=1,
        learning_rate=0.001,
        device=SimpleNamespace(type="cpu"),
    )

    assert result["model_family"] == "sram"
    assert result["epochs_trained"] == 10
    assert result["early_stopped"] is True
    assert "first 10 epochs" in result["early_stop_reason"]
    assert [stage["depth"] for stage in result["stages"]] == [1]


def test_ring_oscillator_auto_uses_logistic_regression(monkeypatch, tmp_path):
    def unexpected_device_resolution(backend, device):
        raise AssertionError("Logistic regression should not resolve a Torch device.")

    monkeypatch.setattr(
        experiment, "_resolve_device", unexpected_device_resolution
    )
    result = experiment.run_ml_attack_experiment(
        families=["ring_oscillator"],
        n=8,
        training_samples=16,
        validation_samples=8,
        test_samples=8,
        seed=31,
        response_bits=1,
        max_depth=2,
        agents=1,
        width=8,
        epochs=1,
        batch_size=8,
        backend="torch_xpu",
        device=0,
        report_root=str(tmp_path),
        save_report=True,
        puf_instances=1,
    )["ring_oscillator"]["instances"][0]

    assert result["model_family"] == "ring_oscillator"
    assert result["model_architecture"] == "ring_oscillator_logistic_regression"
    assert result["attack"] == "ring_oscillator_logistic_regression"
    assert result["depth"] == 0
    assert result["epochs_trained"] == 0
    assert (
        experiment._resolve_attack_model_for_family(
            "ring_oscillator", "neural"
        )
        == "neural"
    )


def test_analytical_families_have_explicit_attack_routes(monkeypatch):
    class DatasetFactory:
        @staticmethod
        def from_simulation(target, N, seed):
            return SimpleNamespace(
                challenges=np.ones((N, 1), dtype=np.int8),
                responses=np.ones((N, 1), dtype=np.int8),
                seed=seed,
            )

    class AgentTrainer:
        def __init__(self, *args, **kwargs):
            self.model = FakeTorchModel(args[3])

        def train_epoch(self):
            pass

    class AttackModel:
        def __init__(self, models, n, response_bits, device):
            self.depth = models[0].depth

        def eval(self, challenges):
            return np.ones((len(challenges), 1), dtype=np.int8)

    monkeypatch.setattr(experiment, "ChallengeResponseSet", DatasetFactory)
    monkeypatch.setattr(experiment, "_AgentTrainer", AgentTrainer)
    monkeypatch.setattr(experiment, "_TorchPUFModel", AttackModel)
    monkeypatch.setattr(experiment, "_pair_accuracy", lambda model, dataset: 0.75)

    expected_routes = {
        "permutation": ("permutation_nonlinear_ensemble", 1),
        "sram": ("sram_nonlinear_ensemble", 1),
        "memristive": ("memristive_linear", 0),
        "silicon_photonic": ("silicon_photonic_nonlinear_ensemble", 1),
    }
    for family, (attack, depth) in expected_routes.items():
        result = experiment._attack_one_puf(
            SimpleNamespace(response_length=1),
            n=1,
            training_samples=1,
            validation_samples=1,
            test_samples=1,
            seed=32,
            family=family,
            success_threshold=0.95,
            max_depth=1,
            agents=1,
            width=None,
            epochs=1,
            batch_size=1,
            learning_rate=0.001,
            device=SimpleNamespace(type="cpu"),
        )
        assert result["model_family"] == family
        assert result["attack"] == attack
        assert result["depth"] == depth


def test_activation_profiles_cover_all_default_families():
    profiles = {
        family: experiment._activation_profile(family, depth=2)
        for family in experiment.DEFAULT_ATTACK_FAMILIES
    }

    assert profiles["optical"]["output_layer"] == "Linear regression"
    assert profiles["optical"]["hidden_layers"] == []
    assert profiles["arbiter"]["hidden_layers"] == ["Tanh", "Tanh"]
    assert profiles["ff_apuf"]["hidden_layers"] == ["Tanh"] * 4
    assert profiles["ff_apuf"]["internal_layers"] == ["SignSTE"]
    assert profiles["xor_apuf"]["internal_layers"] == [
        "Tanh chain scores",
        "Product parity",
    ]
    assert profiles["interpose"]["internal_layers"] == [
        "Annealed mixture over both upstream-bit hypotheses",
        "Downstream XOR parity with sign-aware STE gradients",
    ]
    assert profiles["ring_oscillator"]["hidden_layers"] == []
    assert profiles["memristive"]["hidden_layers"] == []
    assert profiles["sram"]["hidden_layers"] == ["Tanh", "Tanh"]
    assert profiles["permutation"]["hidden_layers"] == ["Tanh", "Tanh"]
    assert profiles["silicon_photonic"]["hidden_layers"] == ["Tanh", "Tanh"]


def test_summary_counts_instances_meeting_per_instance_threshold():
    instance_results = [
        {
            "epochs_trained": 2,
            "fit_elapsed_seconds": 1.0,
            "instance_matched": False,
            "test_accuracy": 0.60,
            "test_element_accuracy": 0.65,
        },
        {
            "epochs_trained": 3,
            "fit_elapsed_seconds": 1.5,
            "instance_matched": True,
            "test_accuracy": 0.70,
            "test_element_accuracy": 0.75,
        },
        {"epochs_trained": 4, "fit_elapsed_seconds": 2.0,
         "instance_matched": True, "test_accuracy": 0.90,
         "test_element_accuracy": 0.95},
    ]

    result = experiment._summarize_puf_instances(instance_results, success_threshold=0.66)

    assert result["puf_instances"] == 3
    assert result["matched_instances"] == 2
    assert result["instance_match_rate"] == 2 / 3
    assert np.isclose(result["mean_test_accuracy"], 0.7333333333)
    assert np.isclose(result["mean_test_element_accuracy"], 0.7833333333)
    assert result["success_threshold"] == 0.66
    assert result["epochs_trained"] == 9
    assert result["fit_elapsed_seconds"] == 4.5
    assert "required_matches" not in result
    assert "success" not in result
    assert "threshold_reached" not in result


def test_instances_are_independent_and_aggregated(monkeypatch, capsys, tmp_path):
    created_seeds = []
    test_scores = iter([0.964, 0.949, 1.0])
    test_sample_count = 1000

    def create_target(family, **kwargs):
        created_seeds.append(kwargs["seed"])
        return SimpleNamespace(seed=kwargs["seed"])

    def attack_target(target, *args):
        assert args[5] == 0.95
        assert args[13] == "arbiter"
        accuracy = next(test_scores)
        matched_count = int(accuracy * test_sample_count)
        response_matches = np.arange(test_sample_count) < matched_count
        puf_responses = np.ones((test_sample_count, 1), dtype=np.uint8)
        attack_responses = puf_responses.copy()
        attack_responses[~response_matches] = 0
        return {
            "test_accuracy": accuracy,
            "test_element_accuracy": accuracy,
            "matched_test_responses": matched_count,
            "test_response_count": test_sample_count,
            "epochs_trained": 1,
            "fit_elapsed_seconds": 0.5,
            "_response_rows": (
                puf_responses,
                attack_responses,
                response_matches,
            ),
        }

    monkeypatch.setattr(experiment, "create_puf", create_target)
    monkeypatch.setattr(experiment, "_attack_one_puf", attack_target)
    monkeypatch.setattr(experiment, "_resolve_device", lambda backend, device: "cpu")

    results = experiment.run_ml_attack_experiment(
        families=["arbiter"],
        puf_instances=3,
        test_samples=test_sample_count,
        success_threshold=0.95,
        backend="cpu",
        save_report=True,
        report_root=tmp_path,
        print_progress=True,
    )

    result = results["arbiter"]
    output = capsys.readouterr().out
    assert len(set(created_seeds)) == 3
    assert result["matched_instances"] == 2
    assert result["instance_match_rate"] == 2 / 3
    assert result["success_threshold"] == 0.95
    assert [instance["instance_matched"] for instance in result["instances"]] == [
        True, False, True
    ]
    assert "success" not in result
    assert "threshold_reached" not in result
    assert "\x1b[1A\x1b[2KAttacking 'arbiter': instance 1/3..." in output
    assert "\x1b[1A\x1b[2KAttacking 'arbiter': instance 3/3..." in output
    assert "Test results for 'arbiter', instance 1/3: 96.4000% accuracy" in output
    assert "Test results for 'arbiter', instance 2/3: 94.9000% accuracy" in output
    assert "complete responses; required threshold 95.0000%." in output
    assert (
        "Test summary for 'arbiter': mean test accuracy 97.1000% across 3 "
        "instances; required threshold 95.0000% per instance."
    ) in output
    assert "threshold met" not in output
    assert "threshold not met" not in output
    response_file = next((tmp_path / "general").iterdir()) / "responses.csv"
    with response_file.open(encoding="utf-8", newline="") as handle:
        response_rows = list(csv.reader(handle))
    assert len(response_rows) == 1 + 3 * test_sample_count
    match_counts = [
        sum(int(row[-1]) for row in response_rows[1 + index * test_sample_count:
                                                     1 + (index + 1) * test_sample_count])
        for index in range(3)
    ]
    assert match_counts == [964, 949, 1000]


def test_report_layout_includes_response_comparisons(tmp_path):
    results = {
        "arbiter": {
            "attack": "nonlinear_ensemble",
            "depth": 1,
            "agents": 3,
            "test_accuracy": 0.97,
            "success": True,
            "stages": [{"depth": 0}, {"depth": 1}],
        }
    }
    report_directory = save_ml_attack_report(
        results,
        {
            "default_families": ["arbiter"],
            "puf_instances": 1,
            "test_samples": 2,
        },
        ["arbiter"],
        scenario="iot",
        report_root=tmp_path,
        responses={
            "arbiter": [
                (
                    np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
                    np.asarray([[1, 0], [1, 1]], dtype=np.uint8),
                    np.asarray([True, False]),
                )
            ]
        },
    )

    assert report_directory.parent.name == "iot"
    assert report_directory.name.endswith("Report 1 - All PUFs")
    assert {path.name for path in report_directory.iterdir()} == {
        "parameters.json", "results.json", "summary.csv", "responses.csv"
    }
    assert json.loads((report_directory / "results.json").read_text()) == results
    with (report_directory / "summary.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["family"] == "arbiter"
    assert row["test_accuracy"] == "0.97"
    with (report_directory / "responses.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        response_rows = list(csv.reader(handle))
    assert response_rows[0] == [
        "label", "bit_0", "bit_1", "ml_bit_0", "ml_bit_1", "match"
    ]
    assert response_rows[1] == [
        "family arbiter, instance 1, sample 1, repetition 1",
        "1", "0", "1", "0", "1",
    ]
    assert response_rows[2][-1] == "0"


def test_unknown_crps_are_grouped_by_blinded_instance(tmp_path):
    dataset_directory = generate_crps(
        {"xor_apuf": 1, "arbiter": 1},
        n=8,
        total_samples=40,
        k=2,
        seed=45,
        data_root=tmp_path,
        dataset_name="blinded",
    )

    with (dataset_directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["instance_count"] == 2
    assert manifest["total_samples"] == 40
    assert [instance["instance_id"] for instance in manifest["instances"]] == [
        "instance_0001",
        "instance_0002",
    ]
    assert all("family" not in instance for instance in manifest["instances"])
    for instance in manifest["instances"]:
        with np.load(dataset_directory / instance["file"]) as arrays:
            assert arrays["challenges"].shape == (20, 8)
            assert arrays["responses"].shape == (20, 1)


def test_unknown_bagging_runs_candidates_and_saves_best_reports(tmp_path):
    dataset_directory = generate_crps(
        {"arbiter": 1},
        n=8,
        samples_per_instance=40,
        seed=46,
        data_root=tmp_path,
        dataset_name="attack-data",
    )
    result = run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=2,
        seed=47,
        candidate_families=["arbiter"],
        attack_models=["linear_logits"],
        epochs=1,
        batch_size=8,
        max_depth=1,
        backend="cpu",
        report_root=tmp_path / "reports",
    )

    instance = result["instances"][0]
    assert len(instance["trials"]) == 2
    assert instance["best_validation_selected"]["model_type"] == "linear_logits"
    assert instance["best_test_oracle"]["model_type"] == "linear_logits"
    assert all(
        trial["trial_elapsed_seconds"] >= 0 for trial in instance["trials"]
    )
    assert all(
        trial["total_elapsed_seconds"] >= trial["trial_elapsed_seconds"]
        for trial in instance["trials"]
    )
    assert result["best_guessed_model"]["model_type"] == "linear_logits"
    assert result["best_guessed_model"]["mean_test_accuracy"] == pytest.approx(
        np.mean([trial["test_accuracy"] for trial in instance["trials"]])
    )
    report_directory = Path(result["report_directory"])
    assert (report_directory / "responses.csv").is_file()
    assert (report_directory / "responses_validation_selected.csv").is_file()
    with (report_directory / "responses.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        response_rows = list(csv.DictReader(handle))
    assert response_rows
    assert response_rows[0]["instance_id"] == "instance_0001"
    assert response_rows[0]["model_type"] == "linear_logits"
    assert response_rows[0]["effective_batch_size"] == "8"
    assert response_rows[0]["selection_basis"] == "best_test_oracle"
    with (report_directory / "results.json").open(encoding="utf-8") as handle:
        saved_result = json.load(handle)
    assert saved_result["best_guessed_model"]["model_type"] == "linear_logits"
    assert saved_result["overall_summary"]["completed_instances"] == 1
    assert saved_result["instances"][0]["trials"][0]["trial_elapsed_seconds"] >= 0


def test_unknown_report_is_refreshed_after_each_instance(
    tmp_path, monkeypatch, capsys
):
    dataset_directory = generate_crps(
        {"arbiter": 2},
        n=8,
        samples_per_instance=40,
        seed=56,
        data_root=tmp_path,
        dataset_name="incremental-report-data",
    )
    report_root = tmp_path / "reports"
    original_attack = unknown_attack._experiment._attack_one_puf
    attack_calls = 0

    def check_report_before_next_instance(*args, **kwargs):
        nonlocal attack_calls
        if attack_calls == 1:
            report_path = next(report_root.rglob("results.json"))
            with report_path.open(encoding="utf-8") as handle:
                saved_result = json.load(handle)
            assert saved_result["overall_summary"]["completed_instances"] == 1
            assert saved_result["instances"][0]["instance_id"] == "instance_0001"
        attack_calls += 1
        return original_attack(*args, **kwargs)

    monkeypatch.setattr(
        unknown_attack._experiment,
        "_attack_one_puf",
        check_report_before_next_instance,
    )
    result = run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=1,
        seed=57,
        candidate_families=["arbiter"],
        attack_models=["linear_logits"],
        epochs=1,
        batch_size=8,
        max_depth=1,
        backend="cpu",
        report_root=report_root,
        print_progress=True,
    )

    assert attack_calls == 2
    output = capsys.readouterr().out
    assert "Timer started at " in output
    assert "lap_elapsed=" in output
    assert "total_elapsed=" in output
    assert result["overall_summary"]["completed_instances"] == 2


def test_unknown_auto_candidates_use_the_family_router(tmp_path):
    dataset_directory = generate_crps(
        {"arbiter": 1},
        n=8,
        samples_per_instance=40,
        seed=49,
        data_root=tmp_path,
        dataset_name="auto-data",
    )
    result = run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=1,
        seed=50,
        attack_models=["auto"],
        epochs=1,
        batch_size=8,
        backend="cpu",
        report_root=tmp_path / "reports",
    )

    assert any(
        trial["family_hypothesis"] == "arbiter"
        and trial["model_type"] == "logistic_regression"
        for trial in result["instances"][0]["trials"]
    )
    assert "arbiter" in result["auto_selected_attack_to_families"][
        "logistic_regression"
    ]
    assert any(
        candidate["family_hypothesis"] == "permutation"
        for candidate in result["instances"][0]["skipped_candidates"]
    )


def test_unknown_family_hypothesis_can_select_models_per_family(tmp_path):
    dataset_directory = generate_crps(
        {"arbiter": 1},
        n=8,
        samples_per_instance=40,
        seed=58,
        data_root=tmp_path,
        dataset_name="per-family-models",
    )
    result = run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=1,
        seed=59,
        family_hypothesis=["arbiter", "xor_apuf"],
        candidate_model_types={
            "arbiter": "linear_logits",
            "xor_apuf": ["linear_logits"],
        },
        epochs=1,
        batch_size=8,
        max_depth=1,
        backend="cpu",
        report_root=tmp_path / "reports",
    )

    trials = result["instances"][0]["trials"]
    assert {
        (trial["family_hypothesis"], trial["model_type"])
        for trial in trials
    } == {
        ("arbiter", "linear_logits"),
        ("xor_apuf", "linear_logits"),
    }
    report_path = Path(result["report_directory"]) / "parameters.json"
    with report_path.open(encoding="utf-8") as handle:
        parameters = json.load(handle)
    assert parameters["family_hypothesis"] == ["arbiter", "xor_apuf"]
    assert parameters["candidate_model_types"] == {
        "arbiter": ["linear_logits"],
        "xor_apuf": ["linear_logits"],
    }


def test_unknown_linear_logits_candidate_works_for_xor_family(tmp_path):
    dataset_directory = generate_crps(
        {"xor_apuf": 1},
        n=8,
        samples_per_instance=80,
        seed=56,
        data_root=tmp_path,
        dataset_name="xor-linear-data",
    )
    result = run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=1,
        seed=57,
        candidate_families=["xor_apuf"],
        epochs=1,
        batch_size=16,
        backend="cpu",
        report_root=tmp_path / "reports",
    )

    tested_models = {
        trial["model_type"] for trial in result["instances"][0]["trials"]
    }
    assert tested_models == {
        "linear_logits",
        "neural",
        "svm",
        "fc_lstm",
        "xor_rank_two",
        "deep_neural",
        "xor_quadratic",
    }


def test_unknown_linear_logits_candidate_works_for_interpose_family(
    tmp_path, capsys
):
    dataset_directory = generate_crps(
        {"interpose": 1},
        n=8,
        samples_per_instance=80,
        seed=59,
        data_root=tmp_path,
        dataset_name="interpose-linear-data",
    )
    result = run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=1,
        seed=60,
        candidate_families=["interpose"],
        attack_models=["linear_logits"],
        epochs=1,
        batch_size=16,
        backend="cpu",
        report_root=tmp_path / "reports",
        print_progress=True,
    )

    assert result["instances"][0]["trials"][0]["model_type"] == "linear_logits"
    assert (
        experiment.CLEAR_LINE
        + "Depth 0, trainers 1, epoch 1/1, validation accuracy "
        in capsys.readouterr().out
    )


def test_unknown_fc_lstm_candidates_use_positive_depth_for_linear_families(tmp_path):
    dataset_directory = generate_crps(
        {"arbiter": 1},
        n=8,
        samples_per_instance=40,
        seed=61,
        data_root=tmp_path,
        dataset_name="fc-lstm-data",
    )

    for family in ("ring_oscillator", "memristive"):
        result = run_attack_unknown_puf(
            dataset_directory,
            bagging_runs=1,
            seed=62,
            candidate_families=[family],
            attack_models=["fc_lstm"],
            max_depth=1,
            epochs=1,
            batch_size=8,
            backend="cpu",
            report_root=tmp_path / f"reports-{family}",
        )

        assert result["instances"][0]["trials"][0]["depth"] == 1


def test_unknown_runner_passes_torch_xpu_backend_to_attack(monkeypatch, tmp_path):
    dataset_directory = generate_crps(
        {"arbiter": 1},
        n=8,
        samples_per_instance=40,
        seed=58,
        data_root=tmp_path,
        dataset_name="xpu-data",
    )
    xpu_device = SimpleNamespace(type="xpu")
    resolved_backends = []

    def resolve_device(backend, device):
        resolved_backends.append((backend, device))
        return xpu_device

    def attack_with_fake_device(*args, **kwargs):
        assert args[13] is xpu_device
        training, validation, test = kwargs["crp_splits"]
        test_responses = np.asarray(test.responses).reshape(
            len(test.challenges), -1
        )
        matches = np.ones(len(test.challenges), dtype=bool)
        sample_counts = (len(training.challenges), len(validation.challenges),
                         len(test.challenges))
        return {
            "attack": "linear_logits",
            "depth": 0,
            "validation_accuracy": 1.0,
            "validation_element_accuracy": 1.0,
            "test_accuracy": 1.0,
            "test_element_accuracy": 1.0,
            "training_samples": sample_counts[0],
            "validation_samples": sample_counts[1],
            "test_samples": sample_counts[2],
            "epochs_trained": 0,
            "fit_elapsed_seconds": 0.0,
            "_response_rows": (
                (test_responses > 0).astype(np.uint8),
                (test_responses > 0).astype(np.uint8),
                matches,
            ),
        }

    monkeypatch.setattr(experiment, "_resolve_device", resolve_device)
    monkeypatch.setattr(experiment, "_attack_one_puf", attack_with_fake_device)
    run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=1,
        candidate_families=["arbiter"],
        attack_models=["linear_logits"],
        backend="torch_xpu",
        device=0,
        report_root=tmp_path / "reports",
    )

    assert resolved_backends == [("torch_xpu", 0)]


def test_unknown_runner_retries_xpu_resource_error_with_smaller_batch(
    monkeypatch, tmp_path, capsys
):
    dataset_directory = generate_crps(
        {"arbiter": 1},
        n=8,
        samples_per_instance=40,
        seed=63,
        data_root=tmp_path,
        dataset_name="xpu-retry-data",
    )
    xpu_device = SimpleNamespace(type="xpu")
    batch_sizes = []
    cache_clears = []

    monkeypatch.setattr(
        experiment,
        "_resolve_device",
        lambda backend, device: xpu_device,
    )
    monkeypatch.setattr(
        unknown_attack,
        "_clear_xpu_cache",
        lambda: cache_clears.append(True),
    )

    def attack_with_resource_failures(*args, **kwargs):
        batch_sizes.append(args[11])
        if len(batch_sizes) < 3:
            raise RuntimeError(
                "level_zero backend failed with error: "
                "UR_RESULT_ERROR_OUT_OF_RESOURCES"
            )
        _, _, test = kwargs["crp_splits"]
        test_responses = np.asarray(test.responses).reshape(
            len(test.challenges), -1
        )
        matches = np.ones(len(test.challenges), dtype=bool)
        sample_counts = (
            len(kwargs["crp_splits"][0].challenges),
            len(kwargs["crp_splits"][1].challenges),
            len(test.challenges),
        )
        return {
            "attack": "linear_logits",
            "model_architecture": "linear_logits",
            "depth": 0,
            "validation_accuracy": 1.0,
            "validation_element_accuracy": 1.0,
            "test_accuracy": 1.0,
            "test_element_accuracy": 1.0,
            "training_samples": sample_counts[0],
            "validation_samples": sample_counts[1],
            "test_samples": sample_counts[2],
            "epochs_trained": 0,
            "fit_elapsed_seconds": 0.0,
            "_response_rows": (
                (test_responses > 0).astype(np.uint8),
                (test_responses > 0).astype(np.uint8),
                matches,
            ),
        }

    monkeypatch.setattr(
        experiment, "_attack_one_puf", attack_with_resource_failures
    )
    result = run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=1,
        candidate_families=["arbiter"],
        attack_models=["linear_logits"],
        batch_size=256,
        backend="torch_xpu",
        device=0,
        report_root=tmp_path / "reports",
        print_progress=True,
    )

    assert batch_sizes == [256, 128, 64]
    assert cache_clears == [True, True]
    output = capsys.readouterr().out
    assert "model_type='linear_logits'" in output
    assert "architecture='linear_logits'" in output
    assert (
        result["instances"][0]["trials"][0]["effective_batch_size"] == 64
    )
    response_file = Path(result["report_directory"]) / "responses.csv"
    with response_file.open(encoding="utf-8", newline="") as handle:
        response_rows = list(csv.DictReader(handle))
    assert response_rows[0]["effective_batch_size"] == "64"
    candidate_scores_file = (
        Path(result["report_directory"]) / "candidate_scores.csv"
    )
    with candidate_scores_file.open(
        encoding="utf-8", newline=""
    ) as handle:
        candidate_score_rows = list(csv.DictReader(handle))
    assert candidate_score_rows[0]["effective_batch_size"] == "64"
    results_file = Path(result["report_directory"]) / "results.json"
    with results_file.open(encoding="utf-8") as handle:
        report_results = json.load(handle)
    assert (
        report_results["instances"][0]["trials"][0]["effective_batch_size"]
        == 64
    )


def test_unknown_runner_reports_incompatible_family_hypotheses(tmp_path):
    dataset_directory = generate_crps(
        {"arbiter": 1},
        n=8,
        samples_per_instance=40,
        seed=54,
        data_root=tmp_path,
        dataset_name="incompatible-hypothesis",
    )
    result = run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=1,
        seed=55,
        candidate_families=["arbiter", "permutation"],
        attack_models=["auto"],
        epochs=1,
        backend="cpu",
        report_root=tmp_path / "reports",
    )

    skipped = result["instances"][0]["skipped_candidates"]
    assert len(skipped) == 1
    assert skipped[0]["family_hypothesis"] == "permutation"
    assert "not supported for n=8" in skipped[0]["reason"]


def test_unknown_continuous_crps_use_optical_compatible_models(tmp_path):
    dataset_directory = generate_crps(
        {"optical": 1},
        n=8,
        samples_per_instance=60,
        seed=52,
        data_root=tmp_path,
        dataset_name="optical-data",
    )
    result = run_attack_unknown_puf(
        dataset_directory,
        bagging_runs=1,
        seed=53,
        candidate_families=["optical"],
        attack_models=["auto"],
        epochs=1,
        batch_size=16,
        backend="cpu",
        report_root=tmp_path / "reports",
    )

    assert result["instances"][0]["trials"][0]["model_type"] == (
        "optical_regression"
    )
    assert result["model_to_compatible_family_hypotheses"] == {
        "optical_regression": ["optical"]
    }


def test_linear_logits_is_available_in_known_family_experiment():
    result = experiment.run_ml_attack_experiment(
        families=["arbiter"],
        n=8,
        training_samples=32,
        validation_samples=16,
        test_samples=16,
        seed=51,
        max_depth=1,
        agents=1,
        epochs=1,
        batch_size=8,
        backend="cpu",
        attack_model="linear_logits",
        save_report=False,
        puf_instances=1,
    )["arbiter"]["instances"][0]

    assert result["model_architecture"] == "linear_logits"
    assert result["depth"] == 0
    assert result["activation_types"]["hidden_layers"] == []
    assert result["activation_types"]["output_layer"] == "Linear logits"


def test_unknown_split_keeps_repeated_challenges_together():
    challenges = np.asarray(
        [
            [1, -1], [1, -1], [1, -1], [1, -1],
            [-1, 1], [-1, 1], [-1, 1], [-1, 1],
            [1, 1], [1, 1], [1, 1], [1, 1],
        ],
        dtype=np.int8,
    )
    splits = _split_indices(challenges, seed=48)

    split_challenges = [
        {tuple(row) for row in challenges[indices]}
        for indices in splits
    ]
    assert all(
        split_challenges[left].isdisjoint(split_challenges[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
