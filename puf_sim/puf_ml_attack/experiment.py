"""Train progressively more expressive models against PUF implementations."""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from pypuf.io import ChallengeResponseSet
from pypuf.simulation import Simulation

from puf_sim.puf_implementations import PUF_IMPLEMENTATIONS, create_puf

from . import arbiter as _arbiter
from . import fc_lstm as _fc_lstm
from . import logistic_regression as _logistic_regression
from . import optical as _optical
from . import permutation as _permutation
from . import ring_oscillator as _ring_oscillator
from . import sram as _sram
from . import svm as _svm
from .common import (
    AgentTrainer as _AgentTrainer,
    TorchPUFModel as _TorchPUFModel,
    pair_accuracy as _pair_accuracy,
    resolve_device as _resolve_device,
)
from .writer import save_ml_attack_report

_ArbiterAgentTrainer = _arbiter.AgentTrainer
_ArbiterTorchPUFModel = _arbiter.TorchPUFModel
_OpticalTorchPUFModel = _optical.TorchPUFModel
_optical_pair_accuracy = _optical.pair_accuracy

DEFAULT_ATTACK_FAMILIES = (
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

CLEAR_PREVIOUS_LINE = "\033[1A\x1b[2K"
MODEL_SCREENING_EPOCHS = 10
MODEL_SCREENING_THRESHOLD = 0.5


def _normalize_family_key(family: str) -> str:
    key = family.strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "feed_forward_arbiter": "ff_apuf",
        "xor_arbiter": "xor_apuf",
        "ro_puf": "ring_oscillator",
        "static_random_access_memory": "sram",
        "memristive_puf": "memristive",
    }.get(key, key)


def _resolve_attack_model_for_family(
    family_key: str,
    attack_model: str,
    n: int = 64,
    k: Optional[int] = 2,
    response_bits: int = 1,
    hash_responses: bool = True,
) -> str:
    if family_key == "optical":
        return "optical_regression"
    if attack_model == "auto":
        if family_key == "arbiter":
            return "logistic_regression"
        if family_key == "sram":
            return (
                "sram_logistic_regression"
                if not hash_responses
                else "sram_hash_bruteforce"
            )
        if family_key == "ring_oscillator":
            return "ring_oscillator_logistic_regression"
        if family_key == "permutation":
            return "permutation_parity"
        if family_key == "ff_apuf":
            return "deep_neural"
        if family_key in {"xor_apuf", "xor_arbiter"}:
            effective_chain_count = n if k is None else max(k, response_bits)
            if response_bits > 0:
                chains_per_bit, remainder = divmod(
                    effective_chain_count, response_bits
                )
                if remainder == 0 and chains_per_bit == 1:
                    return "xor_logistic_regression"
                if remainder == 0 and chains_per_bit == 2:
                    return "xor_rank_two"
            return "neural"
        if family_key == "interpose":
            return "interpose_logistic_regression"
        return "neural"
    return attack_model


def _response_element_accuracy(
    model: Simulation, dataset: ChallengeResponseSet, family_key: str
) -> float:
    predicted = np.asarray(model.eval(dataset.challenges), dtype=float).reshape(
        len(dataset.challenges), -1
    )
    expected = np.asarray(dataset.responses, dtype=float).reshape(
        len(dataset.challenges), -1
    )
    if family_key == "optical":
        matches = np.isclose(
            predicted,
            expected,
            rtol=_optical.RELATIVE_TOLERANCE,
            atol=_optical.ABSOLUTE_TOLERANCE,
        )
    else:
        matches = predicted == expected
    return float(np.mean(matches))


def _validation_selection_key(
    family_key: str,
    response_bits: int,
    exact_accuracy: float,
    element_accuracy: float,
) -> tuple[float, float]:
    if family_key == "interpose" and response_bits > 1:
        return element_accuracy, exact_accuracy
    return exact_accuracy, element_accuracy


def _optimal_bipolar_thresholds(
    logits: np.ndarray, responses: np.ndarray
) -> np.ndarray:
    score_matrix = np.asarray(logits, dtype=np.float32)
    response_matrix = np.asarray(responses).reshape(len(score_matrix), -1)
    if len(score_matrix) == 0:
        return np.zeros((response_matrix.shape[1],), dtype=np.float32)
    thresholds = np.zeros((response_matrix.shape[1],), dtype=np.float32)
    for bit_index in range(response_matrix.shape[1]):
        bit_scores = score_matrix[:, bit_index]
        bit_labels = np.where(response_matrix[:, bit_index] > 0, 1, -1).astype(
            np.int8
        )
        sorted_order = np.argsort(bit_scores, kind="mergesort")
        sorted_scores = bit_scores[sorted_order]
        sorted_labels = bit_labels[sorted_order]
        positive_count = int(np.count_nonzero(bit_labels == 1))
        best_correct = positive_count
        current_correct = positive_count
        best_threshold = float(sorted_scores[0] - 1.0)
        index = 0
        while index < len(sorted_scores):
            score_value = sorted_scores[index]
            next_index = index
            positive_at_value = 0
            negative_at_value = 0
            while (
                next_index < len(sorted_scores)
                and sorted_scores[next_index] == score_value
            ):
                if sorted_labels[next_index] == 1:
                    positive_at_value += 1
                else:
                    negative_at_value += 1
                next_index += 1
            current_correct -= positive_at_value
            current_correct += negative_at_value
            if next_index < len(sorted_scores):
                candidate_threshold = float(
                    (score_value + sorted_scores[next_index]) / 2.0
                )
            else:
                candidate_threshold = float(score_value + 1.0)
            if current_correct > best_correct:
                best_correct = current_correct
                best_threshold = candidate_threshold
            index = next_index
        thresholds[bit_index] = best_threshold
    return thresholds


def _calibrate_model_thresholds(
    models: Sequence[Any],
    dataset: ChallengeResponseSet,
    device: Any,
) -> np.ndarray:
    import torch

    inputs = torch.as_tensor(dataset.challenges, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = torch.stack([model(inputs) for model in models]).mean(dim=0)
    return _optimal_bipolar_thresholds(logits.cpu().numpy(), dataset.responses)


def _activation_profile(family_key: str, depth: int) -> Dict[str, Any]:
    if family_key in {"ring_oscillator", "memristive"}:
        depth = 0
    profile: Dict[str, Any] = {
        "hidden_layers": ["Tanh"] * depth,
        "internal_layers": [],
        "output_layer": "Linear logits",
        "prediction": "Bipolar threshold",
    }
    if family_key == "optical":
        profile.update({
            "hidden_layers": [],
            "output_layer": "Linear regression",
            "prediction": "Continuous output",
        })
    elif family_key == "ff_apuf":
        profile.update({
            "hidden_layers": ["Tanh"] * (2 * depth),
            "internal_layers": ["SignSTE"],
            "output_layer": "Linear delay scores",
        })
    elif family_key == "xor_apuf":
        profile.update({
            "internal_layers": [
                "Linear Arbiter-chain scores with nonlinear residual",
                "Product parity with sign-aware STE gradients",
            ],
            "output_layer": "Parity score",
        })
    elif family_key == "interpose":
        profile.update({
            "hidden_layers": ["Tanh"] * (2 * depth),
            "internal_layers": [
                "Annealed mixture over both upstream-bit hypotheses",
                "Downstream XOR parity with sign-aware STE gradients",
            ],
            "output_layer": "Parity score",
        })
    return profile


def _model_profile(
    family_key: str, depth: int, attack_model: str, device: Any = None
) -> Dict[str, Any]:
    if attack_model == "logistic_regression":
        return {
            "feature_mapping": ["Arbiter Threshold Transform"],
            "estimator": "LogisticRegression per response bit",
            "internal_layers": [],
            "output_layer": "Logistic decision boundary",
            "prediction": "Bipolar class prediction",
        }
    if attack_model == "sram_logistic_regression":
        return {
            "feature_mapping": ["Corresponding raw SRAM challenge bit"],
            "estimator": "LogisticRegression per response bit",
            "internal_layers": [],
            "output_layer": "Logistic decision boundary",
            "prediction": "Bipolar class prediction",
        }
    if attack_model == "ring_oscillator_logistic_regression":
        return {
            "feature_mapping": ["All raw ring-oscillator challenge bits"],
            "estimator": "LogisticRegression per response bit",
            "internal_layers": [],
            "output_layer": "Logistic decision boundary",
            "prediction": "Bipolar class prediction",
        }
    if attack_model == "sram_hash_bruteforce":
        return {
            "feature_mapping": ["Challenge XOR candidate SRAM fingerprint"],
            "search": "SHA-256 comparison against observed CRPs",
            "candidate_fingerprints": _sram.HASH_CANDIDATE_LIMIT,
            "prediction": "Bipolar threshold",
        }
    if attack_model == "xor_logistic_regression":
        return {
            "feature_mapping": ["XOR Arbiter threshold transform"],
            "estimator": "PyTorch logistic regression per response bit",
            "internal_layers": [],
            "output_layer": "Linear logits",
            "prediction": "Bipolar threshold",
        }
    if attack_model == "xor_quadratic":
        return {
            "feature_mapping": [
                "Arbiter threshold transform",
                "Degree-two pairwise products",
            ],
            "estimator": "PyTorch logistic regression per response bit",
            "internal_layers": ["Two-chain XOR as a quadratic threshold"],
            "output_layer": "Linear logits",
            "prediction": "Bipolar threshold",
        }
    if attack_model == "xor_rank_two":
        return {
            "feature_mapping": ["Arbiter threshold transform"],
            "estimator": "Factorized PyTorch Arbiter-pair model per response bit",
            "internal_layers": [
                "Two learned Arbiter delay scores per response bit",
                "Product of the paired chain scores",
            ],
            "output_layer": "Arbiter-score product",
            "prediction": "Bipolar threshold",
        }
    if attack_model in {
        "interpose_logistic_regression",
        "staged_split",
    }:
        return {
            "feature_mapping": [
                "Arbiter Threshold Transform for upstream and downstream challenges"
            ],
            "estimator": (
                "Linear logistic Arbiter scores with a two-hypothesis "
                "upstream-bit mixture and downstream XOR composition"
            ),
            "hidden_layers": [],
            "internal_layers": (
                [
                    "Alternating downstream and upstream logistic fitting",
                    "Annealed mixture over both upstream-bit hypotheses",
                    "Downstream XOR parity per response bit",
                ]
                if attack_model == "staged_split"
                else [
                    "Annealed mixture over both upstream-bit hypotheses",
                    "Downstream XOR parity per response bit",
                ]
            ),
            "output_layer": "Parity score",
            "prediction": "Bipolar threshold",
        }
    if attack_model == "ff_fc_lstm":
        return {
            "feature_mapping": [
                "Arbiter Threshold Transform per feed-forward segment"
            ],
            "hidden_layers": ["Per-bit Linear projection", "Tanh"],
            "recurrent_layers": (
                ["Upstream Bidirectional LSTM"] * depth
                + ["Downstream Bidirectional LSTM"] * depth
            ),
            "internal_layers": [
                "SignSTE virtual bit",
                "Feed-forward challenge insertion",
                "Downstream challenge gate",
                "Upstream delay × gate + downstream delay",
            ],
            "output_layer": "Feed-forward delay logits",
            "prediction": "Bipolar threshold",
        }
    if attack_model == "permutation_parity":
        return {
            "feature_mapping": [
                "Per-chain fixed challenge permutation",
                "Arbiter Threshold Transform per chain",
            ],
            "hidden_layers": [],
            "internal_layers": [
                "Independent linear Arbiter score per permutation chain",
                "Direct paired-score product with BCE for two-chain groups",
                "Deterministic spectral start and alternating chainwise BCE warm-up",
                "Sign-aware straight-through parity for larger groups",
            ],
            "output_layer": "Parity score per response bit",
            "prediction": "Bipolar threshold",
        }
    if attack_model == "deep_neural":
        family_operations = {
            "ff_apuf": [
                "Lane-specific affine delay estimators",
                "Segment delay residual DNNs",
                "SignSTE virtual-bit insertion",
                "Feed-forward delay recurrence",
            ],
            "xor_apuf": [
                "Arbiter Threshold Transform",
                "Deep chain-score network",
                "Sign-aware parity composition",
            ],
            "interpose": [
                "Upstream Arbiter DNN",
                "Annealed mixture over both upstream-bit hypotheses",
                "Downstream XOR Arbiter DNN",
            ],
            "permutation": [
                "Fixed per-chain challenge permutation",
                "Per-chain Arbiter Threshold Transform",
                "Sign-aware parity composition",
            ],
        }
        return {
            "feature_mapping": ["Family-specific Arbiter challenge features"],
            "estimator": "PyTorch structure-aware deep neural network",
            "hidden_layers": ["Linear + Tanh"] * depth,
            "internal_layers": family_operations.get(family_key, []),
            "output_layer": "Family-specific delay or chain scores",
            "prediction": "Bipolar threshold",
        }
    if attack_model == "fc_lstm":
        return {
            "feature_mapping": ["Ordered raw challenge bits"],
            "hidden_layers": ["Per-bit Linear projection", "Tanh"],
            "recurrent_layers": ["Bidirectional LSTM"] * depth,
            "internal_layers": [],
            "output_layer": "Fully connected response logits",
            "prediction": "Bipolar threshold",
        }
    if attack_model == "svm":
        if family_key == "xor_apuf":
            feature_mapping = ["XOR Arbiter threshold transform"]
            estimator = (
                "PyTorch parity-composed hinge-loss SVM over learned chain scores"
            )
            internal_layers = [
                "Linear chain margins",
                "Tanh chain scores",
                "Product parity",
            ]
        else:
            feature_mapping = [
                "Raw challenge features",
                f"RBF random Fourier features ({_svm.RBF_FEATURE_COUNT})",
            ]
            internal_layers = []
            if getattr(device, "type", None) == "xpu":
                estimator = (
                    "PyTorch epsilon-insensitive SVR per response bit"
                    if family_key == "optical"
                    else "PyTorch hinge-loss SVM per response bit"
                )
            else:
                estimator = (
                    "One-vs-rest LinearSVR per response bit"
                    if family_key == "optical"
                    else "One-vs-rest LinearSVC per response bit"
                )
        return {
            "feature_mapping": feature_mapping,
            "estimator": estimator,
            "internal_layers": internal_layers,
            "prediction": "Continuous output" if family_key == "optical" else "Bipolar threshold",
        }
    return _activation_profile(family_key, depth)


def _attack_one_puf(
    target: Simulation,
    n: int,
    training_samples: int,
    validation_samples: int,
    test_samples: int,
    seed: int,
    success_threshold: float,
    max_depth: int,
    agents: int,
    width: Optional[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: Any,
    family: str,
    print_progress: bool = False,
    attack_model: str = "neural",
) -> Dict[str, Any]:
    training = ChallengeResponseSet.from_simulation(target, N=training_samples, seed=seed)
    validation = ChallengeResponseSet.from_simulation(
        target, N=validation_samples, seed=seed + 1
    )    
    test = ChallengeResponseSet.from_simulation(target, N=test_samples, seed=seed + 2)
    hidden_width = width or max(16, 2 * n)
    best_model: Optional[Simulation] = None
    best_validation_accuracy = -1.0
    best_validation_element_accuracy = -1.0
    best_validation_selection_key = (-1.0, -1.0)
    selected_stage = "neural_ensemble"
    selected_depth = 1
    selected_agent_count = agents
    fit_seconds = 0.0
    epochs_trained = 0
    agent_epochs_trained = 0
    screening_element_accuracy = -1.0
    screening_passed = False
    threshold_reached = False
    early_stopped = False
    early_stop_reason: Optional[str] = None
    stage_history: list[Dict[str, Any]] = []

    family_key = _normalize_family_key(family)
    select_by_element_accuracy = (
        family_key == "interpose" and target.response_length > 1
    )
    if attack_model == "neural" and family_key in {"xor_apuf", "xor_arbiter"}:
        chains_per_bit, remainder = divmod(target.k, target.response_length)
        if remainder == 0 and chains_per_bit == 2:
            attack_model = "xor_rank_two"
    feed_forward_loop: Optional[tuple[int, int]] = None
    if attack_model == "ff_fc_lstm":
        feed_forward_loops = getattr(target, "ff", None)
        if feed_forward_loops is None or len(feed_forward_loops) != 1:
            raise ValueError(
                "The feed-forward-aware FC-BiLSTM currently requires exactly "
                "one feed-forward loop."
            )
        feed_forward_loop = tuple(feed_forward_loops[0])

    if attack_model in {"fc_lstm", "ff_fc_lstm"}:
        trainer_type = _fc_lstm.AgentTrainer
        model_type = _fc_lstm.TorchFCBiLSTMModel
        stage_name = (
            _fc_lstm.FF_APUF_STAGE_NAME
            if attack_model == "ff_fc_lstm"
            else _fc_lstm.STAGE_NAME
        )
        pair_accuracy_fn = _pair_accuracy
    elif attack_model == "deep_neural":
        trainer_type = _arbiter.deep_neural.AgentTrainer
        model_type = _TorchPUFModel
        stage_name = (
            f"{_arbiter.deep_neural.STAGE_NAME}_{family_key}"
        )
        pair_accuracy_fn = _pair_accuracy
    elif attack_model in {
        "interpose_logistic_regression",
        "staged_split",
    }:
        trainer_type = (
            _arbiter.interpose.StagedSplitAgentTrainer
            if attack_model == "staged_split"
            else _arbiter.interpose.LogisticAgentTrainer
        )
        model_type = _arbiter.interpose.InterposeTorchPUFModel
        stage_name = (
            _arbiter.interpose.STAGED_SPLIT_STAGE_NAME
            if attack_model == "staged_split"
            else _arbiter.interpose.LOGISTIC_STAGE_NAME
        )
        pair_accuracy_fn = _pair_accuracy
    elif attack_model == "xor_logistic_regression":
        trainer_type = _arbiter.xor_apuf.LogisticAgentTrainer
        model_type = _arbiter.xor_apuf.XorTorchPUFModel
        stage_name = _arbiter.xor_apuf.LOGISTIC_STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif attack_model == "xor_quadratic":
        trainer_type = _arbiter.xor_apuf.QuadraticAgentTrainer
        model_type = _arbiter.xor_apuf.XorTorchPUFModel
        stage_name = _arbiter.xor_apuf.QUADRATIC_STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif attack_model == "xor_rank_two":
        trainer_type = _arbiter.xor_apuf.RankTwoAgentTrainer
        model_type = _arbiter.xor_apuf.XorTorchPUFModel
        stage_name = _arbiter.xor_apuf.RANK_TWO_STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif family_key == "arbiter":
        trainer_type = _ArbiterAgentTrainer
        model_type = _ArbiterTorchPUFModel
        stage_name = _arbiter.STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif family_key in {"ff_apuf", "feed_forward_arbiter"}:
        trainer_type = _arbiter.feed_forward.AgentTrainer
        model_type = _TorchPUFModel
        stage_name = _arbiter.feed_forward.STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif family_key in {"xor_apuf", "xor_arbiter"}:
        trainer_type = _arbiter.xor_apuf.AgentTrainer
        model_type = _arbiter.xor_apuf.XorTorchPUFModel
        stage_name = _arbiter.xor_apuf.STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif family_key == "interpose":
        trainer_type = _arbiter.interpose.AgentTrainer
        model_type = _arbiter.interpose.InterposeTorchPUFModel
        stage_name = _arbiter.interpose.STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif family_key == "optical":
        model_type = _OpticalTorchPUFModel
        stage_name = _optical.BOOTSTRAP_STAGE_NAME
        pair_accuracy_fn = _optical_pair_accuracy
    elif family_key == "permutation":
        if attack_model == "permutation_parity":
            trainer_type = _permutation.AgentTrainer
        else:
            trainer_type = _AgentTrainer
        model_type = _TorchPUFModel
        stage_name = (
            _permutation.STAGE_NAME
            if attack_model == "permutation_parity"
            else "permutation_nonlinear_ensemble"
        )
        pair_accuracy_fn = _pair_accuracy
    elif family_key == "ring_oscillator":
        trainer_type = _AgentTrainer
        model_type = _TorchPUFModel
        stage_name = "ring_oscillator_linear"
        pair_accuracy_fn = _pair_accuracy
    elif family_key == "sram":
        trainer_type = _AgentTrainer
        model_type = _TorchPUFModel
        stage_name = "sram_nonlinear_ensemble"
        pair_accuracy_fn = _pair_accuracy
    elif family_key == "memristive":
        trainer_type = _AgentTrainer
        model_type = _TorchPUFModel
        stage_name = "memristive_linear"
        pair_accuracy_fn = _pair_accuracy
    elif family_key == "silicon_photonic":
        trainer_type = _AgentTrainer
        model_type = _TorchPUFModel
        stage_name = "silicon_photonic_nonlinear_ensemble"
        pair_accuracy_fn = _pair_accuracy
    else:
        raise ValueError(f"Unsupported PUF family for ML attack: {family!r}.")

    def build_attack_model(
        models: Sequence[Any],
        model_device: Any,
        response_thresholds: Optional[np.ndarray] = None,
    ) -> Simulation:
        if attack_model in {"fc_lstm", "ff_fc_lstm"}:
            return model_type(
                models,
                n,
                target.response_length,
                model_device,
                inference_batch_size=batch_size,
            )
        if response_thresholds is not None and attack_model == "deep_neural":
            return model_type(
                models,
                n,
                target.response_length,
                model_device,
                response_thresholds=response_thresholds,
            )
        return model_type(models, n, target.response_length, model_device)

    if attack_model in {
        "logistic_regression",
        "sram_logistic_regression",
        "sram_hash_bruteforce",
        "ring_oscillator_logistic_regression",
    }:
        fit_start = time.perf_counter()
        if attack_model == "ring_oscillator_logistic_regression":
            best_model = _ring_oscillator.RingOscillatorLogisticRegression(
                training.challenges,
                training.responses,
                target.response_length,
                seed,
            )
            selected_stage = _ring_oscillator.STAGE_NAME
        elif attack_model == "sram_logistic_regression":
            best_model = _sram.SRAMLogisticRegression(
                training.challenges,
                training.responses,
                target.response_length,
                seed,
            )
            selected_stage = _sram.STAGE_NAME
        elif attack_model == "sram_hash_bruteforce":
            best_model = _sram.SRAMHashBruteForceAttack(
                training.challenges,
                training.responses,
                target.response_length,
                seed,
            )
            selected_stage = _sram.HASH_BRUTE_FORCE_STAGE_NAME
        else:
            best_model = _logistic_regression.ArbiterLogisticRegression(
                training.challenges,
                training.responses,
                target.response_length,
                seed,
            )
            selected_stage = _logistic_regression.STAGE_NAME
        fit_seconds = time.perf_counter() - fit_start
        best_validation_accuracy = pair_accuracy_fn(best_model, validation)
        best_validation_element_accuracy = _response_element_accuracy(
            best_model, validation, family_key
        )
        selected_depth = 0
        selected_agent_count = 1
        threshold_reached = best_validation_accuracy >= success_threshold
        stage_history.append({
            "attack": selected_stage,
            "depth": 0,
            "agents": 1,
            "epochs": 0,
            "activation_types": _model_profile(
                family_key, 0, attack_model, device
            ),
            "validation_accuracy": best_validation_accuracy,
            "validation_element_accuracy": best_validation_element_accuracy,
            "fit_elapsed_seconds": float(fit_seconds),
        })
        if print_progress:
            print(
                f"{selected_stage} validation accuracy "
                f"{best_validation_accuracy:.4%}"
            )
        stages = []
    elif attack_model == "svm":
        fit_start = time.perf_counter()
        if family_key == "xor_apuf":
            import torch

            svm_device = (
                device if device is not None else torch.device("cpu")
            )
            best_model = _svm.TorchXorParitySVM(
                training.challenges,
                training.responses,
                target.response_length,
                target.k,
                seed,
                svm_device,
                epochs,
                batch_size,
                learning_rate,
            )
            epochs_trained = epochs
            agent_epochs_trained = epochs
            selected_stage = _svm.XOR_STAGE_NAME
        elif getattr(device, "type", None) == "xpu":
            best_model = _svm.TorchFeatureMappedSVM(
                training.challenges,
                training.responses,
                target.response_length,
                seed,
                device,
                epochs,
                batch_size,
                learning_rate,
                regression=family_key == "optical",
            )
            epochs_trained = epochs
            agent_epochs_trained = epochs
        else:
            best_model = _svm.FeatureMappedSVM(
                training.challenges,
                training.responses,
                target.response_length,
                seed,
                regression=family_key == "optical",
            )
        fit_seconds = time.perf_counter() - fit_start
        best_validation_accuracy = pair_accuracy_fn(best_model, validation)
        best_validation_element_accuracy = _response_element_accuracy(
            best_model, validation, family_key
        )
        if family_key != "xor_apuf":
            selected_stage = _svm.STAGE_NAME
        selected_depth = 0
        selected_agent_count = 1
        threshold_reached = best_validation_accuracy >= success_threshold
        stage_history.append({
            "attack": selected_stage,
            "depth": 0,
            "agents": 1,
            "epochs": epochs_trained,
            "activation_types": _model_profile(
                family_key, 0, attack_model, device
            ),
            "validation_accuracy": best_validation_accuracy,
            "validation_element_accuracy": best_validation_element_accuracy,
            "fit_elapsed_seconds": float(fit_seconds),
        })
        if print_progress:
            print(f"SVM validation accuracy {best_validation_accuracy:.4%}")
        stages = []
    elif family_key == "optical":
        import torch

        fit_start = time.perf_counter()
        optical_linear_model = _optical.bootstrap_linear_model(
            training.challenges,
            training.responses,
            target.response_length,
            device,
        )
        fit_seconds = time.perf_counter() - fit_start
        best_model = model_type(
            [optical_linear_model.cpu()],
            n,
            target.response_length,
            torch.device("cpu"),
        )
        best_validation_accuracy = pair_accuracy_fn(best_model, validation)
        best_validation_element_accuracy = _response_element_accuracy(
            best_model, validation, family_key
        )
        selected_stage = stage_name
        selected_depth = 0
        selected_agent_count = 1
        threshold_reached = best_validation_accuracy >= success_threshold
        stage_history.append({
            "attack": stage_name,
            "depth": 0,
            "agents": 1,
            "epochs": 0,
            "activation_types": _model_profile(
                family_key, 0, attack_model, device
            ),
            "validation_accuracy": best_validation_accuracy,
            "validation_element_accuracy": best_validation_element_accuracy,
            "fit_elapsed_seconds": float(fit_seconds),
        })
        if print_progress:
            print(
                f"Optical linear regression validation accuracy "
                f"{best_validation_accuracy:.4%}"
            )
        stages = []
    elif attack_model in {
        "xor_logistic_regression",
        "xor_quadratic",
        "xor_rank_two",
        "permutation_parity",
        "interpose_logistic_regression",
        "staged_split",
    }:
        stages = [(0, agents)]
    elif family_key in {"ring_oscillator", "memristive"}:
        stages = [(0, agents)]
    else:
        stages = [(layer_count, agents) for layer_count in range(1, max_depth + 1)]
    for depth, agent_count in stages:
        if threshold_reached or early_stopped:
            break
        trainers = []
        trainer_setup_start = time.perf_counter()
        for agent_index in range(agent_count):
            trainer_arguments = (
                training.challenges,
                training.responses,
                target.response_length,
                depth,
                hidden_width,
                seed + depth * agents + agent_index,
                batch_size,
                learning_rate,
                device,
            )
            if attack_model == "xor_logistic_regression":
                trainers.append(trainer_type(
                    training.challenges,
                    training.responses,
                    target.response_length,
                    seed + depth * agents + agent_index,
                    batch_size,
                    learning_rate,
                    device,
                ))
            elif attack_model in {"xor_quadratic", "xor_rank_two"}:
                trainers.append(trainer_type(
                    training.challenges,
                    training.responses,
                    target.response_length,
                    seed + depth * agents + agent_index,
                    batch_size,
                    learning_rate,
                    device,
                ))
            elif attack_model == "permutation_parity":
                trainers.append(trainer_type(
                    *trainer_arguments,
                    permutation_table=target._permutation_table,
                ))
            elif attack_model in {
                "interpose_logistic_regression",
                "staged_split",
            }:
                trainers.append(trainer_type(
                    *trainer_arguments,
                    down_chain_count=target.down.k,
                    interpose_pos=target.interpose_pos,
                ))
            elif attack_model == "deep_neural":
                trainers.append(trainer_type(
                    *trainer_arguments,
                    family_key=family_key,
                    chain_count=getattr(target, "k", None),
                    feed_forward_loops=getattr(target, "ff", None),
                    interpose_pos=getattr(target, "interpose_pos", None),
                    down_chain_count=getattr(
                        getattr(target, "down", None), "k", None
                    ),
                    permutation_table=getattr(
                        target, "_permutation_table", None
                    ),
                ))
            elif attack_model == "ff_fc_lstm":
                trainers.append(trainer_type(
                    *trainer_arguments,
                    feed_forward_loop=feed_forward_loop,
                ))
            elif attack_model == "fc_lstm":
                trainers.append(trainer_type(*trainer_arguments))
            elif family_key in {"xor_apuf", "xor_arbiter"}:
                trainers.append(trainer_type(*trainer_arguments, chain_count=target.k))
            elif family_key == "interpose":
                trainers.append(trainer_type(
                    *trainer_arguments,
                    down_chain_count=target.down.k,
                    interpose_pos=target.interpose_pos,
                ))
            else:
                trainers.append(trainer_type(*trainer_arguments))
        trainer_setup_seconds = time.perf_counter() - trainer_setup_start
        fit_seconds += trainer_setup_seconds
        if print_progress:
            print(f"\nCreated {agent_count} trainers for depth {depth}.")
        stage_history.append({
            "attack": stage_name,
            "depth": depth,
            "agents": agent_count,
            "epochs": 0,
            "activation_types": _model_profile(
                family_key, depth, attack_model, device
            ),
            "validation_accuracy": None,
            "validation_element_accuracy": None,
            "fit_elapsed_seconds": float(trainer_setup_seconds),
        })
        track_training_history = attack_model in {
            "deep_neural",
            "permutation_parity",
        } or select_by_element_accuracy
        if track_training_history:
            stage_history[-1]["epoch_history"] = []
        while True:
            epoch_start = time.perf_counter()
            for trainer in trainers:
                trainer.train_epoch()
            apply_threshold_calibration = (
                attack_model == "deep_neural"
                and family_key == "ff_apuf"
                and target.response_length > 1
            )
            ensemble_thresholds = (
                _calibrate_model_thresholds(
                    [trainer.model for trainer in trainers], validation, device
                )
                if apply_threshold_calibration
                else None
            )
            model = build_attack_model(
                [trainer.model for trainer in trainers],
                device,
                response_thresholds=ensemble_thresholds,
            )
            validation_accuracy = pair_accuracy_fn(model, validation)
            candidate_trainer_models = [[trainer.model for trainer in trainers]]
            candidate_thresholds = [ensemble_thresholds]
            candidate_accuracies = [validation_accuracy]
            candidate_element_accuracies = [
                _response_element_accuracy(model, validation, family_key)
            ]
            if agent_count > 1:
                # Restarts can converge to disagreeing local optima; averaging
                # their logits then cancels a confident correct agent against
                # a confident-but-wrong one. Also consider each agent alone so
                # one straggler cannot cap the whole ensemble's accuracy.
                for trainer in trainers:
                    solo_thresholds = (
                        _calibrate_model_thresholds(
                            [trainer.model], validation, device
                        )
                        if apply_threshold_calibration
                        else None
                    )
                    solo_model = build_attack_model(
                        [trainer.model], device, response_thresholds=solo_thresholds
                    )
                    candidate_trainer_models.append([trainer.model])
                    candidate_thresholds.append(solo_thresholds)
                    candidate_accuracies.append(pair_accuracy_fn(solo_model, validation))
                    candidate_element_accuracies.append(
                        _response_element_accuracy(
                            solo_model, validation, family_key
                        )
                    )
            best_candidate_index = max(
                range(len(candidate_accuracies)),
                key=lambda index: _validation_selection_key(
                    family_key,
                    target.response_length,
                    candidate_accuracies[index],
                    candidate_element_accuracies[index],
                ),
            )
            selected_trainer_models = candidate_trainer_models[best_candidate_index]
            selected_response_thresholds = candidate_thresholds[best_candidate_index]
            validation_accuracy = candidate_accuracies[best_candidate_index]
            validation_element_accuracy = candidate_element_accuracies[
                best_candidate_index
            ]
            for trainer in trainers:
                update_fn = getattr(trainer, "update_validation_feedback", None)
                if callable(update_fn):
                    update_fn(
                        validation_element_accuracy
                        if select_by_element_accuracy
                        else validation_accuracy
                    )
            lr_events = []
            for trainer in trainers:
                consume_fn = getattr(trainer, "consume_lr_events", None)
                if callable(consume_fn):
                    lr_events.extend(consume_fn())
            lr_note = ""
            if print_progress and lr_events:
                latest_event = lr_events[-1]
                lr_note = (
                    " | lr reduced to "
                    f"{latest_event['lr']:.1e}"
                    f" (EMA {latest_event['ema']:.4%})"
                )
            if epochs_trained <= MODEL_SCREENING_EPOCHS:
                screening_element_accuracy = max(
                    screening_element_accuracy, validation_element_accuracy
                )
                screening_passed = (
                    screening_passed
                    or validation_element_accuracy > MODEL_SCREENING_THRESHOLD
                )
            elapsed = time.perf_counter() - epoch_start
            fit_seconds += elapsed
            epochs_trained += 1
            agent_epochs_trained += agent_count
            stage_history[-1]["epochs"] += 1
            stage_history[-1]["validation_accuracy"] = validation_accuracy
            stage_history[-1]["validation_element_accuracy"] = (
                validation_element_accuracy
            )
            stage_history[-1]["fit_elapsed_seconds"] += float(elapsed)
            if track_training_history:
                stage_history[-1]["epoch_history"].append({
                    "epoch": stage_history[-1]["epochs"],
                    "validation_accuracy": float(validation_accuracy),
                    "validation_element_accuracy": float(
                        validation_element_accuracy
                    ),
                    "validation_emas": [
                        getattr(trainer, "validation_ema", None)
                        for trainer in trainers
                    ],
                    "learning_rates": [
                        getattr(trainer, "learning_rate", None)
                        for trainer in trainers
                    ],
                    "lr_events": lr_events,
                })
            if print_progress:
                if stage_history[-1]["epochs"] != 1:
                    print(CLEAR_PREVIOUS_LINE, end="")
                validation_note = (
                    f"validation accuracy {validation_accuracy:.4%}"
                )
                if select_by_element_accuracy:
                    validation_note += (
                        f", per-bit accuracy {validation_element_accuracy:.4%}"
                    )
                print(
                    f"Depth {depth}, trainers {agent_count}, "
                    f"epoch {stage_history[-1]['epochs']}/{epochs}, "
                    f"{validation_note}"
                    f"{lr_note}"
                )

            validation_selection_key = _validation_selection_key(
                family_key,
                target.response_length,
                validation_accuracy,
                validation_element_accuracy,
            )
            if validation_selection_key > best_validation_selection_key:
                import torch

                best_model = build_attack_model(
                    [
                        copy.deepcopy(trainer_model).cpu()
                        for trainer_model in selected_trainer_models
                    ],
                    torch.device("cpu"),
                    response_thresholds=selected_response_thresholds,
                )
                best_validation_accuracy = validation_accuracy
                best_validation_element_accuracy = validation_element_accuracy
                best_validation_selection_key = validation_selection_key
                selected_stage = stage_name
                selected_depth = depth
                selected_agent_count = len(selected_trainer_models)

            if validation_accuracy >= success_threshold:
                threshold_reached = True
                if print_progress:
                    print(
                        "Success threshold reached with validation accuracy "
                        f"{validation_accuracy:.4%}"
                    )
                break
            if (
                not screening_passed
                and epochs_trained >= MODEL_SCREENING_EPOCHS
                and screening_element_accuracy <= MODEL_SCREENING_THRESHOLD
            ):
                early_stopped = True
                early_stop_reason = (
                    "Per-response validation accuracy did not exceed 50% "
                    f"within the first {MODEL_SCREENING_EPOCHS} epochs."
                )
                if print_progress:
                    print(early_stop_reason + " Stopping this model.")
                break
            if stage_history[-1]["epochs"] >= epochs:
                if print_progress:
                    print(
                        f"Maximum epochs reached for depth {depth} with validation "
                        f"accuracy {validation_accuracy:.4%}"
                    )
                break
        if threshold_reached:
            break

    if best_model is None:
        raise RuntimeError("No attack model was trained.")
    validation_accuracy = best_validation_accuracy
    test_responses = np.asarray(test.responses).reshape(len(test.challenges), -1)
    attack_responses = np.asarray(best_model.eval(test.challenges)).reshape(
        len(test.challenges), -1
    )
    if family_key == "optical":
        response_matches = np.isclose(
            attack_responses,
            test_responses,
            rtol=_optical.RELATIVE_TOLERANCE,
            atol=_optical.ABSOLUTE_TOLERANCE,
        )
    else:
        response_matches = attack_responses == test_responses
    response_row_matches = np.all(response_matches, axis=1)
    test_accuracy = float(np.mean(response_row_matches))
    test_element_accuracy = float(np.mean(response_matches))
    if family_key == "optical":
        csv_test_responses = test_responses
        csv_attack_responses = attack_responses
    else:
        csv_test_responses = (test_responses > 0).astype(np.uint8)
        csv_attack_responses = (attack_responses > 0).astype(np.uint8)
    result = {
        "attack": selected_stage,
        "model_family": family_key,
        "model_architecture": attack_model,
        "activation_types": _model_profile(
            family_key, selected_depth, attack_model, device
        ),
        "depth": selected_depth,
        "agents": selected_agent_count,
        "validation_accuracy": validation_accuracy,
        "validation_element_accuracy": best_validation_element_accuracy,
        "selection_metric": (
            "response_element_accuracy"
            if select_by_element_accuracy
            else "complete_response_accuracy"
        ),
        "test_accuracy": test_accuracy,
        "test_element_accuracy": test_element_accuracy,
        "matched_test_responses": int(np.count_nonzero(response_row_matches)),
        "test_response_count": len(response_row_matches),
        "success_threshold": success_threshold,
        "success": test_accuracy >= success_threshold,
        "threshold_reached": threshold_reached,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "epochs_to_threshold": epochs_trained if threshold_reached else None,
        "time_to_threshold_seconds": fit_seconds if threshold_reached else None,
        "epochs_trained": epochs_trained,
        "agent_epochs_trained": agent_epochs_trained,
        "epochs_per_depth_limit": epochs,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "fit_elapsed_seconds": float(fit_seconds),
        "stages": stage_history,
        "_response_rows": (
            csv_test_responses,
            csv_attack_responses,
            response_row_matches,
        ),
    }
    if isinstance(best_model, _sram.SRAMHashBruteForceAttack):
        result["candidate_fingerprints_tested"] = best_model.candidate_limit
        result["training_match_count"] = best_model.training_match_count
        result["training_response_count"] = best_model.training_response_count
    return result


def _summarize_puf_instances(
    instance_results: Sequence[Dict[str, Any]], success_threshold: float
) -> Dict[str, Any]:
    instance_count = len(instance_results)
    matched_instances = sum(
        instance_result["instance_matched"] for instance_result in instance_results
    )
    epochs_trained = sum(
        instance_result["epochs_trained"] for instance_result in instance_results
    )
    fit_elapsed_seconds = sum(
        instance_result["fit_elapsed_seconds"] for instance_result in instance_results
    )
    mean_test_accuracy = float(np.mean([
        instance_result["test_accuracy"] for instance_result in instance_results
    ]))
    mean_test_element_accuracy = float(np.mean([
        instance_result["test_element_accuracy"]
        for instance_result in instance_results
    ]))

    return {
        "puf_instances": instance_count,
        "matched_instances": matched_instances,
        "instance_match_rate": matched_instances / instance_count,
        "mean_test_accuracy": mean_test_accuracy,
        "mean_test_element_accuracy": mean_test_element_accuracy,
        "success_threshold": success_threshold,
        "epochs_trained": epochs_trained,
        "fit_elapsed_seconds": fit_elapsed_seconds,
        "instances": list(instance_results),
    }


def run_ml_attack_experiment(
    families: Optional[Sequence[str]] = None,
    n: int = 64,
    training_samples: int = 10000,
    validation_samples: int = 2000,
    test_samples: int = 2000,
    seed: int = 20260926,
    k: int = 2,
    response_bits: int = 1,
    noisiness: float = 0.0,
    success_threshold: float = 0.95,
    max_depth: int = 3,
    agents: int = 3,
    width: Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    backend: str = "auto",
    device: Optional[int] = None,
    scenario: str = "general",
    report_root: Optional[str] = None,
    save_report: bool = True,
    puf_instances: int = 1000,
    print_progress: bool = False,
    attack_model: str = "auto",
    hash_responses: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Train separate models for a cohort of independent PUF instances.

    Each instance receives separate training, validation, and test CRP sets.
    Neural models train until validation reaches ``success_threshold`` or the
    per-depth epoch limit. Auto routing uses logistic regression for Arbiter and selects an XOR
    classifier from the effective chain group size: PyTorch logistic
    regression for one chain per bit, quadratic ATT features for two chains,
    and the parity-aware XOR neural model for larger groups. Interpose auto
    routing uses a structure-aware linear/logistic model; ``staged_split``
    alternates upstream and downstream component updates. Feed-forward
    Arbiter PUFs use the structure-aware deep neural attack. Optical PUFs use
    their structure-aware attacks. Permutation PUFs use chain-specific
    permuted ATT features and parity-composed Arbiter scores.
    Ring-oscillator PUFs use per-response-bit logistic regression on raw
    challenges. Unhashed SRAM responses use direct per-bit logistic
    regression; hashed SRAM responses use a bounded random fingerprint search.
    Explicit ``neural`` or ``svm`` selection overrides these choices except
    for optical PUFs. ``deep_neural`` selects a structure-aware PyTorch DNN
    for feed-forward Arbiter, XOR Arbiter, Interpose, and Permutation PUFs.
    ``staged_split`` is limited to Interpose PUFs. An instance counts as
    matched when held-out test accuracy reaches ``success_threshold``.
    """
    selected_families = tuple(
        DEFAULT_ATTACK_FAMILIES if families is None else families
    )
    if not selected_families:
        raise ValueError("At least one PUF family is required.")
    attack_model = attack_model.strip().lower()
    if attack_model not in {
        "auto",
        "neural",
        "svm",
        "deep_neural",
        "staged_split",
    }:
        raise ValueError(
            "attack_model must be 'auto', 'neural', 'svm', 'deep_neural', "
            "or 'staged_split'."
        )
    normalized_backend = backend.strip().lower()
    normalized_family_keys = {
        family: _normalize_family_key(family) for family in selected_families
    }
    unknown = [
        family
        for family, family_key in normalized_family_keys.items()
        if family_key not in PUF_IMPLEMENTATIONS
    ]
    if unknown:
        raise ValueError(f"Unknown PUF families: {', '.join(unknown)}")
    if attack_model == "staged_split" and any(
        family_key != "interpose"
        for family_key in normalized_family_keys.values()
    ):
        raise ValueError(
            "attack_model='staged_split' is only supported for the interpose family."
        )
    attack_model_by_family = {
        family: _resolve_attack_model_for_family(
            family_key,
            attack_model,
            n=n,
            k=k,
            response_bits=response_bits,
            hash_responses=hash_responses,
        )
        for family, family_key in normalized_family_keys.items()
    }
    if attack_model == "deep_neural":
        unsupported_families = [
            family
            for family, family_key in normalized_family_keys.items()
            if family_key not in _arbiter.deep_neural.SUPPORTED_FAMILIES
        ]
        if unsupported_families:
            raise ValueError(
                "attack_model='deep_neural' supports only ff_apuf, xor_apuf, "
                "interpose, and permutation; unsupported families: "
                + ", ".join(unsupported_families)
            )
    uses_svm = any(
        family_attack_model == "svm"
        for family_attack_model in attack_model_by_family.values()
    )
    if uses_svm and normalized_backend not in {
        "auto", "cpu", "torch_xpu"
    }:
        raise ValueError("backend must be 'auto', 'cpu', or 'torch_xpu'.")
    if uses_svm and device is not None and normalized_backend != "torch_xpu":
        raise ValueError("device can only be set when backend='torch_xpu'.")
    if n < 1 or min(
        training_samples, validation_samples, test_samples, epochs, batch_size, puf_instances
    ) < 1:
        raise ValueError("Dimensions, sample counts, epochs, and batch size must be positive.")
    if not 0.5 < success_threshold <= 1.0:
        raise ValueError("success_threshold must be greater than 0.5 and at most 1.0.")
    if max_depth < 1 or agents < 1 or (width is not None and width < 1):
        raise ValueError("max_depth must be at least 1; agents and width must be positive.")
    results: Dict[str, Dict[str, Any]] = {}
    response_data: Dict[
        str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {}
    for family_index, family in enumerate(selected_families):
        family_attack_model = attack_model_by_family[family]
        training_device = (
            None
            if family_attack_model in {
                "logistic_regression",
                "sram_logistic_regression",
                "sram_hash_bruteforce",
                "ring_oscillator_logistic_regression",
            }
            or (family_attack_model == "svm" and normalized_backend == "cpu")
            else _resolve_device(backend, device)
        )
        instance_results = []
        for instance_index in range(puf_instances):
            print(
                f"{CLEAR_PREVIOUS_LINE}\nAttacking {family!r}: "
                f"instance {instance_index + 1}/{puf_instances}...",
                end="",
                flush=True,
            )
            instance_seed = seed + (family_index * puf_instances + instance_index) * 4
            target = create_puf(
                family,
                n=n,
                seed=instance_seed,
                k=k,
                response_bits=response_bits,
                noisiness=noisiness,
                hash_responses=hash_responses,
            )
            instance_result = _attack_one_puf(
                target,
                n,
                training_samples,
                validation_samples,
                test_samples,
                instance_seed + 1,
                success_threshold,
                max_depth,
                agents,
                width,
                epochs,
                batch_size,
                learning_rate,
                training_device,
                family,
                print_progress,
                family_attack_model,
            )
            response_rows = instance_result.pop("_response_rows", None)
            if save_report:
                if response_rows is None:
                    raise RuntimeError(
                        "ML attack result did not include test response rows "
                        "required for the report."
                    )
                response_data.setdefault(family, []).append(response_rows)
            instance_result.update({
                "instance_index": instance_index + 1,
                "instance_seed": instance_seed,
                "instance_matched": (
                    instance_result["test_accuracy"] >= success_threshold
                ),
            })
            if print_progress:
                print(
                    f"\nTest results for {family!r}, instance "
                    f"{instance_index + 1}/{puf_instances}: "
                    f"{instance_result['test_accuracy']:.4%} accuracy "
                    f"({instance_result['matched_test_responses']}/"
                    f"{instance_result['test_response_count']} complete responses; "
                    f"required threshold {success_threshold:.4%}; "
                    f"{instance_result['test_element_accuracy']:.4%} "
                    "response-element accuracy)."
                )
            instance_results.append(instance_result)
        print()
        family_results = _summarize_puf_instances(instance_results, success_threshold)
        results[family] = family_results
        if print_progress:
            print(
                f"Test summary for {family!r}: "
                f"mean test accuracy {family_results['mean_test_accuracy']:.4%} "
                f"across {puf_instances} instances; required threshold "
                f"{success_threshold:.4%} per instance. Mean response-element "
                f"accuracy {family_results['mean_test_element_accuracy']:.4%}."
            )

    if save_report:
        parameters = {
            "n": n,
            "training_samples": training_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "puf_instances": puf_instances,
            "seed": seed,
            "k": k,
            "response_bits": response_bits,
            "noisiness": noisiness,
            "hash_responses": hash_responses,
            "success_threshold": success_threshold,
            "max_depth": max_depth,
            "agents": agents,
            "width": width,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "backend": backend,
            "device": (
                "cpu (scikit-learn)"
                if all(
                    attack_model_by_family[family]
                    in {
                        "logistic_regression",
                        "sram_logistic_regression",
                        "sram_hash_bruteforce",
                        "ring_oscillator_logistic_regression",
                    }
                    for family in selected_families
                )
                else "cpu (PyTorch XOR parity SVM; scikit-learn otherwise)"
                if normalized_backend == "cpu"
                and any(
                    attack_model_by_family[family] == "svm"
                    and normalized_family_keys[family] == "xor_apuf"
                    for family in selected_families
                )
                else "cpu (scikit-learn)"
                if normalized_backend == "cpu"
                and any(
                    attack_model_by_family[family] == "svm"
                    and normalized_family_keys[family] != "xor_apuf"
                    for family in selected_families
                )
                else str(_resolve_device(backend, device))
            ),
            "scenario": scenario,
            "attack_model": attack_model,
            "attack_model_by_family": attack_model_by_family,
            "default_families": DEFAULT_ATTACK_FAMILIES,
        }
        report_directory = save_ml_attack_report(
            results,
            parameters,
            selected_families,
            scenario,
            report_root,
            responses=response_data,
        )
        print(f"Saved ML attack report to: {report_directory}")
    return results


__all__ = ["DEFAULT_ATTACK_FAMILIES", "run_ml_attack_experiment"]