"""CRP generation and model search when the target PUF family is unknown."""
from __future__ import annotations

import csv
import gc
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from pypuf.io import ChallengeResponseSet

from puf_sim.puf_implementations import PUF_IMPLEMENTATIONS, create_puf

from . import experiment as _experiment

DEFAULT_CRPS_ROOT = Path(__file__).resolve().parent / "data_crps"
DEFAULT_UNKNOWN_REPORT_ROOT = (
    Path(__file__).resolve().parent.parent / "reports_unknown_puf_attacks"
)

_PURE_CPU_ATTACKS = {
    "logistic_regression",
    "sram_logistic_regression",
    "sram_hash_bruteforce",
    "ring_oscillator_logistic_regression",
}
_LOGISTIC_ATTACKS = {
    "logistic_regression",
    "sram_logistic_regression",
    "ring_oscillator_logistic_regression",
    "xor_logistic_regression",
    "interpose_logistic_regression",
}


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    if not name:
        raise ValueError("Names must contain at least one valid path character.")
    return name


def _canonical_family(family: str) -> str:
    key = _experiment._normalize_family_key(family)
    if key not in PUF_IMPLEMENTATIONS:
        raise ValueError(f"Unknown PUF family {family!r}.")
    return key


def _normalize_responses(responses: np.ndarray, sample_count: int) -> np.ndarray:
    response_array = np.asarray(responses)
    if response_array.ndim == 1:
        response_array = response_array.reshape(sample_count, 1)
    elif response_array.ndim == 3 and response_array.shape[-1] == 1:
        response_array = response_array[..., 0]
    if response_array.ndim != 2 or response_array.shape[0] != sample_count:
        raise ValueError(
            "PUF responses must have shape (samples, response_bits), or "
            "(samples, response_bits, 1)."
        )
    if not np.issubdtype(response_array.dtype, np.number) or not np.all(
        np.isfinite(response_array)
    ):
        raise ValueError("PUF responses must contain finite numeric values.")
    return response_array


def _dataset_directory(root: Path, dataset_name: Optional[str]) -> Path:
    if dataset_name is None:
        dataset_name = datetime.now().strftime("dataset_%Y%m%d_%H%M%S_%f")
    target = root / _safe_name(dataset_name)
    target.mkdir(parents=True, exist_ok=False)
    return target


def generate_crps(
    family_instance_counts: Mapping[str, int],
    n: int,
    samples_per_instance: Optional[int] = None,
    total_samples: Optional[int] = None,
    seed: int = 20261020,
    k: Optional[int] = 2,
    response_bits: int = 1,
    noisiness: Optional[float] = None,
    ff: Optional[Sequence[tuple[int, int]]] = None,
    interpose_pos: Optional[int] = None,
    bch_correction_capacity: int = 1,
    hash_responses: bool = True,
    data_root: Optional[str | Path] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """Generate opaque, instance-grouped CRP files and return their directory.

    Exactly one of ``samples_per_instance`` or ``total_samples`` is required.
    The returned directory contains one ``.npz`` file per PUF instance and a
    manifest that identifies instances without recording their PUF families.
    """
    if (samples_per_instance is None) == (total_samples is None):
        raise ValueError(
            "Specify exactly one of samples_per_instance or total_samples."
        )
    if n < 1 or response_bits < 1:
        raise ValueError("n and response_bits must be positive.")
    if samples_per_instance is not None and samples_per_instance < 10:
        raise ValueError("samples_per_instance must be at least 10.")
    if total_samples is not None and total_samples < 10:
        raise ValueError("total_samples must be at least 10.")
    if not family_instance_counts:
        raise ValueError("At least one PUF family is required.")

    requested_instances: list[str] = []
    for family, count in family_instance_counts.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(
                f"Instance count for {family!r} must be a positive integer."
            )
        canonical_family = _canonical_family(family)
        requested_instances.extend([canonical_family] * count)

    instance_count = len(requested_instances)
    if total_samples is not None and total_samples < instance_count * 10:
        raise ValueError(
            "total_samples must provide at least 10 CRPs per PUF instance."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(requested_instances)
    if samples_per_instance is not None:
        sample_counts = [samples_per_instance] * instance_count
    else:
        assert total_samples is not None
        base_count, remainder = divmod(total_samples, instance_count)
        sample_counts = [
            base_count + (instance_index < remainder)
            for instance_index in range(instance_count)
        ]

    root = DEFAULT_CRPS_ROOT if data_root is None else Path(data_root)
    dataset_directory = _dataset_directory(root, dataset_name)
    instance_metadata = []
    for instance_index, (family, sample_count) in enumerate(
        zip(requested_instances, sample_counts), start=1
    ):
        puf_seed = seed + 2 * instance_index
        challenge_seed = puf_seed + 1
        puf = create_puf(
            family,
            n=n,
            seed=puf_seed,
            k=k,
            response_bits=response_bits,
            noisiness=noisiness,
            ff=ff,
            interpose_pos=interpose_pos,
            bch_correction_capacity=bch_correction_capacity,
            hash_responses=hash_responses,
        )
        challenge_rng = np.random.default_rng(challenge_seed)
        challenges = challenge_rng.choice(
            (-1, 1), size=(sample_count, n)
        ).astype(np.int8)
        responses = _normalize_responses(
            np.asarray(puf.eval(challenges)), sample_count
        )
        instance_id = f"instance_{instance_index:04d}"
        filename = f"{instance_id}.npz"
        np.savez_compressed(
            dataset_directory / filename,
            challenges=challenges,
            responses=responses,
        )
        instance_metadata.append({
            "instance_id": instance_id,
            "file": filename,
            "sample_count": sample_count,
            "challenge_length": int(challenges.shape[1]),
            "response_length": int(responses.shape[1]),
        })

    manifest = {
        "format_version": 1,
        "dataset_id": dataset_directory.name,
        "instance_count": instance_count,
        "total_samples": int(sum(sample_counts)),
        "seed": seed,
        "n": n,
        "k": k,
        "response_bits": response_bits,
        "noisiness": noisiness,
        "instances": instance_metadata,
    }
    with (dataset_directory / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    return dataset_directory


def _load_instances(
    dataset_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(dataset_path)
    if path.is_file() and path.suffix.lower() == ".npz":
        manifest = {
            "dataset_id": path.stem,
            "instances": [{
                "instance_id": path.stem,
                "file": path.name,
            }],
        }
        root = path.parent
    elif path.is_dir():
        root = path
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Dataset manifest not found: {manifest_path}"
            )
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest.get("instances"), list):
            raise ValueError("Dataset manifest must contain an instances list.")
    else:
        raise FileNotFoundError(f"CRP dataset not found: {path}")

    instances = []
    seen_ids: set[str] = set()
    for record in manifest["instances"]:
        instance_id = str(record.get("instance_id", "")).strip()
        filename = str(record.get("file", "")).strip()
        if not instance_id or not filename or instance_id in seen_ids:
            raise ValueError(
                "Every dataset instance needs a unique ID and a data filename."
            )
        seen_ids.add(instance_id)
        data_path = root / filename
        with np.load(data_path, allow_pickle=False) as data:
            if "challenges" not in data or "responses" not in data:
                raise ValueError(
                    f"{data_path} must contain challenges and responses arrays."
                )
            challenges = np.asarray(data["challenges"])
            responses = np.asarray(data["responses"])
        if (
            challenges.ndim != 2
            or len(challenges) < 10
            or challenges.shape[1] < 1
            or not np.issubdtype(challenges.dtype, np.number)
            or not np.all(np.isfinite(challenges))
        ):
            raise ValueError(
                f"Invalid challenge array for instance {instance_id!r}."
            )
        responses = _normalize_responses(responses, len(challenges))
        if np.isin(responses, (0, 1)).all() and not np.isin(
            responses, (-1, 1)
        ).all():
            responses = (responses * 2 - 1).astype(np.int8)
        if challenges.shape[0] != responses.shape[0]:
            raise ValueError(
                f"Challenge/response row counts differ for {instance_id!r}."
            )
        instances.append({
            "instance_id": instance_id,
            "challenges": challenges,
            "responses": responses,
        })
    if not instances:
        raise ValueError("CRP dataset contains no PUF instances.")
    return instances, manifest


def _split_indices(
    challenges: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = len(challenges)
    if sample_count < 10:
        raise ValueError("Each instance needs at least 10 CRPs to split.")
    target_counts = np.asarray([
        int(sample_count * 0.7),
        int(sample_count * 0.2),
        sample_count - int(sample_count * 0.7) - int(sample_count * 0.2),
    ])
    unique_challenges, inverse = np.unique(
        challenges, axis=0, return_inverse=True
    )
    if len(unique_challenges) < 3:
        raise ValueError(
            "At least three distinct challenges per instance are required "
            "for disjoint train/validation/test splits."
        )
    groups = [
        np.flatnonzero(inverse == group_index)
        for group_index in range(len(unique_challenges))
    ]
    rng = np.random.default_rng(seed)
    group_order = rng.permutation(len(groups))
    split_rows: list[list[int]] = [[], [], []]
    split_counts = np.zeros(3, dtype=int)
    for position, group_index in enumerate(group_order):
        group = groups[int(group_index)]
        empty_splits = np.flatnonzero(split_counts == 0)
        groups_remaining = len(group_order) - position
        if groups_remaining <= len(empty_splits):
            destination = int(
                empty_splits[np.argmax(target_counts[empty_splits])]
            )
        else:
            deficits = target_counts - split_counts
            destination = int(np.argmax(deficits))
        split_rows[destination].extend(group.tolist())
        split_counts[destination] += len(group)
    if np.any(split_counts == 0):
        raise ValueError(
            "Unable to form non-empty train/validation/test splits from "
            "distinct challenges."
        )
    partitions = []
    for rows in split_rows:
        partition = np.asarray(rows, dtype=np.int64)
        partitions.append(partition[rng.permutation(len(partition))])
    return partitions[0], partitions[1], partitions[2]


def _candidate_specs(
    n: int,
    k: Optional[int],
    response_bits: int,
    candidate_families: Sequence[str],
    attack_models: Optional[Sequence[str]],
    continuous_responses: bool = False,
    candidate_model_types: Optional[Mapping[str, set[str]]] = None,
) -> list[dict[str, Any]]:
    allowed_models = None if attack_models is None else {
        model.strip().lower() for model in attack_models
    }
    if allowed_models is not None and not allowed_models:
        raise ValueError("attack_models cannot be empty.")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, bool]] = set()

    def add(
        family: str,
        attack_model: str,
        hash_responses: bool = True,
        is_auto: bool = False,
    ) -> None:
        key = (family, attack_model, hash_responses)
        if key in seen:
            return
        if (
            allowed_models is not None
            and attack_model not in allowed_models
            and not (is_auto and "auto" in allowed_models)
        ):
            return
        family_models = (
            None
            if candidate_model_types is None
            else candidate_model_types.get(family)
        )
        if (
            family_models is not None
            and attack_model not in family_models
            and not (is_auto and "auto" in family_models)
        ):
            return
        seen.add(key)
        candidates.append({
            "family_hypothesis": family,
            "attack_model": attack_model,
            "hash_responses": hash_responses,
        })

    normalized_families = []
    for family in candidate_families:
        key = _canonical_family(family)
        if key not in normalized_families:
            normalized_families.append(key)
    if not normalized_families:
        raise ValueError("At least one candidate PUF family is required.")

    if continuous_responses:
        if "optical" not in normalized_families:
            raise ValueError(
                "Continuous responses require 'optical' in candidate_families."
            )
        add("optical", "optical_regression", is_auto=True)
        add("optical", "svm")
        if not candidates:
            raise ValueError(
                "No continuous-response model candidates match attack_models."
            )
        return candidates

    for family in normalized_families:
        if family != "optical":
            add(family, "linear_logits")
            add(family, "neural")
            add(family, "svm")
            add(family, "fc_lstm")
        add(
            family,
            _experiment._resolve_attack_model_for_family(
                family,
                "auto",
                n=n,
                k=k,
                response_bits=response_bits,
                hash_responses=True,
            ),
            is_auto=True,
        )
        if family in _experiment._arbiter.deep_neural.SUPPORTED_FAMILIES:
            add(family, "deep_neural")
        if family == "ff_apuf":
            add(family, "ff_fc_lstm")
        elif family == "xor_apuf":
            effective_chain_count = n if k is None else max(k, response_bits)
            chains_per_bit, remainder = divmod(
                effective_chain_count, response_bits
            )
            if remainder == 0 and chains_per_bit == 1:
                add(family, "xor_logistic_regression")
            elif remainder == 0 and chains_per_bit == 2:
                add(family, "xor_quadratic")
                add(family, "xor_rank_two")
        elif family == "interpose":
            add(family, "staged_split")
        elif family == "sram":
            add(family, "sram_logistic_regression", hash_responses=False)
            add(family, "sram_hash_bruteforce", hash_responses=True)

    if not candidates:
        raise ValueError("No supported model candidates match attack_models.")
    candidates.sort(
        key=lambda candidate: (
            0
            if candidate["attack_model"] in {
                "linear_logits",
                "optical_regression",
            }
            else 1
            if candidate["attack_model"] in _LOGISTIC_ATTACKS
            else 2
        )
    )
    return candidates


def _attack_summary(
    result: Mapping[str, Any], candidate: Mapping[str, Any], bagging_run: int
) -> dict[str, Any]:
    return {
        "status": (
            "excluded" if result.get("candidate_excluded", False) else "trained"
        ),
        "bagging_run": bagging_run,
        "family_hypothesis": candidate["family_hypothesis"],
        "model_type": candidate["attack_model"],
        "attack": result["attack"],
        "depth": result["depth"],
        "validation_accuracy": result["validation_accuracy"],
        "validation_element_accuracy": result["validation_element_accuracy"],
        "test_accuracy": result["test_accuracy"],
        "test_element_accuracy": result["test_element_accuracy"],
        "training_samples": result["training_samples"],
        "validation_samples": result["validation_samples"],
        "test_samples": result["test_samples"],
        "epochs_trained": result["epochs_trained"],
        "fit_elapsed_seconds": result["fit_elapsed_seconds"],
        "exclusion_reason": result.get("exclusion_reason"),
        "excluded_depths": result.get("excluded_depths", []),
    }


def _rank_trials(
    trial_records: Sequence[Mapping[str, Any]],
) -> tuple[
    Optional[dict[str, Any]],
    Optional[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not trial_records:
        return None, None, []

    grouped_trials: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for trial in trial_records:
        summary = trial["summary"]
        group_key = (
            summary["family_hypothesis"],
            summary["model_type"],
        )
        grouped_trials.setdefault(group_key, []).append(trial)
    model_summaries = []
    for (family_hypothesis, model_type), grouped in grouped_trials.items():
        model_summaries.append({
            "family_hypothesis": family_hypothesis,
            "model_type": model_type,
            "bagging_runs": len(grouped),
            "mean_validation_accuracy": float(np.mean([
                trial["summary"]["validation_accuracy"]
                for trial in grouped
            ])),
            "mean_validation_element_accuracy": float(np.mean([
                trial["summary"]["validation_element_accuracy"]
                for trial in grouped
            ])),
            "mean_test_accuracy": float(np.mean([
                trial["summary"]["test_accuracy"]
                for trial in grouped
            ])),
            "mean_test_element_accuracy": float(np.mean([
                trial["summary"]["test_element_accuracy"]
                for trial in grouped
            ])),
        })
    best_validation_group = max(
        model_summaries,
        key=lambda summary: (
            summary["mean_validation_accuracy"],
            summary["mean_validation_element_accuracy"],
        ),
    )
    selected_validation_trials = grouped_trials[
        (
            best_validation_group["family_hypothesis"],
            best_validation_group["model_type"],
        )
    ]
    best_validation_trial = max(
        selected_validation_trials,
        key=lambda trial: (
            trial["summary"]["validation_accuracy"],
            trial["summary"]["validation_element_accuracy"],
        ),
    )
    best_validation = {
        **best_validation_trial,
        "summary": {
            **best_validation_trial["summary"],
            **best_validation_group,
            "selected_bagging_run": (
                best_validation_trial["summary"]["bagging_run"]
            ),
        },
    }
    best_test_oracle_trial = max(
        trial_records,
        key=lambda trial: (
            trial["summary"]["test_accuracy"],
            trial["summary"]["test_element_accuracy"],
        ),
    )
    test_oracle_model_summary = next(
        summary
        for summary in model_summaries
        if summary["family_hypothesis"]
        == best_test_oracle_trial["summary"]["family_hypothesis"]
        and summary["model_type"]
        == best_test_oracle_trial["summary"]["model_type"]
    )
    best_test_oracle = {
        **best_test_oracle_trial,
        "summary": {
            **best_test_oracle_trial["summary"],
            **test_oracle_model_summary,
            "selected_bagging_run": (
                best_test_oracle_trial["summary"]["bagging_run"]
            ),
        },
    }
    return best_validation, best_test_oracle, model_summaries


def _rank_overall_trials(
    trial_summaries: Sequence[Mapping[str, Any]],
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    grouped_trials: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for trial in trial_summaries:
        if trial["test_accuracy"] is None:
            continue
        group_key = (
            trial["family_hypothesis"],
            trial["model_type"],
        )
        grouped_trials.setdefault(group_key, []).append(trial)
    if not grouped_trials:
        return None, []

    model_summaries = []
    for (family_hypothesis, model_type), grouped in grouped_trials.items():
        trials_by_instance: dict[str, list[Mapping[str, Any]]] = {}
        for trial in grouped:
            trials_by_instance.setdefault(trial["instance_id"], []).append(
                trial
            )
        model_summaries.append({
            "family_hypothesis": family_hypothesis,
            "model_type": model_type,
            "bagging_trials": len(grouped),
            "instances": len(trials_by_instance),
            "mean_test_accuracy": float(np.mean([
                np.mean([
                    trial["test_accuracy"] for trial in instance_trials
                ])
                for instance_trials in trials_by_instance.values()
            ])),
            "mean_test_element_accuracy": float(np.mean([
                np.mean([
                    trial["test_element_accuracy"]
                    for trial in instance_trials
                ])
                for instance_trials in trials_by_instance.values()
            ])),
            "mean_trial_elapsed_seconds": float(np.mean([
                trial["trial_elapsed_seconds"] for trial in grouped
            ])),
            "total_trial_elapsed_seconds": float(np.sum([
                trial["trial_elapsed_seconds"] for trial in grouped
            ])),
        })
    model_summaries.sort(
        key=lambda summary: summary["mean_test_accuracy"],
        reverse=True,
    )
    return model_summaries[0], model_summaries


def _is_xpu_resource_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return (
        "ur_result_error_out_of_resources" in message
        or "out of resources" in message
    )


def _clear_xpu_cache() -> None:
    import torch

    gc.collect()
    torch.xpu.empty_cache()


def _report_directory(
    root: Path, scenario: str, dataset_id: str
) -> Path:
    safe_scenario = _safe_name(scenario)
    safe_dataset = _safe_name(dataset_id)
    report_root = root / safe_scenario / safe_dataset
    report_root.mkdir(parents=True, exist_ok=True)
    report_number = 1 + sum(
        child.is_dir() for child in report_root.iterdir()
    )
    date_label = datetime.now().strftime("%Y %m %d")
    report_directory = report_root / (
        f"{date_label} Report {report_number} - Unknown PUFs"
    )
    report_directory.mkdir()
    return report_directory


def _write_response_csv(
    path: Path,
    chosen_instances: Sequence[Mapping[str, Any]],
    selection_basis: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "instance_id",
            "bagging_run",
            "selection_basis",
            "model_type",
            "effective_batch_size",
            "family_hypothesis",
            "test_accuracy",
            "sample_index",
            "puf_response",
            "model_response",
            "match",
        ])
        for instance in chosen_instances:
            winner = instance.get(selection_basis)
            if winner is None:
                continue
            responses = winner["response_rows"]
            puf_responses, model_responses, row_matches = responses
            for sample_index, (puf_row, model_row, matched) in enumerate(
                zip(puf_responses, model_responses, row_matches), start=1
            ):
                writer.writerow([
                    instance["instance_id"],
                    winner["summary"]["bagging_run"],
                    selection_basis,
                    winner["summary"]["model_type"],
                    winner["summary"]["effective_batch_size"],
                    winner["summary"]["family_hypothesis"],
                    winner["summary"]["test_accuracy"],
                    sample_index,
                    json.dumps(np.asarray(puf_row).tolist()),
                    json.dumps(np.asarray(model_row).tolist()),
                    int(matched),
                ])


def _serialize_instance_results(
    instance_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in instance.items()
            if key not in {"best_validation_selected", "best_test_oracle"}
        } | {
            "best_validation_selected": (
                instance["best_validation_selected"]["summary"]
                if instance["best_validation_selected"] is not None
                else None
            ),
            "best_test_oracle": (
                instance["best_test_oracle"]["summary"]
                if instance["best_test_oracle"] is not None
                else None
            ),
            "bagging_model_summaries": instance["bagging_model_summaries"],
        }
        for instance in instance_results
    ]


def _write_unknown_report(
    report_directory: Path,
    results: Mapping[str, Any],
    parameters: Mapping[str, Any],
    all_trials: Sequence[Mapping[str, Any]],
    instance_results: Sequence[Mapping[str, Any]],
) -> None:
    with (report_directory / "results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, indent=2)
    with (report_directory / "parameters.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(parameters, handle, indent=2)
    score_fields = [
        "status",
        "instance_id",
        "bagging_run",
        "family_hypothesis",
        "model_type",
        "attack",
        "exclusion_reason",
        "excluded_depths",
        "effective_batch_size",
        "depth",
        "training_samples",
        "validation_samples",
        "test_samples",
        "validation_accuracy",
        "validation_element_accuracy",
        "test_accuracy",
        "test_element_accuracy",
        "epochs_trained",
        "fit_elapsed_seconds",
        "trial_elapsed_seconds",
        "total_elapsed_seconds",
        "split_counts",
    ]
    with (report_directory / "candidate_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields)
        writer.writeheader()
        for trial in all_trials:
            row = dict(trial)
            row["split_counts"] = json.dumps(row["split_counts"], sort_keys=True)
            row["excluded_depths"] = json.dumps(
                row["excluded_depths"], sort_keys=True
            )
            writer.writerow(row)
    _write_response_csv(
        report_directory / "responses.csv",
        instance_results,
        "best_test_oracle",
    )
    _write_response_csv(
        report_directory / "responses_validation_selected.csv",
        instance_results,
        "best_validation_selected",
    )


def run_attack_unknown_puf(
    dataset_path: str | Path,
    bagging_runs: int = 5,
    seed: int = 20261020,
    k: Optional[int] = 2,
    candidate_families: Optional[Sequence[str]] = None,
    attack_models: Optional[Sequence[str]] = None,
    success_threshold: float = 0.95,
    max_depth: int = 3,
    agents: int = 1,
    width: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    backend: str = "auto",
    device: Optional[int] = None,
    noisiness: float = 0.0,
    bch_correction_capacity: int = 1,
    family_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    scenario: str = "general",
    report_root: Optional[str | Path] = None,
    print_progress: bool = False,
    max_total_epochs: Optional[int] = None,
    candidate_model_types: Optional[
        Mapping[str, Sequence[str] | str]
    ] = None,
    family_hypothesis: Optional[str | Sequence[str]] = None,
) -> dict[str, Any]:
    """Compare attack candidates over repeated disjoint 70/20/10 splits.

    Each CRP file is treated as one opaque PUF instance. By default, all
    supported families and their compatible attacks are tested. Families in
    ``candidate_families`` and ``family_hypothesis`` select optional model
    hypotheses only; they are never read from the CRP dataset.
    ``candidate_model_types`` optionally restricts the attack models for
    individual family hypotheses. The test-set winner is reported as an
    oracle result, because selecting a model by test accuracy uses test labels.
    """
    timer_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_started = time.perf_counter()
    if bagging_runs < 1:
        raise ValueError("bagging_runs must be positive.")
    if candidate_families is not None and family_hypothesis is not None:
        raise ValueError(
            "Specify either candidate_families or family_hypothesis, not both."
        )
    if epochs is not None and max_total_epochs is not None:
        raise ValueError("Specify either epochs or max_total_epochs, not both.")
    effective_max_total_epochs = (
        max_total_epochs
        if max_total_epochs is not None
        else epochs if epochs is not None else 200
    )
    if (
        agents < 1
        or max_depth < 1
        or effective_max_total_epochs < 1
        or batch_size < 1
    ):
        raise ValueError(
            "agents, max_depth, max_total_epochs, and batch_size must be positive."
        )
    if not 0.5 < success_threshold <= 1.0:
        raise ValueError("success_threshold must be greater than 0.5 and at most 1.")
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"auto", "cpu", "torch_xpu"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'torch_xpu'.")
    if device is not None and normalized_backend != "torch_xpu":
        raise ValueError("device can only be set when backend='torch_xpu'.")
    normalized_family_options: dict[str, dict[str, Any]] = {}
    for family, options in (family_options or {}).items():
        family_key = _canonical_family(family)
        option_values = dict(options)
        unsupported_options = set(option_values) - {
            "ff",
            "interpose_pos",
            "bch_correction_capacity",
        }
        if unsupported_options:
            raise ValueError(
                f"Unsupported candidate options for {family!r}: "
                + ", ".join(sorted(unsupported_options))
            )
        normalized_family_options[family_key] = option_values
    instances, manifest = _load_instances(dataset_path)
    selected_families_input = (
        family_hypothesis
        if family_hypothesis is not None
        else candidate_families
    )
    if selected_families_input is None:
        selected_families_input = _experiment.DEFAULT_ATTACK_FAMILIES
    elif isinstance(selected_families_input, str):
        selected_families_input = [selected_families_input]
    selected_families = list(dict.fromkeys(
        _canonical_family(family) for family in selected_families_input
    ))
    if not selected_families:
        raise ValueError("At least one candidate PUF family is required.")

    normalized_candidate_model_types: Optional[dict[str, set[str]]] = None
    if candidate_model_types is not None:
        if not candidate_model_types:
            raise ValueError("candidate_model_types cannot be empty.")
        normalized_candidate_model_types = {}
        for family, model_types in candidate_model_types.items():
            family_key = _canonical_family(family)
            if family_key not in selected_families:
                raise ValueError(
                    f"candidate_model_types family {family!r} is not selected "
                    "by family_hypothesis/candidate_families."
                )
            if isinstance(model_types, str):
                model_values = [model_types]
            elif isinstance(model_types, Sequence):
                model_values = model_types
            else:
                raise ValueError(
                    f"candidate_model_types for {family!r} must be a model "
                    "type or a sequence of model types."
                )
            if not all(
                isinstance(model, str) and model.strip()
                for model in model_values
            ):
                raise ValueError(
                    f"candidate_model_types for {family!r} must contain "
                    "non-empty model type names."
                )
            normalized_models = {
                model.strip().lower()
                for model in model_values
            }
            if not normalized_models:
                raise ValueError(
                    f"candidate_model_types for {family!r} must contain at "
                    "least one model type."
                )
            normalized_candidate_model_types[family_key] = normalized_models
    dataset_id = str(manifest.get("dataset_id", Path(dataset_path).stem))
    report_root_path = (
        DEFAULT_UNKNOWN_REPORT_ROOT
        if report_root is None
        else Path(report_root)
    )
    report_directory = _report_directory(
        report_root_path, scenario, dataset_id
    )
    if print_progress:
        print(f"Timer started at {timer_started_at}.")

    instance_results = []
    all_trials = []
    auto_routes: dict[str, set[str]] = {}
    compatible_routes: dict[str, set[str]] = {}
    for instance_index, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        challenges = instance["challenges"]
        responses = instance["responses"]
        n = int(challenges.shape[1])
        response_bits = int(responses.shape[1])
        is_continuous = not np.isin(responses, (-1, 1)).all()
        candidates = _candidate_specs(
            n,
            k,
            response_bits,
            selected_families,
            attack_models,
            continuous_responses=is_continuous,
            candidate_model_types=normalized_candidate_model_types,
        )
        targets: dict[str, Any] = {}
        skipped_candidates = []
        candidate_hypotheses = list(dict.fromkeys(
            candidate["family_hypothesis"] for candidate in candidates
        ))
        for hypothesis_index, family_hypothesis in enumerate(
            candidate_hypotheses
        ):
            try:
                targets[family_hypothesis] = create_puf(
                    family_hypothesis,
                    n=n,
                    seed=seed + instance_index * 100_000 + hypothesis_index,
                    k=k,
                    response_bits=response_bits,
                    noisiness=noisiness,
                    bch_correction_capacity=bch_correction_capacity,
                    hash_responses=True,
                    **normalized_family_options.get(family_hypothesis, {}),
                )
            except ValueError as error:
                skipped_candidates.append({
                    "family_hypothesis": family_hypothesis,
                    "reason": str(error),
                })
        candidates = [
            candidate
            for candidate in candidates
            if candidate["family_hypothesis"] in targets
        ]
        if not candidates:
            reasons = "; ".join(
                f"{item['family_hypothesis']}: {item['reason']}"
                for item in skipped_candidates
            )
            raise ValueError(
                "No candidate PUF structures are valid for this dataset"
                + (f": {reasons}" if reasons else ".")
            )
        for candidate in candidates:
            family_hypothesis = candidate["family_hypothesis"]
            resolved_auto = _experiment._resolve_attack_model_for_family(
                family_hypothesis,
                "auto",
                n=n,
                k=k,
                response_bits=response_bits,
                hash_responses=candidate["hash_responses"],
            )
            auto_routes.setdefault(resolved_auto, set()).add(family_hypothesis)
            compatible_routes.setdefault(
                candidate["attack_model"], set()
            ).add(family_hypothesis)

        trial_records: list[dict[str, Any]] = []
        instance_trials: list[dict[str, Any]] = []
        for bagging_run in range(1, bagging_runs + 1):
            split_seed = seed + instance_index * bagging_runs + bagging_run
            train_indices, validation_indices, test_indices = _split_indices(
                challenges, split_seed
            )
            split_arrays = []
            for indices in (train_indices, validation_indices, test_indices):
                split_arrays.append(ChallengeResponseSet(
                    challenges[indices], responses[indices]
                ))
            split_counts = {
                "training": len(train_indices),
                "validation": len(validation_indices),
                "test": len(test_indices),
            }
            for candidate_index, candidate in enumerate(candidates):
                family_hypothesis = candidate["family_hypothesis"]
                attack_model = candidate["attack_model"]
                candidate_seed = (
                    seed
                    + instance_index * 1_000_000
                    + bagging_run * 10_000
                    + candidate_index
                )
                target = targets[family_hypothesis]
                training_device = (
                    None
                    if attack_model in _PURE_CPU_ATTACKS
                    or (attack_model == "svm" and normalized_backend == "cpu")
                    else _experiment._resolve_device(backend, device)
                )
                effective_batch_size = batch_size
                trial_started = time.perf_counter()
                if print_progress:
                    print(
                        f"\nRunning candidate for {instance_id}, "
                        f"split {bagging_run}/{bagging_runs}: "
                        f"family_hypothesis={family_hypothesis!r}, "
                        f"model_type={attack_model!r}, "
                        f"batch_size={effective_batch_size}."
                    )
                while True:
                    retry_batch_size: Optional[int] = None
                    try:
                        result = _experiment._attack_one_puf(
                            target,
                            n,
                            split_counts["training"],
                            split_counts["validation"],
                            split_counts["test"],
                            candidate_seed + 1,
                            success_threshold,
                            max_depth,
                            agents,
                            width,
                            effective_max_total_epochs,
                            effective_batch_size,
                            learning_rate,
                            training_device,
                            family_hypothesis,
                            print_progress,
                            attack_model,
                            crp_splits=(
                                split_arrays[0],
                                split_arrays[1],
                                split_arrays[2],
                            ),
                            progressive_screening=True,
                        )
                    except RuntimeError as error:
                        if (
                            getattr(training_device, "type", None) != "xpu"
                            or not _is_xpu_resource_error(error)
                            or effective_batch_size == 1
                        ):
                            raise
                        retry_batch_size = max(1, effective_batch_size // 2)
                        if print_progress:
                            print(
                                f"\nXPU resources exhausted for "
                                f"{family_hypothesis}/{attack_model}; "
                                f"retrying with batch_size={retry_batch_size}."
                            )
                    if retry_batch_size is None:
                        break
                    _clear_xpu_cache()
                    effective_batch_size = retry_batch_size
                trial_elapsed_seconds = time.perf_counter() - trial_started
                total_elapsed_seconds = time.perf_counter() - run_started
                summary = _attack_summary(result, candidate, bagging_run)
                summary["effective_batch_size"] = effective_batch_size
                summary["instance_id"] = instance_id
                summary["split_counts"] = split_counts
                summary["trial_elapsed_seconds"] = trial_elapsed_seconds
                summary["total_elapsed_seconds"] = total_elapsed_seconds
                if result.get("candidate_excluded", False):
                    summary["test_accuracy"] = None
                    summary["test_element_accuracy"] = None
                    summary["matched_test_responses"] = None
                    summary["test_response_count"] = split_counts["test"]
                    all_trials.append(summary)
                    instance_trials.append(summary)
                    if print_progress:
                        print(
                            f"{instance_id}, split {bagging_run}/{bagging_runs}: "
                            f"family_hypothesis={family_hypothesis!r}, "
                            f"model_type={attack_model!r} excluded: "
                            f"{summary['exclusion_reason']}; "
                            f"lap_elapsed={trial_elapsed_seconds:.2f}s, "
                            f"total_elapsed={total_elapsed_seconds:.2f}s"
                        )
                    continue
                trial = {
                    "summary": summary,
                    "response_rows": result["_response_rows"],
                }
                trial_records.append(trial)
                all_trials.append(summary)
                instance_trials.append(summary)
                if print_progress:
                    print(
                        f"{instance_id}, split {bagging_run}/{bagging_runs}: "
                        f"family_hypothesis={family_hypothesis!r}, "
                        f"model_type={attack_model!r}, "
                        f"architecture="
                        f"{result.get('model_architecture', attack_model)!r}, "
                        f"attack_stage={summary['attack']!r}, "
                        f"batch_size={effective_batch_size}, "
                        f"validation={summary['validation_accuracy']:.4%}, "
                        f"test={summary['test_accuracy']:.4%}, "
                        f"lap_elapsed={trial_elapsed_seconds:.2f}s, "
                        f"total_elapsed={total_elapsed_seconds:.2f}s"
                    )

        best_validation, best_test_oracle, bagging_model_summaries = (
            _rank_trials(trial_records)
        )
        instance_results.append({
            "instance_id": instance_id,
            "sample_count": len(challenges),
            "challenge_length": n,
            "response_length": response_bits,
            "candidate_count": len(candidates),
            "skipped_candidates": skipped_candidates,
            "bagging_runs": bagging_runs,
            "best_validation_selected": best_validation,
            "best_test_oracle": best_test_oracle,
            "bagging_model_summaries": bagging_model_summaries,
            "trials": instance_trials,
        })
        overall_elapsed_seconds = time.perf_counter() - run_started
        best_guessed_model, overall_model_summaries = _rank_overall_trials(
            all_trials
        )
        results = {
            "best_guessed_model": best_guessed_model,
            "overall_summary": {
                "timer_started_at": timer_started_at,
                "total_elapsed_seconds": overall_elapsed_seconds,
                "completed_instances": len(instance_results),
                "completed_trials": len(all_trials),
                "ranking_metric": "mean_test_accuracy_across_successful_bagged_trials",
                "test_accuracy_is_used_for_model_ranking": True,
            },
            "dataset_id": dataset_id,
            "instance_count": len(instances),
            "bagging_runs": bagging_runs,
            "overall_model_summaries": overall_model_summaries,
            "selection_policy": {
                "primary": "highest_validation_accuracy_then_element_accuracy",
                "test_oracle": (
                    "Highest test accuracy across candidates and bagging splits; "
                    "test-set-selected and not an unbiased generalization estimate."
                ),
                "progressive_screening": {
                    "enabled": True,
                    "max_total_epochs_per_depth": effective_max_total_epochs,
                    "reference_epoch_cap": (
                        _experiment.PROGRESSIVE_GATE_BASE_EPOCHS
                    ),
                    "gate_fractions_of_reference_epoch_cap": {
                        str(milestone): fraction
                        for milestone, (fraction, _) in
                        _experiment.PROGRESSIVE_VALIDATION_GATES.items()
                    },
                    "gate_epoch_rule": (
                        "max(reference_gate_epoch, "
                        "ceil(fraction * max_total_epochs))"
                    ),
                    "validation_accuracy_gates": {
                        "reference_epoch_30": "exclude only when accuracy is exactly 0%",
                        "reference_epoch_50": "exclude below 50%",
                        "reference_epoch_80": "exclude below 90%",
                        "reference_epoch_100": "exclude below 92%",
                    },
                    "threshold_metric": "complete_response_validation_accuracy",
                    "stop_candidate_when_success_threshold_reached": True,
                    "excluded_depths_are_restarted_at_each_candidate_depth": True,
                    "candidates_without_surviving_depths_are_not_ranked": True,
                    "gates_after_epoch_cap_are_not_reached": True,
                },
            },
            "auto_selected_attack_to_families": {
                model: sorted(families)
                for model, families in sorted(auto_routes.items())
            },
            "model_to_compatible_family_hypotheses": {
                model: sorted(families)
                for model, families in sorted(compatible_routes.items())
            },
            "instances": _serialize_instance_results(instance_results),
        }
        parameters = {
            "dataset_path": str(Path(dataset_path)),
            "dataset_id": dataset_id,
            "seed": seed,
            "bagging_runs": bagging_runs,
            "split_ratios": {"training": 0.7, "validation": 0.2, "test": 0.1},
            "k_hypothesis": k,
            "candidate_families": [
                _canonical_family(family) for family in selected_families
            ],
            "family_hypothesis": selected_families,
            "attack_models": (
                list(attack_models) if attack_models is not None else "all"
            ),
            "candidate_model_types": (
                {
                    family: sorted(model_types)
                    for family, model_types in
                    normalized_candidate_model_types.items()
                }
                if normalized_candidate_model_types is not None
                else "default_by_family"
            ),
            "success_threshold": success_threshold,
            "max_depth": max_depth,
            "agents": agents,
            "width": width,
            "epochs": epochs,
            "max_total_epochs": effective_max_total_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "backend": backend,
            "device": device,
            "noisiness_hypothesis": noisiness,
            "bch_correction_capacity": bch_correction_capacity,
            "family_options": normalized_family_options,
            "candidate_count_by_instance": {
                instance["instance_id"]: instance["candidate_count"]
                for instance in instance_results
            },
            "skipped_candidates_by_instance": {
                instance["instance_id"]: instance["skipped_candidates"]
                for instance in instance_results
                if instance["skipped_candidates"]
            },
            "instances": [
                {
                    "instance_id": instance["instance_id"],
                    "sample_count": instance["sample_count"],
                    "challenge_length": instance["challenge_length"],
                    "response_length": instance["response_length"],
                }
                for instance in instance_results
            ],
        }
        _write_unknown_report(
            report_directory,
            results,
            parameters,
            all_trials,
            instance_results,
        )
    return {
        **results,
        "report_directory": str(report_directory),
    }


__all__ = [
    "DEFAULT_CRPS_ROOT",
    "DEFAULT_UNKNOWN_REPORT_ROOT",
    "generate_crps",
    "run_attack_unknown_puf",
]
