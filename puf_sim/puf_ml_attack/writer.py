"""Persistence helpers for machine-learning attack reports."""
from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DEFAULT_REPORT_ROOT = Path(__file__).resolve().parent.parent / "reports_ml_attacks"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _scenario_path(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._ -]+", "_", value.strip()).strip(" .")
    if not label:
        raise ValueError("scenario must contain at least one valid path character.")
    return label


def save_ml_attack_report(
    results: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Any],
    families: Sequence[str],
    scenario: str = "general",
    report_root: str | Path | None = None,
    responses: Mapping[
        str, Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] | None = None,
) -> Path:
    """Save a timestamped attack report and optional test-response comparisons."""
    root = DEFAULT_REPORT_ROOT if report_root is None else Path(report_root)
    if os.name == "nt" and root.root and not root.drive:
        project_root = DEFAULT_REPORT_ROOT.parents[1]
        root = project_root / str(root).lstrip("/\\")
    root = root / _scenario_path(scenario)
    root.mkdir(parents=True, exist_ok=True)
    report_number = 1 + sum(path.is_dir() for path in root.iterdir())
    date_label = datetime.now().strftime("%Y %m %d")
    default_families = parameters.get("default_families", ())
    family_label = (
        "All PUFs" if set(families) == set(default_families) else "Selected PUFs"
    )
    report_directory = root / f"{date_label} Report {report_number} - {family_label}"
    report_directory.mkdir()

    metadata = dict(parameters)
    metadata.update({
        "families": list(families),
        "report_number": report_number,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_directory": str(report_directory),
    })
    with (report_directory / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(results), handle, indent=2)
    with (report_directory / "parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(metadata), handle, indent=2)
    fields = ["family"] + sorted({key for report in results.values() for key in report})
    with (report_directory / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family, report in results.items():
            row = {"family": family}
            row.update({
                key: json.dumps(_json_value(value)) if isinstance(value, (dict, list)) else value
                for key, value in report.items()
            })
            writer.writerow(row)

    if responses:
        single_family = len(responses) == 1
        expected_instances = int(parameters["puf_instances"])
        expected_samples = int(parameters["test_samples"])
        for index, (family, chunks) in enumerate(responses.items(), start=1):
            chunk_iterator = iter(chunks)
            first_chunk = next(chunk_iterator, None)
            if first_chunk is None:
                raise ValueError(f"No test response rows were generated for {family!r}.")
            first_puf, first_attack, first_matches = (
                np.asarray(values) for values in first_chunk
            )
            if (
                first_puf.ndim != 2
                or first_puf.shape[1] < 1
                or first_attack.shape != first_puf.shape
                or first_matches.shape != (first_puf.shape[0],)
            ):
                raise ValueError(
                    "Test response data must contain matching two-dimensional "
                    "PUF and attack arrays plus one match value per sample."
                )
            if single_family:
                filename = "responses.csv"
            else:
                safe_family = "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in family
                ).strip("_")
                filename = f"responses_{index}_{safe_family}.csv"

            response_columns = (
                [
                    f"puf_output_{bit_index}"
                    for bit_index in range(first_puf.shape[1])
                ]
                if family.strip().lower() == "optical"
                else [
                    f"bit_{bit_index}"
                    for bit_index in range(first_puf.shape[1])
                ]
            )
            attack_columns = [
                f"ml_{column}" for column in response_columns
            ]
            with (report_directory / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["label", *response_columns, *attack_columns, "match"])
                instance_chunks = enumerate(chain((first_chunk,), chunk_iterator), start=1)
                for instance_index, chunk in instance_chunks:
                    puf_values, attack_values, row_matches = (
                        np.asarray(values) for values in chunk
                    )
                    if (
                        puf_values.ndim != 2
                        or puf_values.shape[1] != first_puf.shape[1]
                        or attack_values.shape != puf_values.shape
                        or row_matches.shape != (puf_values.shape[0],)
                    ):
                        raise ValueError(
                            "Response chunks must have matching PUF/attack "
                            "dimensions and one match value per sample."
                        )
                    if puf_values.shape[0] != expected_samples:
                        raise ValueError(
                            f"Expected {expected_samples} test response rows for "
                            f"instance {instance_index} of {family!r}, got "
                            f"{puf_values.shape[0]}."
                        )
                    if (
                        not np.issubdtype(puf_values.dtype, np.number)
                        or not np.issubdtype(attack_values.dtype, np.number)
                        or not np.all(np.isfinite(puf_values))
                        or not np.all(np.isfinite(attack_values))
                    ):
                        raise ValueError("Response CSV values must be finite numbers.")
                    for sample_index, (puf_row, attack_row, matched) in enumerate(
                        zip(puf_values, attack_values, row_matches), start=1
                    ):
                        label = (
                            f"family {family}, instance {instance_index}, "
                            f"sample {sample_index}, repetition 1"
                        )
                        writer.writerow([
                            label,
                            *puf_row.tolist(),
                            *attack_row.tolist(),
                            int(matched),
                        ])
            if instance_index != expected_instances:
                raise ValueError(
                    f"Expected {expected_instances} response chunks for {family!r}, "
                    f"got {instance_index}."
                )
    return report_directory


__all__ = ["DEFAULT_REPORT_ROOT", "save_ml_attack_report"]