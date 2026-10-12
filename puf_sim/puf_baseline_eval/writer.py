"""Persistence helpers for baseline evaluation reports."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_REPORT_ROOT = Path(__file__).resolve().parent.parent / "reports_baseline_eval"


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


def _summary_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "[]"
        if np.issubdtype(value.dtype, np.number):
            return float(np.mean(value))
        return str(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None:
        return ""
    return value


def save_baseline_report(
    results: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Any],
    families: Sequence[str],
    report_root: Path | str | None = None,
    responses: Mapping[str, Iterable[np.ndarray]] | None = None,
) -> Path:
    """Save one timestamped baseline report in the reports data directory."""
    root = DEFAULT_REPORT_ROOT if report_root is None else Path(report_root)
    if os.name == "nt" and root.root and not root.drive:
        project_root = DEFAULT_REPORT_ROOT.parents[1]
        root = project_root / str(root).lstrip("/\\")
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

    summary_fields = ["family"] + sorted(
        {key for report in results.values() for key in report}
    )
    with (report_directory / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for family, report in results.items():
            row = {"family": family}
            row.update({
                key: _summary_value(value)
                for key, value in report.items()
                if key != "bit_aliasing"
            })

            if "bit_aliasing" in report:
                values = np.asarray(report["bit_aliasing"]).reshape(-1)
                row["bit_aliasing"] = str(values.tolist())

            writer.writerow(row)

    if responses:
        single_family = len(responses) == 1
        repetitions = int(parameters["repetitions"])
        samples = int(parameters["samples"])
        expected_rows_per_instance = samples * repetitions
        for index, (family, chunks) in enumerate(responses.items(), start=1):
            chunk_iterator = iter(chunks)
            first_chunk = next(chunk_iterator, None)
            if first_chunk is None:
                raise ValueError(f"No response rows were generated for {family!r}.")
            first_values = np.asarray(first_chunk)
            if first_values.ndim != 2 or first_values.shape[1] < 1:
                raise ValueError(
                    "Response CSV data must be a two-dimensional matrix "
                    "with at least one response bit."
                )
            if single_family:
                filename = "responses.csv"
            else:
                safe_family = "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in family
                ).strip("_")
                filename = f"responses_{index}_{safe_family}.csv"

            with (report_directory / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["label"]
                    + [
                        f"bit_{bit_index}"
                        for bit_index in range(first_values.shape[1])
                    ]
                )
                for instance_index, chunk in enumerate(
                    chain((first_values,), chunk_iterator), start=1
                ):
                    values = np.asarray(chunk)
                    if (
                        values.ndim != 2
                        or values.shape[1] != first_values.shape[1]
                    ):
                        raise ValueError(
                            "Response CSV chunks must have matching "
                            "two-dimensional shapes."
                        )
                    if not np.issubdtype(values.dtype, np.integer) or np.any(
                        (values < 0) | (values > 1)
                    ):
                        raise ValueError("Response CSV values must be binary 0/1.")
                    if values.shape[0] != expected_rows_per_instance:
                        raise ValueError(
                            f"Expected {expected_rows_per_instance} response rows "
                            f"for instance {instance_index} of {family!r}, "
                            f"got {values.shape[0]}."
                        )
                    for row_index, row in enumerate(values):
                        sample_index, repetition_index = divmod(
                            row_index, repetitions
                        )
                        label = (
                            f"family {family}, instance {instance_index}, "
                            f"sample {sample_index + 1}, "
                            f"repetition {repetition_index + 1}"
                        )
                        writer.writerow([label, *row.tolist()])

    return report_directory


__all__ = ["DEFAULT_REPORT_ROOT", "save_baseline_report"]
