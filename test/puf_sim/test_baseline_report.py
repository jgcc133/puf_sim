import csv

import numpy as np

from puf_sim.puf_baseline_eval import run_baseline_experiment


def test_baseline_report_saves_repeated_binary_response_rows(tmp_path):
    run_baseline_experiment(
        families=["xor_apuf"],
        instances_per_family=2,
        n=4,
        samples=3,
        repetitions=2,
        response_bits=3,
        save_report=True,
        report_root=tmp_path,
        workers=1,
    )

    report_directory = next(tmp_path.iterdir())
    with (report_directory / "responses.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["label", *[f"bit_{index}" for index in range(3)]]
    assert rows[1][0] == (
        "family xor_apuf, instance 1, sample 1, repetition 1"
    )
    assert rows[2][0] == (
        "family xor_apuf, instance 1, sample 1, repetition 2"
    )
    assert rows[3][0] == (
        "family xor_apuf, instance 1, sample 2, repetition 1"
    )
    assert rows[7][0] == (
        "family xor_apuf, instance 2, sample 1, repetition 1"
    )
    responses = np.asarray([row[1:] for row in rows[1:]], dtype=np.uint8)
    assert responses.shape == (2 * 3 * 2, 3)
    assert set(np.unique(responses)).issubset({0, 1})
