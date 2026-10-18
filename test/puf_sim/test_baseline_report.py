import csv
import io
import json

import numpy as np
import pytest

from puf_sim.puf_baseline_eval import run_baseline_experiment
from puf_sim.puf_baseline_eval.writer import save_baseline_report
from puf_sim.puf_backend import Backend, xpu_available
from puf_sim.puf_baseline_eval.puf_test import _collect_population_responses


def test_response_csv_writer_preserves_csv_escaping_and_bytes(tmp_path):
    family = 'example, "quoted" family'
    responses = [np.array([[1, 0], [0, 1]], dtype=np.uint8)]
    report_directory = save_baseline_report(
        {},
        {"samples": 1, "repetitions": 2},
        [family],
        report_root=tmp_path,
        responses={family: responses},
    )

    expected = io.StringIO(newline="")
    writer = csv.writer(expected)
    writer.writerow(["label", "bit_0", "bit_1"])
    writer.writerow([f'family {family}, instance 1, sample 1, repetition 1', 1, 0])
    writer.writerow([f'family {family}, instance 1, sample 1, repetition 2', 0, 1])

    assert (report_directory / "responses.csv").read_bytes() == (
        expected.getvalue().encode("utf-8")
    )


def test_backend_response_collection_batches_repetitions():
    class RepeatedResponsePUF:
        challenge_length = 1
        response_length = 1

        def __init__(self):
            self.backend_calls = 0

        def eval_backend(self, challenges, backend):
            self.backend_calls += 1
            return np.repeat([-1, 1], len(challenges) // 2)

    puf = RepeatedResponsePUF()
    responses = next(
        _collect_population_responses(
            [puf],
            samples=3,
            repetitions=2,
            seed=0,
            backend=Backend("torch_xpu", np),
        )
    )

    assert puf.backend_calls == 1
    np.testing.assert_array_equal(responses[:, 0], [0, 1, 0, 1, 0, 1])


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


def test_ff_apuf_baseline_uses_requested_response_bits(tmp_path):
    results = run_baseline_experiment(
        families=["ff_apuf"],
        instances_per_family=2,
        n=8,
        samples=4,
        repetitions=3,
        response_bits=10,
        save_report=True,
        report_root=tmp_path,
        workers=1,
    )

    report_directory = next(tmp_path.iterdir())
    with (report_directory / "parameters.json").open(
        encoding="utf-8"
    ) as handle:
        parameters = json.load(handle)

    assert parameters["response_bits"] == 10
    assert parameters["effective_response_bits_by_family"]["ff_apuf"] == 10
    assert results["ff_apuf"]["bit_aliasing"].shape == (10,)


def test_baseline_report_can_omit_response_rows(tmp_path):
    run_baseline_experiment(
        families=["xor_apuf"],
        instances_per_family=2,
        n=4,
        samples=3,
        repetitions=2,
        response_bits=3,
        save_report=True,
        save_responses=False,
        report_root=tmp_path,
        workers=1,
    )

    report_directory = next(tmp_path.iterdir())
    assert (report_directory / "results.json").is_file()
    assert (report_directory / "summary.csv").is_file()
    assert not list(report_directory.glob("responses*.csv"))


@pytest.mark.skipif(not xpu_available(), reason="Intel XPU backend is unavailable")
def test_silicon_photonic_xpu_report_saves_batched_responses(tmp_path):
    results = run_baseline_experiment(
        families=["silicon_photonic"],
        instances_per_family=2,
        n=8,
        samples=4,
        repetitions=3,
        response_bits=2,
        noisiness=0.01,
        report_root=tmp_path,
        save_report=True,
        workers=1,
        backend="torch_xpu",
        device=0,
        batch_size=1,
    )

    report_directory = next(tmp_path.iterdir())
    with (report_directory / "responses.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.reader(handle))

    assert results["silicon_photonic"]["bit_aliasing"].shape == (2,)
    assert len(rows) == 1 + 2 * 4 * 3


@pytest.mark.skipif(not xpu_available(), reason="Intel XPU backend is unavailable")
def test_xor_apuf_xpu_report_saves_backend_evaluated_responses(tmp_path):
    run_baseline_experiment(
        families=["xor_apuf"],
        instances_per_family=2,
        n=8,
        k=2,
        samples=4,
        repetitions=3,
        save_report=True,
        report_root=tmp_path,
        workers=1,
        backend="torch_xpu",
        device=0,
    )

    report_directory = next(tmp_path.iterdir())
    with (report_directory / "responses.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 1 + 2 * 4 * 3
    assert set(row[1] for row in rows[1:]).issubset({"0", "1"})
