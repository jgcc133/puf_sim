# Baseline Evaluation Reports

`run_baseline_experiment()` automatically saves each baseline run under this
folder:

```text
puf_sim/reports_baseline_val/
└── yyyy mm dd Report x - All PUFs/
    ├── parameters.json
    ├── results.json
    ├── summary.csv
    └── responses.csv
```

For a selected family subset, the folder label is `Selected PUFs` instead of
`All PUFs`. The folder name includes the date, sequential report number, and
family selection. Parameter values are stored in `parameters.json`:

For a single-family run, `responses.csv` contains one row for each instance,
challenge sample, and repetition, in that order. The first `label` column
identifies the family and one-based instance, sample, and repetition indices;
remaining columns are response bits encoded as `0` or `1`. For a multi-family
run, each family has its own `responses_<index>_<family>.csv` file because
response widths can differ. Set `save_responses=False` to omit these potentially
large CSV files while still saving the metrics and report summary.

- `instances_per_family`
- `n`
- `samples`
- `repetitions`
- `input_noise`
- `seed`
- `k`
- `response_bits`
- `noisiness`

## Automatic saving

```python
from puf_sim.puf_baseline_eval import run_baseline_experiment

results = run_baseline_experiment(
    instances_per_family=100,
    n=64,
    samples=1000,
    repetitions=17,
    seed=20260921,
)
```

The function still returns the in-memory results dictionary. The same results
are saved to the newest report folder.

## Custom report location

```python
from pathlib import Path
from puf_sim.puf_baseline_eval import run_baseline_experiment

run_baseline_experiment(
    families=["arbiter", "xor_apuf"],
    instances_per_family=100,
    report_root=Path("./study_reports"),
)
```

This produces a `Selected PUFs` report under `study_reports`.

Disable all persistence when needed:

```python
run_baseline_experiment(save_report=False)
```

To save metrics without exporting every repeated response:

```python
run_baseline_experiment(save_report=True, save_responses=False)
```

## Files

- `parameters.json` records the run configuration and selected families.
- `results.json` stores the complete metric arrays and values.
- `summary.csv` stores one row per family, reducing array-valued metrics to
  their mean for convenient spreadsheet analysis.
- `responses.csv` (or one `responses_<index>_<family>.csv` per family), when
  `save_responses=True`, stores binary output response rows. Its number of data
  rows is `instances_per_family * samples * repetitions`.
