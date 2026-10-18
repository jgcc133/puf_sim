# Machine-Learning Attack Reports

`run_ml_attack_experiment()` stores reports here by scenario, using the same
date, sequence number, and family-selection naming as baseline evaluations:

```text
puf_sim/reports_ml_attacks/
└── <scenario>/
    └── yyyy mm dd Report x - All PUFs/
        ├── parameters.json
        ├── results.json
        ├── summary.csv
        └── responses.csv (or one responses_<index>_<family>.csv per family)
```

Selected-family runs use `Selected PUFs`. Each results row records the winning
attack stage, nonlinear depth, ensemble size, validation and test imitation
accuracy, and timing for each attempted stage. The report also includes
`responses.csv` for a single-family run, or one
`responses_<index>_<family>.csv` for each family in a multi-family run.

Each response CSV row represents one held-out test challenge. Columns contain
the family/instance/sample label, the PUF response values, the ML output for
that same challenge, and a `match` flag (`1` when the complete response row
matches, otherwise `0`). Discrete PUF responses are encoded as binary `0`/`1`;
Optical PUF responses remain continuous values and use the attack's
tolerance-based comparison.

In `results.json`, each instance includes `matched_test_responses` and
`test_response_count`. The count of CSV rows with `match=1` for that instance
equals `matched_test_responses`, and their ratio is `test_accuracy`. At the
family level, `matched_instances` counts instances whose test accuracy reached
the configured `success_threshold`; there is no additional cohort-level
threshold.