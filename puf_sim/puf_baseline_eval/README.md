# PUF Baseline Evaluation

`puf_baseline_eval` consolidates the PUF evaluation metrics used by `puf_sim`.
It separates calls to the official `pypuf.metrics` API from derived response
data metrics.

## Official `pypuf.metrics` calls

`evaluate_exposed_api()` calls these existing library metrics:

- `bias`: response bias per response bit
- `reliability`: repeatability under evaluation noise
- `uniqueness`: inter-device uniqueness, when a population is supplied
- `similarity`: pairwise response agreement, when two devices are supplied
- `accuracy`: agreement with a supplied `ChallengeResponseSet`
- `correlation`: response correlation with a supplied test set
- `influence`: per-challenge-bit output sensitivity
- `total_influence`: aggregate challenge-bit sensitivity
- `noise_sensitivity`: output changes under probabilistic input flips

```python
from puf_sim.puf_baseline_eval import evaluate_exposed_api
from puf_sim.puf_implementations import create_puf

puf = create_puf("xor_apuf", n=64, k=4, response_bits=2, seed=7)
metrics = evaluate_exposed_api(puf, samples=1000, repetitions=17, seed=8)

print(metrics["bias"])
print(metrics["reliability"].mean())
print(metrics["total_influence"])
```

For an XOR Arbiter PUF, omitting `k` selects `n` chains. The `response_bits`
argument chooses the number of output bits; all chains are distributed across
those output bits in near-equal groups and combined with balanced XOR reductions.
If an explicit `k` is smaller than `response_bits`, the chain count is raised
to `response_bits`.

For `uniqueness` and `similarity`, pass multiple instances:

```python
from puf_sim.puf_baseline_eval import evaluate_exposed_api
from puf_sim.puf_implementations import create_puf

population = [
    create_puf("arbiter", n=64, seed=seed)
    for seed in range(4)
]
metrics = evaluate_exposed_api(
    population[0],
    instances=population,
    samples=1000,
    seed=10,
)
print(metrics["uniqueness"])
print(metrics["similarity"])
```

For `accuracy` and `correlation`, supply a test set:

```python
from pypuf.io import ChallengeResponseSet
from puf_sim.puf_baseline_eval import evaluate_exposed_api

known_data = ChallengeResponseSet.from_simulation(puf, N=1000, seed=11)
metrics = evaluate_exposed_api(puf, test_set=known_data, seed=12)
print(metrics["accuracy"])
print(metrics["correlation"])
```

## Derived metrics

The package also provides metrics not directly exposed by `pypuf.metrics`:

- `binary_entropy`: mean normalized entropy of the 0/1 balance within each
  complete response; `1.0` means each response is evenly balanced. This helper
  is not the report's `randomness` metric.
- `bit_aliasing`: fraction of majority-voted `1` responses for each output bit
  across devices and challenges; `0.5` is balanced
- `cross_challenge_randomness`: one minus the mean absolute Pearson
  correlation across distinct challenge pairs, correlating each output bit's
  majority-voted responses across devices; `1.0` means no measured
  cross-challenge correlation
- `hamming_distance_distribution`: normalized pairwise response distances
- `probability_of_misidentification`: complete-response impostor collision rate
- `uniformity`: mean fraction of majority-voted `1` responses across
  challenge-level population responses and output bits; `0.5` is balanced
- `steadiness`: an alias for the reliability result
- `diffuseness`: `total_influence` divided by challenge length, reported to six
  decimal places

Call the individual helpers with response data:

```python
import numpy as np
from puf_sim.puf_baseline_eval import (
    binary_entropy,
    bit_aliasing,
    cross_challenge_randomness,
    hamming_distance_distribution,
    probability_of_misidentification,
)

responses = np.array([
    [[-1, 1, -1, 1], [1, 1, -1, -1], [-1, -1, -1, -1]],
    [[1, 1, 1, 1], [1, -1, 1, -1], [-1, -1, 1, 1]],
])
print(binary_entropy(responses))
print(bit_aliasing(responses))
print(cross_challenge_randomness(responses))
print(hamming_distance_distribution(responses.reshape(-1, 4)))
print(probability_of_misidentification(responses))
```

`binary_entropy` scores each response string independently from its fraction
of positive bits, then averages those scores across responses. It measures
within-response balance, not per-position bias (reported by `bit_aliasing`)
or independence between bits. A one-bit response has zero within-response
entropy by definition. The report's `uniformity` and `randomness` values are
both computed after majority-voting repetitions for each device/challenge/bit.
`uniformity` averages each challenge's fraction of positive responses across
the population and output bits into one scalar. `randomness` correlates the
response vectors for each pair of distinct challenges across devices and
scores lower absolute correlation higher. Challenge pairs with a constant
response vector are excluded; if no variable pairs remain, the score is
`0.0`.

## Complete report

`evaluate_exposed_metrics()` combines the official and derived metrics in one
dictionary:

`run_baseline_experiment()` evaluates responses through a metric-only adapter:
zero-valued bits are mapped to `-1`, and positive bits to `1`, before metric
calculations. This leaves each PUF implementation's own `eval()` output
unchanged. Optical responses are first median-thresholded, then exposed in
bipolar form. For `uniformity`, `randomness`, and `bit_aliasing`, each
device/challenge/bit is majority-voted across `repetitions` before aggregation.
Exact ties are resolved by independent 50/50 draws from an RNG seeded with the
report's `seed`, making the result reproducible for a fixed run configuration.
`response_bits` is interpreted per family. FF-APUF generates wider outputs by
sequentially evaluating independently seeded one-bit FF-APUF lanes on the same
challenge. Saved baseline parameters record the effective width for each
family as `effective_response_bits_by_family`.

```python
from puf_sim.puf_baseline_eval import evaluate_exposed_metrics

report = evaluate_exposed_metrics(
    population[0],
    instances=population,
    samples=1000,
    repetitions=17,
    input_noise=0.01,
    seed=20,
)

for name, value in report.items():
    print(name, value)
```

The complete report includes `uniqueness`, `steadiness`, `reliability`,
`uniformity`, `randomness`, `bit_aliasing`, `diffuseness`,
`probability_of_misidentification`, and all official comparison and Fourier
metrics where their required inputs are available.

`run_baseline_experiment()` saves the metric report by default. Pass
`save_responses=False` to skip exporting per-instance, per-challenge, and
per-repetition response CSV files while retaining the metric report.

Set `backend="torch_xpu"` to use Intel XPU evaluation for PUFs that provide a
backend evaluator. Response-normalization wrappers preserve that backend path,
and supported PUFs use it while exporting response CSV data as well. PUFs
without a backend evaluator and the official `pypuf.metrics` calls continue to
use their CPU evaluation path, so a baseline run is not entirely GPU-resident.

For SRAM, the baseline factory default uses `noisiness=0.01` and BCH
code-offset correction capacity one. Set these options explicitly to compare
ECC configurations:

```python
from puf_sim.puf_baseline_eval import run_baseline_experiment

results = run_baseline_experiment(
    families=["sram"],
    n=64,
    response_bits=16,
    noisiness=0.01,
    bch_correction_capacity=2,
    samples=1000,
    repetitions=17,
)
print(results["sram"]["reliability"].mean())
```

When `noisiness` is omitted, the baseline uses each family’s factory default
(0.01 for SRAM, 0.0 for the other families). SRAM accepts explicit noise
values from 0.0 to 1.0; setting it to zero disables read noise. SRAM reports
also include the aggregate `bch_total_reads` and `bch_uncorrectable_reads`
counters for decoder diagnostics.

## Command line

The package is primarily a Python API. It can also be exercised from a shell
using Python's `-c` option:

```powershell
python -c "from puf_sim.puf_implementations import create_puf; from puf_sim.puf_baseline_eval import evaluate_exposed_metrics; p=create_puf('arbiter', n=64, seed=7); r=evaluate_exposed_metrics(p, samples=100, repetitions=5); print(r['uniformity'], r['reliability'].mean())"
```
