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

- `binary_entropy`: normalized response-bit entropy, where `1.0` is ideal
- `bit_aliasing`: absolute cross-device response-bit bias, where `0.0` is ideal
- `hamming_distance_distribution`: normalized pairwise response distances
- `probability_of_misidentification`: complete-response impostor collision rate
- `uniformity`: derived as `1 - mean(abs(bias))`
- `steadiness`: an alias for the reliability result
- `diffuseness`: normalized `total_influence`

Call the individual helpers with response data:

```python
import numpy as np
from puf_sim.puf_baseline_eval import (
    binary_entropy,
    bit_aliasing,
    hamming_distance_distribution,
    probability_of_misidentification,
)

responses = np.array([
    [[-1], [1], [-1]],
    [[1], [1], [-1]],
])
print(binary_entropy(responses.reshape(-1, 1)))
print(bit_aliasing(responses))
print(hamming_distance_distribution(responses.reshape(-1, 1)))
print(probability_of_misidentification(responses))
```

## Complete report

`evaluate_exposed_metrics()` combines the official and derived metrics in one
dictionary:

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
(0.01 for SRAM, 0.0 for the other families). SRAM itself rejects noise values
below 0.01. SRAM reports also include the aggregate `bch_total_reads` and
`bch_uncorrectable_reads` counters for decoder diagnostics.

## Command line

The package is primarily a Python API. It can also be exercised from a shell
using Python's `-c` option:

```powershell
python -c "from puf_sim.puf_implementations import create_puf; from puf_sim.puf_baseline_eval import evaluate_exposed_metrics; p=create_puf('arbiter', n=64, seed=7); r=evaluate_exposed_metrics(p, samples=100, repetitions=5); print(r['uniformity'], r['reliability'].mean())"
```
