# PUF Machine-Learning Attacks

`run_ml_attack_experiment()` trains attacks against independently seeded PUF
instances. Each instance gets its own training, validation, and held-out test
challenge-response sets and its own attack run. Predictions are made from the
test challenges; the corresponding PUF test responses are used only to score
those predictions.

When `families=None`, the experiment attacks `optical`, `arbiter`, `ff_apuf`,
`xor_apuf`, `interpose`, `permutation`, `ring_oscillator`, `sram`,
`memristive`, and `silicon_photonic`.

## Model catalog and routing

The experiment supports the following top-level `attack_model` values:

- `"auto"`: family-aware default routing.
- `"linear_logits"`: a raw-challenge, no-hidden-layer PyTorch baseline trained
  with binary cross-entropy and Adam (optical PUFs remain on their regression
  route).
- `"neural"`: family-routed neural path (non-optical families).
- `"svm"`: SVM route for supported non-optical families.
- `"deep_neural"`: structure-aware PyTorch DNN for `ff_apuf`, `xor_apuf`,
  `interpose`, and `permutation`.

With `attack_model="auto"`, routing is:

| Family | Auto-selected attack | Other practical options |
| --- | --- | --- |
| `arbiter` | ATT + scikit-learn logistic regression | `neural`, `svm` |
| `ff_apuf` | Structure-aware deep neural FF model | `deep_neural`, `neural`, `svm` |
| `xor_apuf` | Chain-aware route: logistic (1 chain/bit), rank-two (2 chains/bit), parity neural (larger) | `deep_neural`, `neural`, `svm` |
| `interpose` | Structure-aware linear/logistic model of the up/down stages | `deep_neural`, `neural`, `svm`, `staged_split` |
| `permutation` | Structure-aware permutation parity attack | `deep_neural`, `neural`, `svm` |
| `optical` | Structure-aware rank-two quadratic regression | (always optical-specific route) |
| `ring_oscillator` | Logistic regression on raw challenge features | `neural`, `svm` |
| `sram` | Hash brute-force (`hash_responses=True`) or direct logistic (`False`) | `neural`, `svm` |
| `memristive` | Linear classifier route | `neural`, `svm` |
| `silicon_photonic` | Shared nonlinear classifier route | `neural`, `svm` |

Notes:

- For `xor_apuf`, route details depend on effective chains per response bit.
- For `deep_neural` FF training, per-bit threshold calibration is enabled to
  reduce row-level single-bit mismatches. All supported deep-neural families
  reduce the learning rate by half after eight epochs without a meaningful
  validation-EMA improvement, down to a floor of `1e-4`. The permutation
  parity attack uses the same adaptive learning-rate policy. Their stage
  reports record each epoch's validation scores, EMA, learning rates, and LR
  events.

## Training and success criteria

For each instance, the experiment selects an attack from the supplied PUF
family. Arbiter, feed-forward Arbiter, XOR Arbiter, Interpose, and optical PUFs
use their family-specific attacks; ring-oscillator and memristive PUFs use a
linear classifier; the remaining families use the shared nonlinear classifier.
With the default `attack_model="auto"`, family routing is:

- Arbiter PUF: scikit-learn logistic regression on the Arbiter Threshold
  Transform (ATT) challenge features.
- Feed-forward Arbiter PUF: the structure-aware deep neural network. It models
  segment delays and the feed-forward virtual-bit insertion and delay
  recurrence, applies a weakest-bit margin auxiliary loss during training, and
  calibrates per-bit validation thresholds to reduce row-level single-bit
  mismatches. The deep-neural trainer also applies adaptive learning-rate
  reduction after validation EMA plateaus.
- XOR Arbiter PUF: one chain per response bit uses a PyTorch logistic model on
  ATT features. Two chains per bit use a degree-two ATT-feature model, which
  represents the XOR as the sign of the product of the two Arbiter delays.
  Larger groups use the parity-aware neural model.
- Interpose PUF: a structure-aware, zero-hidden-layer model. It fits linear
  Arbiter-delay/logistic components for the upstream bit and each downstream
  chain, then composes downstream chains by parity. During training it evaluates
  both possible upstream bits and anneals their mixture toward a hard upstream
  decision. Inference uses the hard sign and a single downstream evaluation.
  Downstream XOR groups use a parity-specific straight-through gradient.
- Optical PUF: structure-aware rank-two quadratic intensity regression, fitting
  the response law `|x @ t|**2`.
- Permutation PUF: an independent linear Arbiter score for every fixed,
  permuted challenge, with an ATT computed per chain. Two-chain response
  groups use a deterministic spectral initialization, alternating
  chainwise-BCE warm-up, and a direct product score trained with
  binary cross-entropy; larger groups use sign-aware straight-through parity
  training. Its learning rate is reduced by half after eight epochs without a
  meaningful validation-EMA improvement, with a four-epoch cooldown and a
  `1e-4` floor.
- SRAM PUF: hashed responses use a random fingerprint search, testing up to 100
  candidates against the training CRPs by hashing challenge/fingerprint XOR
  rows and comparing their digests. With
  `hash_responses=False`, auto routing uses per-bit logistic regression on the
  corresponding challenge bit, matching the direct XOR response law.
- Other families: their existing attacks.

For non-optical families, `attack_model="neural"` selects the existing
family-routed attacks; two-chain XOR groups use the factorized rank-two model, while
larger groups use the parity-aware model. `attack_model="svm"` selects SVMs.
The XOR SVM is parity-aware: it applies the Arbiter threshold transform,
learns chain margins, combines chain scores by parity, and trains with a
hinge-loss objective. Other SVMs use raw challenge and RBF random Fourier
features with one estimator per response bit. The XOR SVM uses PyTorch;
`"auto"` backend selects XPU when available, while CPU SVMs use scikit-learn.

For `ff_apuf`, `xor_apuf`, `interpose`, and `permutation`, the explicit
`attack_model="deep_neural"` option selects a PyTorch DNN with family-specific
structure. It learns feed-forward segment delays and inserts sign-estimated
virtual bits into later segments; XOR and permutation models compose learned
chain scores by parity; the Interpose model evaluates downstream predictions
under both upstream-bit hypotheses during training and uses a hard upstream
sign at inference. Each response bit uses its own group of downstream chains,
matching the Interpose simulator. The requested bits are produced in one
evaluation; repeated calls with the same deterministic challenge and PUF would
repeat the same response. This architecture uses the PUF simulator's
feed-forward delay recurrence. The cited
[Fortified-Edge 2.0 paper](https://doi.org/10.3390/fi17070272) describes
feature-abstraction for PUF-based authentication rather than the feed-forward
Arbiter PUF delay recurrence.

For Interpose PUFs, `attack_model="staged_split"` selects the same linear
structure-aware model but alternates downstream-only and upstream-only
optimization steps. This separates the two parameter groups while retaining
the final-response loss; the upstream bit remains latent and is handled by the
two-hypothesis mixture. Use
`families=["interpose"]` with this setting. The default `auto` route instead
jointly trains the structure-aware linear/logistic model. `attack_model="neural"`
selects the existing hidden-layer Interpose model, while `deep_neural` selects
the structure-aware Interpose network with the requested hidden layers.

The scikit-learn logistic-regression attack for Arbiter PUFs fits one model per
response bit and is a depth-zero, non-epoch-based estimator. Structure-aware
Interpose logistic models train their linear components over epochs at depth
zero; hidden-layer Interpose neural models use `max_depth` progressively.
`agents` is the number of independent restarts. Optical regression is also
depth zero. At each trainable depth, the best ensemble or single-agent model is
retained according to validation accuracy. Multi-bit Interpose selects
checkpoints by per-response element accuracy, breaking ties with complete-row
accuracy; other families prioritize complete-row accuracy. The
`success_threshold` remains based on complete-response-row accuracy. A model is
also stopped after 10 epochs if its per-response validation accuracy never
exceeds 50%. This function still chooses the attack from the provided family;
trying every family model against an unknown target is a separate
orchestration path.

Optical regression is a depth-0 attack and does not consume neural training
epochs. Arbiter logistic regression is also depth zero. Ring-oscillator and
memristive attacks use depth-0 linear models; hidden-layer neural attacks
train progressively from depth 1.

`results.json` records `activation_types` for the selected attack model and
for each trained stage. The profile distinguishes hidden-layer activations,
family-specific internal activations/operations, the output layer, and
prediction-time thresholding where applicable.

The threshold is applied separately to each instance's held-out test set. For
discrete PUFs, a test response row matches only when all response bits are
correct. An instance is counted in `matched_instances` when its test accuracy
meets `success_threshold` (95% by default). The returned family summary retains
the number/fraction of instances meeting that criterion and also reports
`mean_test_accuracy`, the arithmetic mean of their held-out test accuracies.
Console summaries show mean accuracy and the required per-instance threshold,
not a binary match-rate score.

Optical PUF responses are continuous intensities. Their row accuracy uses the
same tolerance-based comparison as the optical attack scorer rather than
binary sign matching.

## Python usage

The public function signature is:

```python
from typing import Any, Dict, Optional, Sequence

results = run_ml_attack_experiment(
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
    scenario: str = "generic",
    report_root: Optional[str] = None,
    save_report: bool = True,
    puf_instances: int = 1000,
    print_progress: bool = False,
    attack_model: str = "auto",
    hash_responses: bool = True,
)
    ...
```

`response_bits` controls the output width for supported families, including
`arbiter` and `xor_apuf`. A multi-bit Arbiter PUF has one independent Arbiter
chain per output bit; logistic regression fits one ATT-feature classifier per
bit. A multi-bit XOR Arbiter PUF divides its chains into output groups and XORs
the chain responses within each group. Auto routing uses one ATT-feature
logistic classifier per bit for single-chain groups, pairwise quadratic ATT
features for two-chain groups when explicitly selecting the quadratic model;
auto uses a factorized pair of Arbiter scores for two-chain groups, and the
parity-aware neural model for larger
groups. The test scorer compares complete response rows, so every output bit
in a row must match for that row to count as correct. Training selects the best
checkpoint by complete-response validation accuracy, breaking ties with
per-response element accuracy, except for multi-bit Interpose, which prioritizes
per-response element accuracy and breaks ties with complete-response accuracy.
The success threshold always uses complete-response-row accuracy.
With `attack_model="auto"`, permutation PUFs use their structure-aware
per-chain ATT/parity model, which applies each chain's fixed challenge
permutation before learning its Arbiter delay weights. The model supports
multiple chains per response bit and keeps raw challenges in memory, computing
the per-chain ATT features batch by batch during training.
Explicit `attack_model="svm"` applies SVM routing to selected non-optical
families; optical always retains its structure-aware regression model. All
models fit on training CRPs, use validation CRPs for selection/stopping, and
reserve test CRPs for final scoring.
`hash_responses` applies only to SRAM and defaults to `True`. Set it to `False`
to return challenge/fingerprint XOR bits directly; then `response_bits` cannot
exceed `n`. In auto mode, this selects the structure-aware SRAM logistic attack.
With hashing enabled, auto mode tests at most 100 randomly generated candidate
fingerprints against the training CRPs. For large `n`, this small search is
unlikely to recover the true fingerprint; hashed unseen-CRP accuracy should
remain near chance.

## Parameters

| Parameter | Default | Description |
| --- | ---: | --- |
| `families` | Default family list | PUF family names to attack. |
| `n` | `64` | Challenge width. |
| `training_samples` | `10000` | Training challenge-response rows per instance. |
| `validation_samples` | `2000` | Validation rows used for model selection and stopping. |
| `test_samples` | `2000` | Held-out rows used for final scoring and response export. |
| `seed` | `20260926` | Base seed; each family-instance run receives a derived seed. |
| `k` | `2` | Family-specific chain count for PUFs that use `k`. |
| `response_bits` | `1` | Number of PUF output bits where supported. |
| `noisiness` | `0.0` | PUF simulation noisiness. |
| `hash_responses` | `True` | For SRAM, apply SHA-256 to the challenge/fingerprint XOR; `False` exposes the direct XOR bits. |
| `success_threshold` | `0.95` | Per-instance validation stopping and test-pass threshold. |
| `max_depth` | `3` | Maximum neural/recurrent depth. |
| `agents` | `3` | Number of training restarts at each depth. |
| `width` | `None` | Hidden-layer width; `None` uses `max(16, 2 * n)`. |
| `epochs` | `100` | Maximum epochs at each depth for each instance. |
| `batch_size` | `256` | Training minibatch size and FC-BiLSTM inference batch size. |
| `learning_rate` | `0.001` | Optimizer learning rate. |
| `backend` | `"auto"` | `"auto"`, `"cpu"`, or `"torch_xpu"`. |
| `device` | `None` | Optional Intel XPU index; only used with `"torch_xpu"`. |
| `scenario` | `"general"` | Report subdirectory label. |
| `report_root` | Default report directory | Optional custom report root. |
| `save_report` | `True` | Save JSON, summary, and test-response CSV files. |
| `puf_instances` | `1000` | Number of independent PUF instances per family. |
| `print_progress` | `False` | Print per-stage and per-epoch progress, per-instance test accuracy, and the family test summary. |
| `attack_model` | `"auto"` | `"auto"` uses family-specific attacks, including the structure-aware Interpose linear/logistic model and SRAM fingerprint search or direct logistic model according to `hash_responses`. `"neural"` or `"svm"` selects that route for non-optical families. `"deep_neural"` selects the structure-aware DNN for FF Arbiter, XOR Arbiter, Interpose, and Permutation PUFs. `"staged_split"` runs alternating upstream/downstream logistic fitting and requires `families=["interpose"]`. |

`epochs` is a separate limit at every model depth and for every instance; it is
not a single total-epoch budget. The function validates positive dimensions,
sample counts, epoch and batch limits, and instance count. `success_threshold`
must be greater than `0.5` and no greater than `1.0`.

## Results

The returned mapping contains one report per family. Each family report
includes:

- `matched_instances`: instances whose test accuracy met `success_threshold`.
- `instance_match_rate`: `matched_instances / puf_instances`.
- `mean_test_accuracy`: arithmetic mean of the instances' held-out test accuracies.
- `mean_test_element_accuracy`: arithmetic mean of held-out bit/pixel accuracy,
  scored across individual response values.
- `success_threshold`: the per-instance accuracy threshold used.
- `epochs_trained` and `fit_elapsed_seconds`: totals across the instances.
- `instances`: per-instance attack metrics, including validation/test
  complete-response and element accuracy, training effort, and
  `matched_test_responses` /
  `test_response_count`.

Per-instance `threshold_reached` indicates that validation reached the
threshold and training stopped early. Per-instance `success` indicates that
held-out test accuracy met the threshold. These are different measurements:
validation controls training, while test accuracy determines whether an
instance counts in the family summary.

## Saved reports

Reports are saved below
`puf_sim/reports_ml_attacks/<scenario>/` by default. Each dated report folder
contains `parameters.json`, `results.json`, `summary.csv`, and response
comparisons. Use `save_report=False` to return results without writing files.

A single-family run writes `responses.csv`; a multi-family run writes one
`responses_<index>_<family>.csv` per family. Each row represents one held-out
test challenge and includes:

1. A label identifying the family, instance, sample, and repetition.
2. The PUF response values.
3. The ML model's response values for the same test challenge.
4. `match`: `1` if the full response row matches, otherwise `0`.

Discrete PUF outputs use the baseline CSV's binary `0`/`1` encoding. Optical
output columns retain continuous values. Per-instance `matched_test_responses`
equals the number of `match=1` rows, and `test_response_count` is the number of
test rows; their ratio is the reported `test_accuracy`.
`test_element_accuracy` measures the fraction of individual response bits
(pixels for optical PUFs) that match, independently of whether the rest of the
response row matches.

## Unknown-family CRP search

`generate_crps()` creates one compressed `.npz` file per PUF instance under
`puf_ml_attack/data_crps/` by default. Instance IDs are stable within the
dataset (`instance_0001`, `instance_0002`, ...), and each file contains all
CRPs for that one PUF. The manifest records dimensions and sample counts but
does not disclose the generating family. Set exactly one of
`samples_per_instance` or `total_samples`; the latter is distributed as evenly
as possible across instances.

Run this from the repository root with the project Python environment:

```python
from puf_sim.puf_ml_attack import generate_crps, run_attack_unknown_puf

dataset_directory = generate_crps(
    family_instance_counts={
        "xor_apuf": 1,
        "arbiter": 1,
    },
    n=64,
    samples_per_instance=2_000,  # 2,000 CRPs per PUF instance
    # Alternatively, use total_samples=4_000 instead of samples_per_instance.
    k=2,
    response_bits=1,
    noisiness=0.0,
    seed=23,
)
print(f"CRPs saved to: {dataset_directory}")

report = run_attack_unknown_puf(
    dataset_path=dataset_directory,
    bagging_runs=2,  # two different 70/20/10 splits of each instance's CRPs
    seed=24,
    k=2,
    max_depth=2,
    agents=1,
    epochs=10,
    batch_size=256,
    backend="cpu",
    print_progress=True,
)

for instance in report["instances"]:
    best = instance["best_validation_selected"]
    print(
        instance["instance_id"],
        best["family_hypothesis"],
        best["model_type"],
        f"mean validation accuracy: "
        f"{best['mean_validation_accuracy']:.2%}",
        f"selected run test accuracy: {best['test_accuracy']:.2%}",
    )
print(f"Report saved to: {report['report_directory']}")
```

On a machine with a supported Intel XPU and PyTorch XPU support, use
`backend="torch_xpu", device=0` instead of `backend="cpu"`.

By default, the runner tests all supported family hypotheses and compatible
attack models; neither `candidate_families` nor `attack_models` needs to be
passed. Both are optional filters for smaller searches. The full search can
take substantially longer; reduce `samples_per_instance`, `bagging_runs`,
`max_depth`, or `epochs` for an initial smoke run.

`run_attack_unknown_puf()` treats each CRP file as an opaque instance and
compares supported model/family hypotheses using repeated, seeded 70/20/10
splits. Candidate models share each split; repeated runs use different
allocations. Duplicate challenges are kept within one partition to prevent
challenge leakage, so split sizes can differ slightly from the requested
ratios when the dataset contains repeated challenges. Candidate-family names
are hypotheses, not labels read from the data.

The report includes the family-to-auto-route reverse map, compatible family
hypotheses, per-candidate scores, and both a validation-selected response file
and `responses.csv` for the highest-test-scoring (test-oracle) candidate. Since
the latter is chosen using test responses, its score is not an unbiased
generalization estimate; use the validation-selected result for that purpose.
`attack_models` can restrict candidates, and `candidate_families` can restrict
the structural hypotheses when a smaller search is desired. For structural
assumptions such as feed-forward loop locations or the Interpose bit position,
pass per-family values through `family_options`.

## Command line

The entrypoint is [__main__.py](</C:/Users/jgcc1/OneDrive/Documents/Local Porting/Code for Masters/1. Capstone/code/pypuf/puf_sim/puf_ml_attack/__main__.py>):

- `--list-families`: print supported default families and exit.
- `--quick-run`: evaluator-friendly run profile (`5000/1000/1000` samples,
  `puf_instances=1`, `max_depth=1`, `agents=1`).
- `--show-config`: print resolved configuration before execution.

Quick FF-APUF sanity run:

```powershell
& ".\puf_env\Scripts\python.exe" -m puf_sim.puf_ml_attack `
    --families ff_apuf `
    --quick-run `
    --print-progress
```

Reproducible FF-APUF deep-neural run on XPU:

```powershell
& ".\puf_env\Scripts\python.exe" -m puf_sim.puf_ml_attack `
    --families ff_apuf `
    --attack-model auto `
    --backend torch_xpu `
    --device 0 `
    --n 64 `
    --k 2 `
    --response-bits 10 `
    --training-samples 100000 `
    --validation-samples 30000 `
    --test-samples 20000 `
    --puf-instances 1 `
    --max-depth 1 `
    --agents 1 `
    --width 128 `
    --epochs 200 `
    --batch-size 256 `
    --learning-rate 4e-4 `
    --success-threshold 0.95 `
    --scenario ff_apuf `
    --print-progress `
    --show-config
```

SRAM auto routing without hashing:

```powershell
& ".\puf_env\Scripts\python.exe" -m puf_sim.puf_ml_attack `
    --families sram `
    --n 64 `
    --response-bits 64 `
    --no-sram-hash `
    --attack-model auto `
    --print-progress
```
