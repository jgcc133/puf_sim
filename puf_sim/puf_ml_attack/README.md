# PUF Machine-Learning Attacks

`run_ml_attack_experiment()` trains attacks against independently seeded PUF
instances. Each instance gets its own training, validation, and held-out test
challenge-response sets and its own attack run. Predictions are made from the
test challenges; the corresponding PUF test responses are used only to score
those predictions.

When `families=None`, the experiment attacks `optical`, `arbiter`, `ff_apuf`,
`xor_apuf`, `interpose`, `permutation`, `ring_oscillator`, `sram`,
`memristive`, and `silicon_photonic`.

## Training and success criteria

For each instance, the experiment selects an attack from the supplied PUF
family. Arbiter, feed-forward Arbiter, XOR Arbiter, Interpose, and optical PUFs
use their family-specific attacks; ring-oscillator and memristive PUFs use a
linear classifier; the remaining families use the shared nonlinear classifier.
With the default `attack_model="auto"`, family routing is:

- Arbiter PUF: scikit-learn logistic regression on the Arbiter Threshold
  Transform (ATT) challenge features.
- Feed-forward Arbiter PUF: a feed-forward-aware FC-BiLSTM. Separate
  bidirectional LSTMs model the upstream and downstream delay segments, with
  an explicit virtual bit, loop insertion, gate, and delay combination.
- XOR Arbiter PUF: one chain per response bit uses a PyTorch logistic model on
  ATT features. Two chains per bit use a degree-two ATT-feature model, which
  represents the XOR as the sign of the product of the two Arbiter delays.
  Larger groups use the parity-aware neural model.
- Interpose PUF: a direct fully-connected/BiLSTM response model. Ordered raw
  challenge bits are projected by a shared fully connected layer, processed as
  a sequence by a bidirectional LSTM, then passed to a fully connected
  response head.
- Optical PUF: structure-aware rank-two quadratic intensity regression, fitting
  the response law `|x @ t|**2`.
- Permutation PUF: an independent linear Arbiter score for every fixed,
  permuted challenge, with an ATT computed per chain. Two-chain response
  groups use a deterministic spectral initialization, alternating
  chainwise-BCE warm-up, and a direct product score trained with
  binary cross-entropy; larger groups use sign-aware straight-through parity
  training.
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

The logistic-regression attack fits one model per Arbiter response bit and is a
depth-zero, non-epoch-based estimator. FC-BiLSTM training uses `max_depth` as
the number of recurrent layers at each progressive stage and `agents` as the
number of independent restarts. Optical regression is also depth zero. At each
trainable depth, the best ensemble or single-agent model is retained according
to validation accuracy. Training stops for that instance when validation accuracy reaches
`success_threshold`, or when it reaches the per-depth `epochs` limit. A model
is also stopped after 10 epochs if its per-response validation accuracy never
exceeds 50%. This screening score is computed across individual response
values, while model selection and the success threshold continue to use
complete-response-row accuracy. This distinction avoids prematurely rejecting
multi-bit attacks whose individual bits are learned well but whose entire
response rows do not yet match. This function still chooses the attack from the
provided family; trying every family model against an unknown target is a
separate orchestration path.

Optical regression is a depth-0 attack and does not consume neural training
epochs. Arbiter logistic regression is also depth zero. Ring-oscillator and
memristive attacks use depth-0 linear models; FC-BiLSTM and neural attacks
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
per-response element accuracy.
With `attack_model="auto"`, permutation PUFs use their structure-aware
per-chain ATT/parity model, which applies each chain's fixed challenge
permutation before learning its Arbiter delay weights. The model supports
multiple chains per response bit and keeps raw challenges in memory, computing
the per-chain ATT features batch by batch during training.
`ff_apuf` currently produces one response bit.

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
| `attack_model` | `"auto"` | `"auto"` uses family-specific attacks, including the SRAM fingerprint search or direct logistic model according to `hash_responses`. `"neural"` or `"svm"` selects that route for non-optical families. |

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

## Command line

Run a multi-bit Arbiter attack from PowerShell:

```powershell
& ".\puf_env\Scripts\python.exe" -m puf_sim.puf_ml_attack `
    --families arbiter `
    --n 64 `
    --response-bits 3 `
    --training-samples 5000 `
    --validation-samples 1000 `
    --test-samples 1000 `
    --puf-instances 10 `
    --success-threshold 0.95 `
    --backend cpu `
    --scenario iot
```

By default, Arbiter uses scikit-learn logistic regression,
single-chain-per-bit XOR-APUF uses PyTorch logistic regression, two-chain
XOR-APUF uses the factorized rank-two model, larger XOR groups use the parity-aware
neural model, FF-APUF uses its feed-forward-aware FC-BiLSTM, and Interpose uses
the standard FC-BiLSTM.
Add `--attack-model neural` to select the existing family-routed neural
architectures, or `--attack-model svm` for SVM attacks on non-optical families.
To run PyTorch models on XPU, add `--backend torch_xpu --device 0`. With
`--backend auto`, PyTorch models use XPU when available and otherwise run on
CPU.

The CLI exposes the experiment options as corresponding kebab-case flags, for
example `--no-save-report
```

Run auto routing against unhashed SRAM responses:

```powershell
& ".\puf_env\Scripts\python.exe" -m puf_sim.puf_ml_attack `
--families sram `
--n 64 `
--response-bits 64 `
--noisiness 0 `
--no-sram-hash `
--attack-model auto` and `--print-progress`.
