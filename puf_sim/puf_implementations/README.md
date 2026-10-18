# PUF Implementations

This submodule provides one import location for the PUF families used by
`puf_sim`. Every implementation follows the `pypuf.simulation.Simulation`
interface:

- `challenge_length`
- `response_length`
- `eval(challenges)`

The native families delegate to existing `pypuf` implementations:

- Optical PUF
- Arbiter PUF
- Feed-Forward Arbiter PUF (FF-APUF)
- XOR Arbiter PUF (XOR-APUF)
- Interpose PUF
- Permutation PUF

The following families use deterministic analytical models because the base
`pypuf` library does not provide hardware-level implementations for them:

- Ring Oscillator PUF
- Static Random Access Memory PUF
- Memristive PUF
- Silicon Photonic PUF

These analytical models are useful for software evaluation and comparison;
they are not substitutes for measurements from physical devices.

### SRAM enrollment and noise correction

The SRAM model enrolls a mismatch-derived fingerprint with a shortened binary
BCH code-offset fuzzy extractor. Its public `helper_data` supports reproduction
from later noisy reads; corrected fingerprint bits are XORed with the challenge
and SHA-256 hashed by default. Set `hash_responses=False` to return the direct
challenge/fingerprint XOR bits instead, which is useful for comparing ML
attacks against a learnable, unhashed response law. Unhashed responses can have
at most `n` bits. BCH correction capacity defaults to one bit and can be set to
two when constructing an SRAM PUF. The simulated SRAM `noisiness` must be
between `0.0` and `1.0`; setting it to zero disables read noise.
Omitting `noisiness` from `create_puf` selects `0.01` for SRAM and preserves
the zero-noise default for other families.

```python
sram = create_puf(
    "sram",
    n=64,
    response_bits=16,
    seed=7,
    bch_correction_capacity=2,
)
public_helper_data = sram.helper_data

unhashed_sram = create_puf(
    "sram",
    n=64,
    response_bits=16,
    seed=7,
    noisiness=0.0,
    hash_responses=False,
)
```

The helper data is public by design, but this analytical model does not provide
a formal entropy guarantee, tamper resistance, or a hardware-calibrated noise
distribution. A real deployment must validate helper-data leakage and the
extractor against measured device entropy and environmental conditions. The
simulation exposes `bch_uncorrectable_reads` and `bch_total_reads` counters so
that out-of-capacity reads are observable.

## Python API

Import the factory and Create a 64-bit Arbiter PUF with multiple independent output chains:

```python
from puf_sim.puf_implementations import create_puf
from pypuf.io import random_inputs

puf = create_puf("arbiter", n=64, response_bits=3, seed=7)
challenges = random_inputs(n=puf.challenge_length, N=10, seed=8)
responses = puf.eval(challenges)

print(responses.shape)
```

The resulting shape is `(10, 3)`. Each response column comes from an
independent Arbiter chain; unlike an XOR Arbiter PUF, the chains are not XORed
together.

Create an XOR Arbiter PUF with two response bits, a multi-bit FF-APUF, and an
Interpose PUF. Each FF-APUF response bit is generated sequentially by an
independently seeded one-bit FF-APUF evaluated on the same challenge. For XOR
Arbiter and Interpose PUFs, the lower/output stage chain count
defaults to the challenge length `n` (64 chains here), or can be set with `k`.
The implementation raises a smaller explicit `k` to at least `response_bits`,
then partitions those chains into near-equal groups and combines each group in
a balanced XOR tree. Thus every output-stage chain contributes once, and the
response width is independently selected from `k`.

```python
from puf_sim.puf_implementations import create_puf

xor_puf = create_puf(
    "xor_apuf",
    n=64,
    response_bits=2,
    seed=1,
    noisiness=0.05,
)
ff_puf = create_puf(
    "ff_apuf",
    n=64,
    seed=2,
    response_bits=3,
    ff=[(20, 40)],
)
interpose_puf = create_puf(
    "interpose",
    n=64,
    k=4,
    response_bits=2,
    seed=3,
    interpose_pos=32,
)
```

Create the other families with the same factory:

```python
from puf_sim.puf_implementations import create_puf

families = [
    "optical",
    "arbiter",
    "ff_apuf",
    "xor_apuf",
    "interpose",
    "permutation",
    "ring_oscillator",
    "sram",
    "memristive",
    "silicon_photonic",
]

for family in families:
    puf = create_puf(family, n=64, seed=10, k=2, response_bits=8)
    print(family, puf.challenge_length, puf.response_length)
```

For XOR and Permutation PUFs, `k` is the total number of arbiter chains and
defaults to the challenge length `n`. `response_bits` divides those chains into
near-equal groups, with each group producing one output bit through a balanced
XOR tree; if `k < response_bits`, the chain count is raised to the output width.
The Permutation PUF uses one fixed challenge permutation per chain. Its original
curated permutation seeds are retained for up to ten chains. Larger banks use
a deterministic conjugated cyclic permutation set, which avoids fixed points
and gives each challenge position a distinct destination across the first `n`
chains. If more than `n` chains are requested, further unique permutations are
generated deterministically.

FF-APUF `response_bits` defaults to one and preserves the original one-bit
output shape. With a wider response, independently seeded FF-APUF lanes are
evaluated sequentially on each challenge and their outputs are concatenated.

Interpose PUF retains a single upper-stage bit that is inserted into the
challenge, then uses its `k` lower-stage arbiter chains to produce output
bits. The lower-stage chains are partitioned and balanced-XORed like XOR
Arbiter PUF chains; the upper stage remains one chain. `response_bits`
defaults to one, preserving scalar output. For Interpose PUF, omitted `k`
defaults the lower-stage count to `n`. `response_bits` also controls the
response width of the analytical families and the number of optical response
pixels. `noisiness` controls evaluation noise in supported models; SRAM accepts
values from `0.0` to `1.0`, with a default of `0.01` from `create_puf`.

The native classes are also available directly:

```python
from puf_sim.puf_implementations import (
    ArbiterPUF,
    FeedForwardArbiterPUF,
    InterposePUF,
    OpticalPUF,
    PermutationPUF,
    XORArbiterPUF,
)
from puf_sim.puf_implementations import (
    MemristivePUF,
    RingOscillatorPUF,
    SiliconPhotonicPUF,
    StaticRandomAccessMemoryPUF,
)

puf = RingOscillatorPUF(n=64, response_bits=8, seed=11)
```

## Command line

The package includes a small command-line smoke runner. From the repository
root, list the supported family names:

```powershell
python -m puf_sim.puf_implementations --list
```

Create and evaluate a 64-bit 4-XOR Arbiter PUF on 100 challenges:

```powershell
python -m puf_sim.puf_implementations `
    --family xor_apuf `
    --bits 64 `
    --chains 4 `
    --samples 100 `
    --seed 7
```

Create an analytical SRAM PUF with 32 response bits:

```powershell
python -m puf_sim.puf_implementations `
    --family sram `
    --bits 64 `
    --response-bits 32 `
    --samples 100
```

The command prints the selected family, challenge count, response width, and
response array shape. It is intended as a quick integration check; detailed
metric evaluation is provided by `puf_sim.puf_baseline_eval`.

## Permutation PUF note

The Permutation PUF supports the predefined challenge lengths from `16` through
`256` in steps of two. For up to ten chains it retains the upstream fixed
permutations; larger chain banks use the deterministic extension described
above.
