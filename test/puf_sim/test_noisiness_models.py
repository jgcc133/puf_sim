import hashlib

import numpy as np
import pytest

from pypuf.io import random_inputs
from pypuf.metrics import reliability
from pypuf.simulation import InterposePUF as ReferenceInterposePUF
from pypuf.simulation import PermutationPUF as ReferencePermutationPUF
from pypuf.simulation import Simulation
from puf_sim.puf_backend import get_backend, xpu_available
from puf_sim.puf_implementations import StaticRandomAccessMemoryPUF, create_puf
from puf_sim.puf_implementations.analytical import SiliconPhotonicPUF
from puf_sim.puf_implementations.fuzzy_extractor import (
    BCHCodeOffsetFuzzyExtractor,
    BinaryBCHCode,
)


@pytest.mark.parametrize(
    "family,response_bits",
    [
        ("optical", 8),
        ("ring_oscillator", 8),
        ("sram", 32),
        ("memristive", 8),
        ("silicon_photonic", 8),
    ],
)
def test_same_seed_reproduces_responses(family: str, response_bits: int) -> None:
    noisiness = 0.01 if family == "sram" else 0.0
    puf = create_puf(
        family,
        n=64,
        seed=11,
        response_bits=response_bits,
        noisiness=noisiness,
    )
    challenges = random_inputs(n=puf.challenge_length, N=128, seed=23)
    first = np.asarray(puf.eval(challenges))
    same_device = create_puf(
        family,
        n=64,
        seed=11,
        response_bits=response_bits,
        noisiness=noisiness,
    )
    second = np.asarray(same_device.eval(challenges))
    np.testing.assert_array_equal(first, second)


def test_ff_apuf_generates_response_bits_from_sequential_lanes() -> None:
    n = 12
    response_bits = 4
    seed = 71
    ff = [(4, 8)]
    challenges = random_inputs(n=n, N=32, seed=73)
    puf = create_puf(
        "ff_apuf",
        n=n,
        response_bits=response_bits,
        seed=seed,
        ff=ff,
        noisiness=0.0,
    )
    expected = np.column_stack([
        create_puf(
            "ff_apuf",
            n=n,
            response_bits=1,
            seed=Simulation.seed(
                "FeedForwardArbiterPUF response bit "
                f"{bit_index} seed {seed}"
            ) if bit_index else seed,
            ff=ff,
            noisiness=0.0,
        ).eval(challenges)
        for bit_index in range(response_bits)
    ])

    assert puf.response_length == response_bits
    assert np.asarray(puf.eval(challenges)).shape == (len(challenges), response_bits)
    np.testing.assert_array_equal(puf.eval(challenges), expected)


@pytest.mark.skipif(not xpu_available(), reason="Intel XPU backend is unavailable")
def test_ff_apuf_xpu_evaluation_preserves_multiple_response_bits() -> None:
    backend = get_backend("torch_xpu", device=0)
    puf = create_puf(
        "ff_apuf",
        n=12,
        response_bits=3,
        seed=83,
        ff=[(4, 8)],
        noisiness=0.0,
    )
    challenges = random_inputs(n=12, N=16, seed=89)

    actual = backend.asnumpy(backend.eval_array(puf, challenges))

    assert actual.shape == (len(challenges), 3)
    np.testing.assert_array_equal(actual, puf.eval(challenges))


def test_xor_arbiter_response_bits_partition_all_chains() -> None:
    response_bits = 3
    puf = create_puf(
        "xor_apuf",
        n=8,
        k=7,
        response_bits=response_bits,
        seed=37,
        noisiness=0.0,
    )
    challenges = random_inputs(n=puf.challenge_length, N=16, seed=39)

    transformed = puf.transform(challenges, puf.k)
    chain_responses = np.sign(puf.ltf_eval(transformed))
    expected_groups = []
    start = 0
    base_group_size, extra_chains = divmod(puf.k, response_bits)
    for group_index in range(response_bits):
        group_size = base_group_size + (group_index < extra_chains)
        stop = start + group_size
        expected_groups.append(np.prod(chain_responses[:, start:stop], axis=1))
        start = stop

    responses = np.asarray(puf.eval(challenges))
    assert puf.response_length == response_bits
    assert responses.shape == (len(challenges), response_bits)
    np.testing.assert_array_equal(responses, np.column_stack(expected_groups))


def test_xor_arbiter_keeps_single_bit_output_shape_and_checks_width() -> None:
    challenges = random_inputs(n=8, N=4, seed=41)
    puf = create_puf("xor_apuf", n=8, response_bits=1, seed=43)
    assert puf.k == 8
    assert np.asarray(puf.eval(challenges)).shape == (len(challenges),)

    wider_output = create_puf(
        "xor_apuf", n=8, k=4, response_bits=5, seed=43
    )
    assert wider_output.k == 5
    assert np.asarray(wider_output.eval(challenges)).shape == (len(challenges), 5)


def test_permutation_puf_generates_multiple_response_bits_from_all_chains() -> None:
    response_bits = 3
    puf = create_puf(
        "permutation",
        n=16,
        response_bits=response_bits,
        seed=59,
        noisiness=0.0,
    )
    challenges = random_inputs(n=puf.challenge_length, N=16, seed=61)

    chain_responses = np.sign(puf.ltf_eval(puf.transform(challenges, puf.k)))
    expected_groups = []
    start = 0
    base_group_size, extra_chains = divmod(puf.k, response_bits)
    for group_index in range(response_bits):
        group_size = base_group_size + (group_index < extra_chains)
        stop = start + group_size
        expected_groups.append(np.prod(chain_responses[:, start:stop], axis=1))
        start = stop

    responses = np.asarray(puf.eval(challenges))
    assert puf.k == 16
    assert puf.response_length == response_bits
    assert responses.shape == (len(challenges), response_bits)
    np.testing.assert_array_equal(responses, np.column_stack(expected_groups))
    table = puf._permutation_table
    for chain_index in range(puf.k - 1):
        assert np.all(table[chain_index] != table[-1])
    np.testing.assert_array_equal(
        np.sort(table, axis=0),
        np.broadcast_to(np.arange(puf.k)[:, None], table.shape),
    )


def test_one_bit_permutation_puf_matches_existing_chain_combiner() -> None:
    challenges = random_inputs(n=16, N=16, seed=67)
    actual = create_puf(
        "permutation",
        n=16,
        k=4,
        response_bits=1,
        seed=71,
        noisiness=0.0,
    )
    reference = ReferencePermutationPUF(n=16, k=4, seed=71, noisiness=0.0)

    assert actual.response_length == 1
    assert np.asarray(actual.eval(challenges)).shape == (len(challenges),)
    np.testing.assert_array_equal(actual.eval(challenges), reference.eval(challenges))


def test_interpose_puf_partitions_all_lower_stage_chains() -> None:
    response_bits = 3
    puf = create_puf(
        "interpose",
        n=16,
        k=8,
        response_bits=response_bits,
        interpose_pos=7,
        seed=73,
        noisiness=0.0,
    )
    challenges = random_inputs(n=puf.challenge_length, N=16, seed=75)
    interposed = puf.up.eval(challenges).reshape(len(challenges), 1)
    down_challenges = np.concatenate(
        (
            challenges[:, :puf.interpose_pos],
            interposed,
            challenges[:, puf.interpose_pos:],
        ),
        axis=1,
    )
    chain_responses = np.sign(
        puf.down.ltf_eval(puf.down.transform(down_challenges, puf.down.k))
    )
    expected_groups = []
    start = 0
    base_group_size, extra_chains = divmod(puf.down.k, response_bits)
    for group_index in range(response_bits):
        group_size = base_group_size + (group_index < extra_chains)
        stop = start + group_size
        expected_groups.append(np.prod(chain_responses[:, start:stop], axis=1))
        start = stop

    responses = np.asarray(puf.eval(challenges))
    assert puf.up.k == 1
    assert puf.down.k == 8
    assert puf.response_length == response_bits
    assert responses.shape == (len(challenges), response_bits)
    np.testing.assert_array_equal(responses, np.column_stack(expected_groups))


def test_interpose_default_chain_count_and_single_bit_compatibility() -> None:
    challenges = random_inputs(n=16, N=16, seed=77)
    default = create_puf(
        "interpose",
        n=16,
        response_bits=3,
        interpose_pos=7,
        seed=79,
        noisiness=0.0,
    )
    assert default.down.k == 16
    assert np.asarray(default.eval(challenges)).shape == (len(challenges), 3)

    actual = create_puf(
        "interpose",
        n=16,
        k=4,
        response_bits=1,
        interpose_pos=7,
        seed=81,
        noisiness=0.0,
    )
    reference = ReferenceInterposePUF(
        n=16,
        k_down=4,
        k_up=1,
        interpose_pos=7,
        seed=81,
        noisiness=0.0,
    )
    assert actual.response_length == 1
    assert np.asarray(actual.eval(challenges)).shape == (len(challenges),)
    np.testing.assert_array_equal(actual.eval(challenges), reference.eval(challenges))


@pytest.mark.parametrize(
    "family,response_bits,noisiness",
    [
        ("optical", 8, 0.4),
        ("ring_oscillator", 8, 0.4),
        ("sram", 32, 0.4),
        ("memristive", 8, 0.4),
        ("silicon_photonic", 8, 0.4),
    ],
)
def test_noisiness_introduces_repeated_read_variation(
    family: str,
    response_bits: int,
    noisiness: float,
) -> None:
    puf = create_puf(
        family,
        n=64,
        seed=29,
        response_bits=response_bits,
        noisiness=noisiness,
    )
    challenges = random_inputs(n=puf.challenge_length, N=256, seed=31)
    first = np.asarray(puf.eval(challenges))
    second = np.asarray(puf.eval(challenges))
    assert np.any(first != second)


@pytest.mark.skipif(not xpu_available(), reason="Intel XPU backend is unavailable")
def test_silicon_photonic_xpu_batch_uses_cached_calibration_and_noise():
    backend = get_backend("torch_xpu", device=0)
    pufs = [
        SiliconPhotonicPUF(
            n=16,
            response_bits=3,
            seed=53 + index,
            noisiness=0.2,
            calibration_samples=32,
            calibration_seed=59,
        )
        for index in range(2)
    ]
    challenges = random_inputs(n=16, N=8, seed=61)
    responses = SiliconPhotonicPUF.eval_population_backend(
        pufs,
        challenges,
        backend,
        batch_size=1,
    )

    assert tuple(responses.shape) == (2, 8, 3)
    for puf in pufs:
        expected_reference = np.median(
            np.abs(
                random_inputs(
                    n=16,
                    N=32,
                    seed=59,
                ) @ puf.transfer_matrix
            ) ** 2,
            axis=0,
        )
        np.testing.assert_allclose(puf.reference, expected_reference, rtol=1e-5, atol=1e-5)
    assert set(np.unique(backend.asnumpy(responses))).issubset({-1, 1})


def test_sram_startup_state_is_derived_from_differential_transistor_mismatch() -> None:
    puf = StaticRandomAccessMemoryPUF(
        n=8,
        response_bits=4,
        seed=41,
        transistor_mismatch_sigma=0.03,
    )
    mismatch = puf._transistor_vth_mismatch
    assert mismatch.shape == (8, 2, 2)

    inverter_offsets = mismatch[:, :, 0] - mismatch[:, :, 1]
    differential_mismatch = inverter_offsets[:, 0] - inverter_offsets[:, 1]
    np.testing.assert_allclose(puf._cell_differential_mismatch, differential_mismatch)
    np.testing.assert_array_equal(
        puf._power_up_state,
        np.where(differential_mismatch >= 0.0, 1, -1),
    )


def test_sram_hashes_challenge_xor_startup_fingerprint() -> None:
    puf = StaticRandomAccessMemoryPUF(
        n=8,
        response_bits=16,
        seed=43,
        noisiness=0.01,
    )

    class NoFlipRng:
        @staticmethod
        def random(shape: tuple[int, ...]) -> np.ndarray:
            return np.ones(shape)

    puf.noise_rng = NoFlipRng()
    challenges = np.array(
        [[1, -1, 1, -1, 1, -1, 1, -1], [-1, 1, -1, 1, -1, 1, -1, 1]],
        dtype=np.int8,
    )
    combined_bits = np.logical_xor(
        challenges > 0,
        puf._power_up_state > 0,
    )
    packed_bits = np.packbits(combined_bits, axis=1, bitorder="big")
    prefix = b"pypuf.sram-puf.v1:8:"
    expected = []
    for packed in packed_bits:
        digest = hashlib.sha256(prefix + packed.tobytes()).digest()
        digest_bits = np.unpackbits(
            np.frombuffer(digest, dtype=np.uint8),
            bitorder="big",
        )[:puf.response_length]
        expected.append(np.where(digest_bits == 1, 1, -1))

    actual = puf.eval(challenges)
    assert actual.shape == (2, puf.response_length)
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.int8))


def test_sram_can_return_unhashed_challenge_xor_startup_fingerprint() -> None:
    puf = create_puf(
        "sram",
        n=8,
        response_bits=8,
        seed=47,
        noisiness=0.0,
        hash_responses=False,
    )
    challenges = random_inputs(n=8, N=32, seed=53)
    combined_bits = np.logical_xor(challenges > 0, puf._power_up_state > 0)
    expected = np.where(combined_bits, 1, -1).astype(np.int8)

    actual = puf.eval(challenges)

    assert actual.shape == (len(challenges), 8)
    np.testing.assert_array_equal(actual, expected)


def test_unhashed_sram_response_width_cannot_exceed_challenge_width() -> None:
    with pytest.raises(ValueError, match="cannot have more bits"):
        create_puf("sram", n=8, response_bits=9, hash_responses=False)


def test_sram_mismatch_sigma_must_be_positive() -> None:
    with pytest.raises(ValueError, match="transistor_mismatch_sigma"):
        StaticRandomAccessMemoryPUF(
            n=8,
            transistor_mismatch_sigma=0.0,
            noisiness=0.01,
        )


def test_bch_code_corrects_up_to_its_capacity() -> None:
    code = BinaryBCHCode(length=64, correction_capacity=2)
    message = np.arange(code.message_length) % 2 == 0
    encoded = code.encode(message)
    observed = encoded.copy()
    observed[[3, 51]] = ~observed[[3, 51]]

    corrected, success = code.correct(observed)

    assert success
    np.testing.assert_array_equal(corrected, encoded)


def test_code_offset_fuzzy_extractor_reproduces_fingerprint() -> None:
    fingerprint = np.arange(64) % 3 == 0
    extractor = BCHCodeOffsetFuzzyExtractor(
        fingerprint,
        correction_capacity=2,
        rng=np.random.default_rng(53),
    )
    observation = fingerprint.copy()
    observation[[9, 42]] = ~observation[[9, 42]]

    recovered, success = extractor.reproduce(observation)

    assert success
    np.testing.assert_array_equal(recovered, fingerprint)
    assert extractor.helper_data.shape == fingerprint.shape


def test_sram_default_noise_and_zero_noise_are_supported() -> None:
    assert create_puf("sram", n=64).noisiness == 0.01
    noiseless_puf = create_puf("sram", n=64, noisiness=0.0)
    assert noiseless_puf.noisiness == 0.0
    challenges = random_inputs(n=64, N=32, seed=67)
    first = noiseless_puf.eval(challenges)
    second = noiseless_puf.eval(challenges)
    np.testing.assert_array_equal(first, second)

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        StaticRandomAccessMemoryPUF(n=64, noisiness=-0.001)


def test_sram_reliability_exceeds_90_percent_at_one_percent_noise() -> None:
    puf = create_puf(
        "sram",
        n=64,
        response_bits=16,
        seed=21,
        noisiness=0.01,
        bch_correction_capacity=1,
    )

    score = float(np.mean(reliability(puf, seed=22, N=1000, r=17)))

    assert 0.90 <= score <= 0.97
