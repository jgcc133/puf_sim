"""Analytical models for PUF families not implemented by pypuf."""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from pypuf.io import random_inputs
from pypuf.simulation import Simulation

from puf_sim.puf_backend import Backend
from .fuzzy_extractor import BCHCodeOffsetFuzzyExtractor


def _clip_probability(probability: np.ndarray | float) -> np.ndarray:
    return np.clip(probability, 0.0, 1.0)


def _symmetric_flip_numpy(
    values: np.ndarray,
    probability: np.ndarray | float,
    rng: np.random.Generator,
) -> np.ndarray:
    probabilities = _clip_probability(probability)
    flips = rng.random(values.shape) < probabilities
    return np.where(flips, -values, values)


def _symmetric_flip_backend(
    values: np.ndarray,
    probability: np.ndarray | float,
    rng: np.random.Generator,
    backend: Backend,
):
    probabilities = _clip_probability(probability)
    flips = rng.random(values.shape) < probabilities
    xp = backend.xp
    flip_mask = backend.asarray(flips, dtype=xp.bool)
    return xp.where(flip_mask, -values, values)


def _asymmetric_flip_numpy(
    values: np.ndarray,
    positive_flip_probability: np.ndarray | float,
    negative_flip_probability: np.ndarray | float,
    rng: np.random.Generator,
) -> np.ndarray:
    positive_probabilities = _clip_probability(positive_flip_probability)
    negative_probabilities = _clip_probability(negative_flip_probability)
    random_draws = rng.random(values.shape)
    positive_flips = (values > 0) & (random_draws < positive_probabilities)
    negative_flips = (values < 0) & (random_draws < negative_probabilities)
    return np.where(positive_flips | negative_flips, -values, values)


def _asymmetric_flip_backend(
    values: np.ndarray,
    positive_flip_probability: np.ndarray | float,
    negative_flip_probability: np.ndarray | float,
    rng: np.random.Generator,
    backend: Backend,
):
    positive_probabilities = _clip_probability(positive_flip_probability)
    negative_probabilities = _clip_probability(negative_flip_probability)
    random_draws = rng.random(values.shape)
    xp = backend.xp
    values_np = backend.asnumpy(values)
    positive_flips = (values_np > 0) & (random_draws < positive_probabilities)
    negative_flips = (values_np < 0) & (random_draws < negative_probabilities)
    flip_mask = backend.asarray(positive_flips | negative_flips, dtype=xp.bool)
    return xp.where(flip_mask, -values, values)


def hash_sram_responses(combined_bits: np.ndarray, response_bits: int) -> np.ndarray:
    """Hash challenge/fingerprint XOR rows using the SRAM response format."""
    values = np.asarray(combined_bits, dtype=np.bool_)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Combined SRAM bits must be a non-empty 2D array.")
    if response_bits < 1 or response_bits > 256:
        raise ValueError("response_bits must be between 1 and 256 for SHA-256 responses.")

    packed_bits = np.packbits(values, axis=1, bitorder="big")
    prefix = b"pypuf.sram-puf.v1:" + str(values.shape[1]).encode("ascii") + b":"
    responses = np.empty((len(values), response_bits), dtype=np.int8)
    for index, packed in enumerate(packed_bits):
        digest = hashlib.sha256(prefix + packed.tobytes()).digest()
        digest_bits = np.unpackbits(
            np.frombuffer(digest, dtype=np.uint8),
            bitorder="big",
        )
        responses[index] = np.where(digest_bits[:response_bits] == 1, 1, -1)
    return responses


class RingOscillatorPUF(Simulation):
    """Analytical ring-oscillator PUF based on challenge-selected frequencies."""

    def __init__(
        self,
        n: int,
        response_bits: int = 1,
        seed: int = 0,
        noisiness: float = 0.0,
        uncertainty_margin: float = 0.05,
    ) -> None:
        self.n = n
        self.m = response_bits
        self.noisiness = float(noisiness)
        self.uncertainty_margin = float(max(uncertainty_margin, 1e-9))
        rng = np.random.default_rng(seed)
        self.frequencies = rng.normal(0.0, 1.0, size=(n, response_bits))
        self.offsets = rng.normal(0.0, 0.25, size=response_bits)
        self.noise_rng = np.random.default_rng(seed + 10_001)

    @property
    def challenge_length(self) -> int:
        return self.n

    @property
    def response_length(self) -> int:
        return self.m

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        values = np.asarray(challenges, dtype=float) @ self.frequencies + self.offsets
        responses = np.where(values >= 0, 1, -1).astype(np.int8)
        if self.noisiness <= 0:
            return responses

        mean_abs = np.maximum(np.mean(np.abs(values), axis=1, keepdims=True), 1e-6)
        relative_margin = np.abs(values) / mean_abs
        near_boundary = relative_margin < self.uncertainty_margin
        closeness = np.where(
            near_boundary,
            1.0 - (relative_margin / self.uncertainty_margin),
            0.0,
        )
        flip_probability = self.noisiness * closeness
        return _symmetric_flip_numpy(responses, flip_probability, self.noise_rng).astype(np.int8)

    def eval_backend(self, challenges: np.ndarray, backend: Backend):
        xp = backend.xp
        values = backend.asarray(challenges, dtype=xp.float32) @ backend.asarray(
            self.frequencies,
            dtype=xp.float32,
        )
        values = values + backend.asarray(self.offsets, dtype=xp.float32)
        responses = backend.astype(xp.where(values >= 0, 1, -1), xp.int8)
        if self.noisiness <= 0:
            return responses

        values_np = backend.asnumpy(values)
        mean_abs = np.maximum(np.mean(np.abs(values_np), axis=1, keepdims=True), 1e-6)
        relative_margin = np.abs(values_np) / mean_abs
        near_boundary = relative_margin < self.uncertainty_margin
        closeness = np.where(
            near_boundary,
            1.0 - (relative_margin / self.uncertainty_margin),
            0.0,
        )
        flip_probability = self.noisiness * closeness
        noised = _symmetric_flip_backend(
            responses,
            flip_probability,
            self.noise_rng,
            backend,
        )
        return backend.astype(noised, xp.int8)


class StaticRandomAccessMemoryPUF(Simulation):
    """Analytical SRAM PUF with mismatch-driven startup of cross-coupled inverters.

    Each of n SRAM cells models a pair of cross-coupled inverters. Independent
    NMOS and PMOS threshold-voltage offsets perturb each inverter's switching
    point; the difference between the two inverter offsets selects the cell's
    preferred power-up state. A BCH code-offset fuzzy extractor enrolls the
    n-cell fingerprint and corrects noisy reads before the recovered
    fingerprint is XORed with each challenge. The combined bits can optionally
    be SHA-256 hashed. This is a reduced-order model, not a transistor-level
    circuit simulation.
    """

    def __init__(
        self,
        n: int,
        response_bits: int = 1,
        seed: int = 0,
        noisiness: float = 0.01,
        transistor_mismatch_sigma: float = 0.02,
        bch_correction_capacity: int = 1,
        hash_responses: bool = True,
    ) -> None:
        if n < 1:
            raise ValueError("n must be at least one.")
        if response_bits < 1 or response_bits > 256:
            raise ValueError("response_bits must be between 1 and 256 for SHA-256 responses.")
        if not hash_responses and response_bits > n:
            raise ValueError(
                "Unhashed SRAM responses cannot have more bits than the challenge length."
            )
        if not np.isfinite(noisiness) or noisiness < 0.0 or noisiness > 1.0:
            raise ValueError("SRAM noisiness must be between 0.0 and 1.0.")
        if not np.isfinite(transistor_mismatch_sigma) or transistor_mismatch_sigma <= 0:
            raise ValueError("transistor_mismatch_sigma must be finite and greater than zero.")
        self.n = n
        self.m = response_bits
        self.noisiness = float(noisiness)
        self.hash_responses = hash_responses
        self.transistor_mismatch_sigma = float(transistor_mismatch_sigma)
        self.bch_correction_capacity = bch_correction_capacity
        rng = np.random.default_rng(seed)
        self._transistor_vth_mismatch = rng.normal(
            loc=0.0,
            scale=self.transistor_mismatch_sigma,
            size=(n, 2, 2),
        )
        inverter_offsets = (
            self._transistor_vth_mismatch[:, :, 0]
            - self._transistor_vth_mismatch[:, :, 1]
        )
        self._cell_differential_mismatch = inverter_offsets[:, 0] - inverter_offsets[:, 1]
        self._power_up_state = np.where(
            self._cell_differential_mismatch >= 0.0,
            1,
            -1,
        ).astype(np.int8)
        self._cell_margin = np.abs(self._cell_differential_mismatch) / (
            2.0 * self.transistor_mismatch_sigma
        )
        self.fuzzy_extractor = BCHCodeOffsetFuzzyExtractor(
            self._power_up_state > 0,
            correction_capacity=bch_correction_capacity,
            rng=np.random.default_rng(seed + 40_001),
        )
        self.noise_rng = np.random.default_rng(seed + 20_001)

    @property
    def helper_data(self) -> np.ndarray:
        """Public BCH code-offset helper data produced during enrollment."""
        return self.fuzzy_extractor.helper_data

    @property
    def bch_uncorrectable_reads(self) -> int:
        """Number of reads whose syndromes exceeded the decoder capacity."""
        return self.fuzzy_extractor.uncorrectable_reads

    @property
    def bch_total_reads(self) -> int:
        """Number of reads processed by the fuzzy extractor."""
        return self.fuzzy_extractor.total_reads

    @property
    def challenge_length(self) -> int:
        return self.n

    @property
    def response_length(self) -> int:
        return self.m

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        challenge_values = np.asarray(challenges)
        if challenge_values.ndim != 2 or challenge_values.shape[1] != self.n:
            raise ValueError(f"Challenges must have shape (samples, {self.n}).")

        count = challenge_values.shape[0]
        cell_states = np.broadcast_to(self._power_up_state, (count, self.n)).copy()
        if self.noisiness > 0:
            instability = np.exp(-self._cell_margin)
            flip_probability = self.noisiness * instability.reshape(1, -1)
            cell_states = _symmetric_flip_numpy(
                cell_states,
                flip_probability,
                self.noise_rng,
            ).astype(np.int8)

        challenge_bits = challenge_values > 0
        corrected_fingerprints = np.empty((count, self.n), dtype=np.bool_)
        for index, observation in enumerate(cell_states > 0):
            corrected_fingerprints[index], _ = self.fuzzy_extractor.reproduce(
                observation
            )
        combined_bits = np.logical_xor(challenge_bits, corrected_fingerprints)
        if not self.hash_responses:
            return np.where(combined_bits[:, :self.m], 1, -1).astype(np.int8)

        return hash_sram_responses(combined_bits, self.m)

    def eval_backend(self, challenges: np.ndarray, backend: Backend):
        responses = self.eval(backend.asnumpy(challenges))
        return backend.asarray(responses, dtype=backend.xp.int8)


class MemristivePUF(Simulation):
    """Analytical memristive PUF using device-specific conductance thresholds."""

    def __init__(
        self,
        n: int,
        response_bits: int = 1,
        seed: int = 0,
        noisiness: float = 0.0,
    ) -> None:
        self.n = n
        self.m = response_bits
        self.noisiness = float(noisiness)
        rng = np.random.default_rng(seed)
        self.conductance = rng.normal(0.0, 1.0, size=(n, response_bits))
        self.threshold = rng.normal(0.0, 0.25, size=response_bits)
        self.noise_rng = np.random.default_rng(seed + 30_001)

    @property
    def challenge_length(self) -> int:
        return self.n

    @property
    def response_length(self) -> int:
        return self.m

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        values = np.asarray(challenges, dtype=float) @ self.conductance
        values = np.tanh(values) - self.threshold
        responses = np.where(values >= 0, 1, -1).astype(np.int8)
        if self.noisiness <= 0:
            return responses

        uncertainty = np.exp(-np.abs(values))
        positive_flip_probability = self.noisiness * (0.35 + 0.65 * uncertainty)
        negative_flip_probability = self.noisiness * 0.15 * uncertainty
        return _asymmetric_flip_numpy(
            responses,
            positive_flip_probability,
            negative_flip_probability,
            self.noise_rng,
        ).astype(np.int8)

    def eval_backend(self, challenges: np.ndarray, backend: Backend):
        xp = backend.xp
        values = backend.asarray(challenges, dtype=xp.float32) @ backend.asarray(
            self.conductance,
            dtype=xp.float32,
        )
        values = xp.tanh(values) - backend.asarray(self.threshold, dtype=xp.float32)
        responses = backend.astype(xp.where(values >= 0, 1, -1), xp.int8)
        if self.noisiness <= 0:
            return responses

        values_np = backend.asnumpy(values)
        uncertainty = np.exp(-np.abs(values_np))
        positive_flip_probability = self.noisiness * (0.35 + 0.65 * uncertainty)
        negative_flip_probability = self.noisiness * 0.15 * uncertainty
        noised = _asymmetric_flip_backend(
            responses,
            positive_flip_probability,
            negative_flip_probability,
            self.noise_rng,
            backend,
        )
        return backend.astype(noised, xp.int8)


class SiliconPhotonicPUF(Simulation):
    """Analytical silicon-photonic PUF based on a complex transfer matrix."""

    def __init__(
        self,
        n: int,
        response_bits: int = 1,
        seed: int = 0,
        noisiness: float = 0.0,
        calibration_samples: int = 1000,
        calibration_seed: int = 0,
    ) -> None:
        self.n = n
        self.m = response_bits
        self.seed = seed
        self.noisiness = float(noisiness)
        self.calibration_samples = calibration_samples
        self.calibration_seed = calibration_seed
        rng = np.random.default_rng(seed)
        amplitude = rng.uniform(0.1, 1.0, size=(n, response_bits))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, response_bits))
        self.transfer_matrix = amplitude * np.exp(1j * phase)
        self._reference: np.ndarray | None = None
        self._backend_states: dict[str, tuple[Any, Any, Any, Any]] = {}
        self.noise_rng = np.random.default_rng(seed + 40_001)

    @property
    def reference(self) -> np.ndarray:
        if self._reference is None:
            calibration_challenges = random_inputs(
                n=self.n,
                N=self.calibration_samples,
                seed=self.calibration_seed,
            )
            calibration_field = calibration_challenges @ self.transfer_matrix
            calibration_intensity = np.abs(calibration_field) ** 2
            self._reference = np.median(calibration_intensity, axis=0)
        return self._reference

    @property
    def challenge_length(self) -> int:
        return self.n

    @property
    def response_length(self) -> int:
        return self.m

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        challenge_values = np.asarray(challenges, dtype=float)
        if self.noisiness > 0:
            transform_flip_probability = min(0.5, self.noisiness * 0.1)
            transform_flips = self.noise_rng.random(challenge_values.shape) < transform_flip_probability
            challenge_values = np.where(transform_flips, -challenge_values, challenge_values)

        optical_field = challenge_values @ self.transfer_matrix
        intensity = np.abs(optical_field) ** 2
        if self.noisiness > 0:
            intensity_scale = np.maximum(np.std(intensity, axis=0, keepdims=True), 1e-6)
            intensity = intensity + self.noise_rng.normal(
                loc=0.0,
                scale=self.noisiness * intensity_scale,
                size=intensity.shape,
            )
            threshold_noise = self.noise_rng.normal(
                loc=0.0,
                scale=self.noisiness * 0.05 * np.maximum(np.abs(self.reference), 1e-6),
                size=self.reference.shape,
            )
        else:
            threshold_noise = 0.0
        return np.where(intensity >= (self.reference + threshold_noise), 1, -1).astype(np.int8)

    @classmethod
    def prepare_backend_batch(
        cls,
        instances: Sequence[SiliconPhotonicPUF],
        backend: Backend,
        batch_size: int | None = None,
    ) -> None:
        """Prepare cached transfer matrices and calibration thresholds on XPU."""
        if backend.name != "torch_xpu":
            return
        if not instances:
            return

        state_key = str(backend.device)
        groups: dict[tuple[int, int, int, int], list[SiliconPhotonicPUF]] = {}
        for instance in instances:
            if state_key in instance._backend_states:
                continue
            group_key = (
                instance.n,
                instance.m,
                instance.calibration_samples,
                instance.calibration_seed,
            )
            groups.setdefault(group_key, []).append(instance)

        step = batch_size or len(instances)
        xp = backend.xp
        for (n, _, calibration_samples, calibration_seed), group in groups.items():
            calibration_challenges = backend.asarray(
                random_inputs(n=n, N=calibration_samples, seed=calibration_seed),
                dtype=xp.float32,
            )
            for start in range(0, len(group), step):
                chunk = group[start:start + step]
                transfer_matrices = backend.asarray(
                    np.stack([puf.transfer_matrix for puf in chunk]),
                    dtype=xp.complex64,
                )
                calibration_batch = calibration_challenges.unsqueeze(0).expand(
                    len(chunk), -1, -1
                )
                calibration_fields_real = xp.bmm(
                    calibration_batch,
                    transfer_matrices.real,
                )
                calibration_fields_imag = xp.bmm(
                    calibration_batch,
                    transfer_matrices.imag,
                )
                intensities = (
                    calibration_fields_real ** 2
                    + calibration_fields_imag ** 2
                )
                ordered_intensities = xp.sort(intensities, dim=1).values
                lower = (calibration_samples - 1) // 2
                upper = calibration_samples // 2
                references = (
                    ordered_intensities[:, lower, :]
                    + ordered_intensities[:, upper, :]
                ) * 0.5
                references_numpy = backend.asnumpy(references)

                for index, puf in enumerate(chunk):
                    generator = xp.Generator(device=backend.device)
                    generator.manual_seed(puf.seed + 40_001)
                    puf._reference = references_numpy[index]
                    puf._backend_states[state_key] = (
                        transfer_matrices[index].real,
                        transfer_matrices[index].imag,
                        references[index],
                        generator,
                    )

    @classmethod
    def eval_population_backend(
        cls,
        instances: Sequence[SiliconPhotonicPUF],
        challenges: np.ndarray,
        backend: Backend,
        batch_size: int | None = None,
    ) -> Any:
        """Evaluate a PUF batch with its photonic and noise calculations on XPU."""
        if backend.name != "torch_xpu":
            raise ValueError("Silicon-photonic batch evaluation requires torch_xpu.")
        if not instances:
            raise ValueError("At least one silicon-photonic PUF is required.")
        if any(
            (puf.n, puf.m, puf.noisiness) !=
            (instances[0].n, instances[0].m, instances[0].noisiness)
            for puf in instances
        ):
            raise ValueError("Batched silicon-photonic PUFs must have matching dimensions and noisiness.")

        cls.prepare_backend_batch(instances, backend, batch_size=batch_size)
        xp = backend.xp
        state_key = str(backend.device)
        matrices = xp.stack(
            [puf._backend_states[state_key][0] for puf in instances],
            dim=0,
        )
        imaginary_matrices = xp.stack(
            [puf._backend_states[state_key][1] for puf in instances],
            dim=0,
        )
        references = xp.stack(
            [puf._backend_states[state_key][2] for puf in instances],
            dim=0,
        )
        generator = instances[0]._backend_states[state_key][3]
        challenge_values = backend.asarray(challenges, dtype=xp.float32)
        noisiness = backend.asarray(
            [puf.noisiness for puf in instances],
            dtype=xp.float32,
        ).reshape(-1, 1, 1)

        if instances[0].noisiness > 0:
            random_values = xp.rand(
                (len(instances), *challenge_values.shape),
                dtype=xp.float32,
                device=backend.device,
                generator=generator,
            )
            flip_mask = random_values < noisiness * 0.1
            challenge_batch = xp.where(
                flip_mask,
                -challenge_values[None, :, :],
                challenge_values[None, :, :],
            )
        else:
            challenge_batch = challenge_values.unsqueeze(0).expand(
                len(instances), -1, -1
            )

        optical_fields_real = xp.bmm(
            challenge_batch,
            matrices,
        )
        optical_fields_imag = xp.bmm(
            challenge_batch,
            imaginary_matrices,
        )
        intensities = optical_fields_real ** 2 + optical_fields_imag ** 2
        if instances[0].noisiness > 0:
            intensity_scale = xp.std(
                intensities,
                dim=1,
                keepdim=True,
                correction=0,
            ).clamp_min(1e-6)
            intensity_noise = xp.randn(
                intensities.shape,
                dtype=xp.float32,
                device=backend.device,
                generator=generator,
            )
            threshold_noise = xp.randn(
                references.shape,
                dtype=xp.float32,
                device=backend.device,
                generator=generator,
            )
            intensities = intensities + (
                intensity_noise * noisiness * intensity_scale
            )
            references = references + (
                threshold_noise
                * noisiness.reshape(-1, 1)
                * 0.05
                * xp.abs(references).clamp_min(1e-6)
            )
        return backend.astype(
            xp.where(intensities >= references[:, None, :], 1, -1),
            xp.int8,
        )

    def eval_backend(self, challenges: np.ndarray, backend: Backend):
        return self.eval_population_backend([self], challenges, backend)[0]


__all__ = [
    "MemristivePUF",
    "RingOscillatorPUF",
    "SiliconPhotonicPUF",
    "StaticRandomAccessMemoryPUF",
]
