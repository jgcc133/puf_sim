"""GPU-aware adapters for PUF families already implemented by pypuf."""
from __future__ import annotations

from typing import Any

import numpy as np

from puf_sim.puf_backend import Backend
from pypuf.simulation import (
    ArbiterPUF as _ArbiterPUF,
    FeedForwardArbiterPUF as _FeedForwardArbiterPUF,
    IntegratedOpticalPUF as _IntegratedOpticalPUF,
    InterposePUF as _InterposePUF,
    PermutationPUF as _PermutationPUF,
    XORArbiterPUF as _XORArbiterPUF,
)


def _att_backend(challenges: Any, xp: Any) -> Any:
    """Apply the arbiter threshold transform to ``(N, k, n)`` tensors."""
    return xp.flip(xp.cumprod(xp.flip(challenges, dims=(2,)), dim=2), dims=(2,))


def _noise_backend(instance: Any, backend: Backend, shape: tuple[int, ...], sigma: float) -> Any:
    noise = instance.random.normal(loc=0, scale=sigma, size=shape)
    return backend.asarray(noise, dtype=backend.xp.float32)


def _balanced_xor(values: Any, backend: Backend | None = None) -> Any:
    """XOR bipolar chain responses through pairwise, balanced reduction levels."""
    xp = backend.xp if backend is not None else np
    while values.shape[1] > 1:
        pair_count = values.shape[1] // 2
        paired_values = (
            values[:, :2 * pair_count:2]
            * values[:, 1:2 * pair_count:2]
        )
        if values.shape[1] % 2:
            if backend is not None and backend.name == "torch_xpu":
                values = xp.cat((paired_values, values[:, -1:]), dim=1)
            else:
                values = xp.concatenate((paired_values, values[:, -1:]), axis=1)
        else:
            values = paired_values
    return values[:, 0]


def _grouped_xor(
    chain_responses: Any,
    response_bits: int,
    backend: Backend | None = None,
) -> Any:
    xp = backend.xp if backend is not None else np
    base_group_size, extra_chains = divmod(chain_responses.shape[1], response_bits)
    output_responses = []
    start = 0
    for group_index in range(response_bits):
        group_size = base_group_size + (group_index < extra_chains)
        stop = start + group_size
        output_responses.append(
            _balanced_xor(chain_responses[:, start:stop], backend)
        )
        start = stop

    if backend is not None and backend.name == "torch_xpu":
        combined = xp.stack(output_responses, dim=1)
    else:
        combined = xp.stack(output_responses, axis=1)
    return combined[:, 0] if response_bits == 1 else combined


class _XORBackendMixin:
    def _combine_chain_responses(
        self, chain_responses: Any, backend: Backend
    ) -> Any:
        return _balanced_xor(chain_responses, backend)

    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        challenges = backend.asarray(challenges, dtype=xp.float32)
        sub_challenges = _att_backend(
            challenges[:, None, :].expand(-1, self.k, -1),
            xp,
        )
        weights = backend.asarray(self.weight_array[:, :-1], dtype=xp.float32)
        values = xp.einsum("nki,ki->nk", sub_challenges, weights)
        values = values + backend.asarray(self.weight_array[:, -1], dtype=xp.float32)
        values = values + _noise_backend(self, backend, (challenges.shape[0], self.k), self.sigma_noise)
        responses = xp.sign(values)
        grouped_responses = self._combine_chain_responses(responses, backend)
        return backend.astype(grouped_responses, xp.int8)


class XORArbiterPUF(_XORBackendMixin, _XORArbiterPUF):
    def __init__(
        self,
        n: int,
        k: int | None = None,
        seed: int | None = None,
        transform: Any = None,
        noisiness: float = 0.0,
        response_bits: int = 1,
    ) -> None:
        if (
            isinstance(response_bits, bool)
            or not isinstance(response_bits, (int, np.integer))
            or response_bits < 1
        ):
            raise ValueError("response_bits must be a positive integer.")
        if k is not None and (
            isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1
        ):
            raise ValueError("k must be a positive integer or None.")
        effective_k = n if k is None else max(int(k), int(response_bits))
        self.response_bits = int(response_bits)
        super().__init__(
            n=n,
            k=effective_k,
            seed=seed,
            transform=transform,
            noisiness=noisiness,
        )

    @property
    def response_length(self) -> int:
        return self.response_bits

    def _combine_chain_responses(
        self, chain_responses: Any, backend: Backend | None = None
    ) -> Any:
        return _grouped_xor(chain_responses, self.response_bits, backend)

    def val(self, challenges: np.ndarray) -> np.ndarray:
        sub_challenges = self.transform(challenges, self.k)
        chain_values = self.ltf_eval(sub_challenges)
        chain_responses = np.sign(chain_values)
        return self._combine_chain_responses(chain_responses)

    def eval(self, challenges: np.ndarray, block_size: int | None = 10**6) -> np.ndarray:
        count = challenges.shape[0]
        if count == 0:
            shape = (0,) if self.response_bits == 1 else (0, self.response_bits)
            return np.empty(shape, dtype=challenges.dtype)

        block_size = block_size or count
        responses = np.empty((count, self.response_bits), dtype=challenges.dtype)
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            responses[start:stop] = self.eval_block(challenges[start:stop]).reshape(
                stop - start, self.response_bits
            )
        return responses[:, 0] if self.response_bits == 1 else responses


class ArbiterPUF(_XORBackendMixin, _ArbiterPUF):
    pass


class FeedForwardArbiterPUF(_FeedForwardArbiterPUF):
    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        challenges = backend.asarray(challenges, dtype=xp.float32)
        count = challenges.shape[0]
        loops = sorted(self.ff, key=lambda loop: loop[0])
        feed_points = {feed_point for _, feed_point in loops}
        ff_challenges = xp.zeros((count, self.n + len(loops)), dtype=xp.float32, device=challenges.device)
        offset = 0
        for index in range(self.n + len(loops)):
            if index in feed_points:
                offset += 1
            else:
                ff_challenges[:, index] = challenges[:, index - offset]

        delay_difference = xp.zeros((count,), dtype=xp.float32, device=challenges.device)
        delay_difference_pos = 0
        for arbiter_point, feed_point in loops + [(self.n + len(loops), None)]:
            section = ff_challenges[:, delay_difference_pos:arbiter_point]
            transformed = _att_backend(section[:, None, :], xp)[:, 0, :]
            weights = backend.asarray(
                self.weight_array[:, delay_difference_pos:arbiter_point],
                dtype=xp.float32,
            )
            section_value = xp.sum(transformed * weights, dim=1)
            section_noise = np.random.default_rng(self.noise_prng.integers(2**32)).normal(
                loc=0,
                scale=self.sigma_noise_from_random_weights(
                    n=(arbiter_point or self.n) - delay_difference_pos,
                    sigma_weight=1,
                    noisiness=self.noisiness,
                ),
                size=(count,),
            )
            section_value = section_value + backend.asarray(section_noise, dtype=xp.float32)
            delay_difference = delay_difference * xp.prod(section, dim=1) + section_value
            delay_difference_pos = arbiter_point
            if feed_point:
                ff_challenges[:, feed_point] = xp.sign(delay_difference)

        return backend.astype(xp.sign(delay_difference), xp.int8)


class PermutationPUF(_XORBackendMixin, _PermutationPUF):
    def __init__(
        self,
        n: int,
        k: int | None = None,
        seed: int | None = None,
        noisiness: float = 0.0,
        response_bits: int = 1,
    ) -> None:
        if (
            isinstance(response_bits, bool)
            or not isinstance(response_bits, (int, np.integer))
            or response_bits < 1
        ):
            raise ValueError("response_bits must be a positive integer.")
        if k is not None and (
            isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1
        ):
            raise ValueError("k must be a positive integer or None.")
        effective_k = n if k is None else max(int(k), int(response_bits))
        self.response_bits = int(response_bits)
        super().__init__(n=n, k=effective_k, seed=seed, noisiness=noisiness)
        self._permutation_table = self._build_permutation_table(n, effective_k)

    @classmethod
    def _build_permutation_table(cls, n: int, k: int) -> np.ndarray:
        if n not in cls.FIXED_PERMUTATION_SEEDS:
            raise ValueError(
                f"Fixed permutation currently not supported for n={n}; "
                f"supported values are {tuple(cls.FIXED_PERMUTATION_SEEDS)}."
            )

        seeds = cls.FIXED_PERMUTATION_SEEDS[n]
        if k <= len(seeds):
            return np.stack(
                [
                    np.random.RandomState(seed).permutation(n)
                    for seed in seeds[:k]
                ],
                axis=0,
            )

        rng = np.random.default_rng(0x5045524D + n)
        labels = rng.permutation(n)
        inverse_labels = np.argsort(labels)
        shifts = np.arange(1, n)
        rng.shuffle(shifts)
        permutations = [
            inverse_labels[(labels + shift) % n]
            for shift in shifts
        ]
        permutations.append(np.arange(n))
        if k > n:
            seen = {permutation.tobytes() for permutation in permutations}
            while len(permutations) < k:
                candidate = rng.permutation(n)
                key = candidate.tobytes()
                if key not in seen:
                    permutations.append(candidate)
                    seen.add(key)
        return np.stack(permutations[:k], axis=0)

    @property
    def response_length(self) -> int:
        return self.response_bits

    def transform_fixed_permutation(
        self, challenges: np.ndarray, k: int
    ) -> np.ndarray:
        if k > self._permutation_table.shape[0]:
            raise ValueError("Requested more permutation chains than were initialized.")
        transformed = np.stack(
            [challenges[:, permutation] for permutation in self._permutation_table[:k]],
            axis=1,
        )
        self.att(transformed)
        return transformed

    def _combine_chain_responses(
        self, chain_responses: Any, backend: Backend | None = None
    ) -> Any:
        return _grouped_xor(chain_responses, self.response_bits, backend)

    def val(self, challenges: np.ndarray) -> np.ndarray:
        chain_values = self.ltf_eval(self.transform(challenges, self.k))
        return self._combine_chain_responses(np.sign(chain_values))

    def eval(self, challenges: np.ndarray, block_size: int | None = 10**6) -> np.ndarray:
        count = challenges.shape[0]
        if count == 0:
            shape = (0,) if self.response_bits == 1 else (0, self.response_bits)
            return np.empty(shape, dtype=challenges.dtype)

        block_size = block_size or count
        responses = np.empty((count, self.response_bits), dtype=challenges.dtype)
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            responses[start:stop] = self.eval_block(challenges[start:stop]).reshape(
                stop - start, self.response_bits
            )
        return responses[:, 0] if self.response_bits == 1 else responses

    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        challenges = backend.asarray(challenges, dtype=xp.float32)
        permutation_table = self._permutation_table[:self.k]
        if backend.name == "torch_xpu":
            transformed = xp.stack(
                [
                    challenges.index_select(
                        1,
                        backend.asarray(permutation, dtype=xp.int64),
                    )
                    for permutation in permutation_table
                ],
                dim=1,
            )
        else:
            transformed = xp.stack(
                [challenges[:, permutation] for permutation in permutation_table],
                axis=1,
            )
        return self._eval_transformed_backend(transformed, backend)

    def _eval_transformed_backend(self, transformed: Any, backend: Backend) -> Any:
        xp = backend.xp
        transformed = _att_backend(transformed, xp)
        weights = backend.asarray(self.weight_array[:, :-1], dtype=xp.float32)
        values = xp.einsum("nki,ki->nk", transformed, weights)
        values = values + backend.asarray(self.weight_array[:, -1], dtype=xp.float32)
        values = values + _noise_backend(self, backend, (transformed.shape[0], self.k), self.sigma_noise)
        responses = _grouped_xor(xp.sign(values), self.response_bits, backend)
        return backend.astype(responses, xp.int8)


class InterposePUF(_InterposePUF):
    def __init__(
        self,
        n: int,
        k_down: int | None = None,
        k_up: int = 1,
        interpose_pos: int | None = None,
        seed: int | None = None,
        noisiness: float = 0.0,
        response_bits: int = 1,
    ) -> None:
        if (
            isinstance(response_bits, bool)
            or not isinstance(response_bits, (int, np.integer))
            or response_bits < 1
        ):
            raise ValueError("response_bits must be a positive integer.")
        if k_down is not None and (
            isinstance(k_down, bool)
            or not isinstance(k_down, (int, np.integer))
            or k_down < 1
        ):
            raise ValueError("k_down must be a positive integer or None.")
        effective_k_down = (
            n if k_down is None else max(int(k_down), int(response_bits))
        )
        self.response_bits = int(response_bits)
        super().__init__(
            n=n,
            k_down=effective_k_down,
            k_up=k_up,
            interpose_pos=interpose_pos,
            seed=seed,
            noisiness=noisiness,
        )
        self.up = self._native_child(self.up)
        self.down = self._native_child(self.down)
        self.down.response_bits = self.response_bits

    @staticmethod
    def _native_child(child: Any) -> XORArbiterPUF:
        native_child = XORArbiterPUF.__new__(XORArbiterPUF)
        native_child.__dict__.update(child.__dict__)
        native_child.response_bits = 1
        return native_child

    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        challenges = backend.asarray(challenges, dtype=xp.float32)
        interpose_bits = self.up.eval_backend(challenges, backend).reshape(challenges.shape[0], 1)
        down_challenges = xp.cat(
            (challenges[:, :self.interpose_pos], interpose_bits, challenges[:, self.interpose_pos:]),
            dim=1,
        )
        return self.down.eval_backend(down_challenges, backend)


class OpticalPUF(_IntegratedOpticalPUF):
    def __init__(self, n: int, m: int, seed: int, noisiness: float = 0.0) -> None:
        super().__init__(n=n, m=m, seed=seed)
        self.noisiness = float(noisiness)
        self.noise_rng = np.random.default_rng(seed + 50_001)

    def eval(self, challenges: Any) -> Any:
        challenge_values = np.asarray(challenges, dtype=np.float32)
        if self.noisiness > 0:
            transform_flip_probability = min(0.5, self.noisiness * 0.1)
            transform_flips = self.noise_rng.random(challenge_values.shape) < transform_flip_probability
            challenge_values = np.where(transform_flips, -challenge_values, challenge_values)
        intensity = np.abs(challenge_values @ self.T) ** 2
        if self.noisiness > 0:
            scale = np.maximum(np.std(intensity, axis=0, keepdims=True), 1e-6)
            intensity = intensity + self.noise_rng.normal(
                loc=0.0,
                scale=self.noisiness * scale,
                size=intensity.shape,
            )
        return intensity

    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        challenge_values = backend.asnumpy(challenges)
        if self.noisiness > 0:
            transform_flip_probability = min(0.5, self.noisiness * 0.1)
            transform_flips = self.noise_rng.random(challenge_values.shape) < transform_flip_probability
            challenge_values = np.where(transform_flips, -challenge_values, challenge_values)

        values = backend.asarray(challenge_values, dtype=xp.float32)
        transfer = backend.asarray(self.T, dtype=xp.complex64)
        intensity = xp.abs(values.to(dtype=xp.complex64) @ transfer) ** 2
        if self.noisiness > 0:
            intensity_np = backend.asnumpy(intensity)
            scale = np.maximum(np.std(intensity_np, axis=0, keepdims=True), 1e-6)
            intensity_noise = self.noise_rng.normal(
                loc=0.0,
                scale=self.noisiness * scale,
                size=intensity_np.shape,
            )
            intensity = intensity + backend.asarray(intensity_noise, dtype=xp.float32)
        return intensity

__all__ = [
    "ArbiterPUF",
    "FeedForwardArbiterPUF",
    "InterposePUF",
    "OpticalPUF",
    "PermutationPUF",
    "XORArbiterPUF",
]
