"""Derived response-data metrics not directly exposed by pypuf."""
from __future__ import annotations

import numpy as np

from puf_sim.puf_backend import Backend, get_backend


def response_matrix(responses: np.ndarray, backend: Backend | None = None):
    """Return responses as ``(samples, response_bits)`` in ``{-1, 1}``."""
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim == 3:
        values = xp.sign(xp.mean(values, axis=-1))
    if values.ndim != 2:
        raise ValueError("Responses must have one, two, or three dimensions.")
    return backend.astype(xp.where(values > 0, 1, -1), xp.int8)


def binary_entropy(responses: np.ndarray, backend: Backend | None = None) -> float:
    """Return mean within-response 0/1 balance entropy, normalized to ``[0, 1]``.

    Each response is scored by the binary entropy of its fraction of positive
    bits. One-bit responses therefore have zero within-response entropy.
    """
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim == 3:
        values = values.reshape(-1, values.shape[-1])
    elif values.ndim != 2:
        raise ValueError("Responses must have one, two, or three dimensions.")
    if values.shape[1] < 1:
        raise ValueError("Responses must contain at least one response bit.")
    if values.shape[0] == 0:
        return 0.0

    probability_one = xp.mean(
        backend.astype(values > 0, xp.float32),
        axis=1,
    )
    non_deterministic = (probability_one > 0) & (probability_one < 1)
    probability = xp.where(non_deterministic, probability_one, 0.5)
    entropy = -(
        probability * xp.log2(probability)
        + (1 - probability) * xp.log2(1 - probability)
    )
    terms = xp.where(non_deterministic, entropy, 0.0)
    return float(backend.asnumpy(xp.mean(terms)))


def hamming_distance_distribution(responses: np.ndarray, backend: Backend | None = None) -> np.ndarray:
    """Return normalized pairwise Hamming distances.

    Two-dimensional input compares response rows directly. Three-dimensional
    population input is shaped ``(devices, challenges, response_bits)`` and
    returns one distance per device pair, averaged over challenges and bits.
    """
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim == 3:
        values = backend.astype(xp.where(values > 0, 1, -1), xp.float32)
        pair_distances = []
        for left_index in range(values.shape[0] - 1):
            right_values = values[left_index + 1:]
            distances = xp.mean(
                (1.0 - right_values * values[left_index]) / 2.0,
                axis=(1, 2),
            )
            pair_distances.append(distances)
        if not pair_distances:
            return np.array([], dtype=float)
        return backend.asnumpy(xp.concatenate(pair_distances))

    values = response_matrix(values, backend)
    values = backend.astype(values, xp.float32)
    pair_distances = []
    for left_index in range(values.shape[0] - 1):
        distances = xp.mean(
            (1.0 - values[left_index + 1:] * values[left_index]) / 2.0,
            axis=1,
        )
        pair_distances.append(distances)
    if not pair_distances:
        return np.array([], dtype=float)
    return backend.asnumpy(xp.concatenate(pair_distances))


def bit_aliasing(responses: np.ndarray, backend: Backend | None = None) -> np.ndarray:
    """Return the fraction of 1 responses for each output bit; 0.5 is balanced."""
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim != 3:
        raise ValueError("Bit aliasing expects (devices, challenges, response_bits).")

    values = xp.where(values > 0, 1, -1)
    probability_one = xp.mean(
        backend.astype(values == 1, xp.float32),
        axis=(0, 1),
    )
    return backend.asnumpy(probability_one)


def cross_challenge_randomness(
    responses: np.ndarray,
    backend: Backend | None = None,
) -> float:
    """Score low correlation across distinct challenges as high randomness.

    ``responses`` has shape ``(devices, challenges, response_bits)`` and must
    contain majority-voted responses. For each output bit, Pearson
    correlations are calculated across devices for every distinct challenge
    pair. The score is ``1 - mean(abs(correlation))`` over pairs where both
    challenge response vectors vary across devices. If no such pairs exist,
    the score is zero because the responses provide no evidence of
    cross-challenge independence.
    """
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim != 3:
        raise ValueError(
            "Cross-challenge randomness expects "
            "(devices, challenges, response_bits)."
        )
    if any(dimension < 1 for dimension in values.shape):
        raise ValueError("Responses must contain devices, challenges, and bits.")

    values = backend.astype(xp.where(values > 0, 1, -1), xp.float32)
    challenge_count = values.shape[1]
    upper_triangle = backend.asarray(
        np.triu(np.ones((challenge_count, challenge_count), dtype=np.float32), k=1)
    ) > 0
    absolute_correlation_sum = 0.0
    valid_pair_count = 0

    for bit_index in range(values.shape[2]):
        bit_responses = values[:, :, bit_index]
        centered = bit_responses - xp.mean(bit_responses, axis=0, keepdims=True)
        covariance = centered.T @ centered
        sums_of_squares = xp.sum(centered * centered, axis=0)
        denominator = xp.sqrt(
            sums_of_squares[:, None] * sums_of_squares[None, :]
        )
        valid_pairs = upper_triangle & (denominator > 0)
        safe_denominator = xp.where(valid_pairs, denominator, 1.0)
        correlations = covariance / safe_denominator
        absolute_correlation_sum += float(backend.asnumpy(
            xp.sum(xp.where(valid_pairs, xp.abs(correlations), 0.0))
        ))
        valid_pair_count += int(backend.asnumpy(xp.sum(
            backend.astype(valid_pairs, xp.int64)
        )))

    if valid_pair_count == 0:
        return 0.0
    return float(np.clip(1.0 - absolute_correlation_sum / valid_pair_count, 0.0, 1.0))


def uniqueness_from_responses(
    responses: np.ndarray,
    backend: Backend | None = None,
) -> np.ndarray:
    """Compute pypuf-style uniqueness from ``(devices, challenges, bits)`` responses."""
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim != 3:
        raise ValueError("Uniqueness expects (devices, challenges, response_bits).")
    if values.shape[0] < 2 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("Uniqueness requires at least two devices and non-empty responses.")

    device_count, challenge_count, response_bits = values.shape
    pair_count = device_count * (device_count - 1) // 2
    upper_triangle = backend.asarray(
        np.triu(np.ones((device_count, device_count), dtype=np.float32), k=1)
    ) > 0
    values = backend.astype(xp.where(values > 0, 1, -1), xp.float32)
    uniqueness = []
    for bit_index in range(response_bits):
        bit_values = values[:, :, bit_index]
        similarities = (
            (bit_values @ bit_values.T) / challenge_count + 1.0
        ) * 0.5
        mean_absolute_bias = xp.sum(
            xp.where(upper_triangle, xp.abs(0.5 - similarities), 0.0)
        ) / pair_count
        uniqueness.append(1.0 - 2.0 * mean_absolute_bias)
    return backend.asnumpy(xp.stack(uniqueness))


def majority_vote_repetitions(
    responses: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return per-sample, per-bit majority votes across the last axis.

    Exact ties are resolved independently with a fair draw from ``rng``.
    """
    values = np.asarray(responses)
    if values.ndim != 3:
        raise ValueError(
            "Repeated responses must have shape (samples, response_bits, repetitions)."
        )
    if values.shape[2] < 1:
        raise ValueError("Repeated responses must contain at least one repetition.")

    positive_counts = np.count_nonzero(values > 0, axis=2)
    ties = positive_counts * 2 == values.shape[2]
    votes_one = positive_counts * 2 > values.shape[2]
    tie_count = int(np.count_nonzero(ties))
    if tie_count:
        votes_one[ties] = rng.integers(0, 2, size=tie_count).astype(bool)
    return np.where(votes_one, 1, -1).astype(np.int8)


def probability_of_misidentification(responses: np.ndarray, backend: Backend | None = None) -> float:
    """Return the average complete-response impostor collision probability."""
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim != 3:
        raise ValueError("Misidentification expects (devices, challenges, response_bits).")
    pair_rates = []
    for left_index in range(values.shape[0] - 1):
        matches = xp.all(values[left_index + 1:] == values[left_index], axis=2)
        pair_rates.append(xp.mean(backend.astype(matches, xp.float32), axis=1))
    if not pair_rates:
        return 0.0
    return float(backend.asnumpy(xp.mean(xp.concatenate(pair_rates))))


__all__ = [
    "binary_entropy",
    "bit_aliasing",
    "cross_challenge_randomness",
    "hamming_distance_distribution",
    "majority_vote_repetitions",
    "probability_of_misidentification",
    "response_matrix",
    "uniqueness_from_responses",
]
