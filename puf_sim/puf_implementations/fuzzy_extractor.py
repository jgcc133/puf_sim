"""BCH code-offset fuzzy extraction for binary SRAM fingerprints."""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np


def _gf_multiply(left: int, right: int, polynomial: int, degree: int) -> int:
    result = 0
    high_bit = 1 << degree
    while right:
        if right & 1:
            result ^= left
        right >>= 1
        left <<= 1
        if left & high_bit:
            left ^= polynomial
    return result


@lru_cache(maxsize=None)
def _primitive_polynomial(degree: int) -> int:
    order = (1 << degree) - 1
    prime_factors = []
    factor = 2
    remaining = order
    while factor * factor <= remaining:
        if remaining % factor == 0:
            prime_factors.append(factor)
            while remaining % factor == 0:
                remaining //= factor
        factor += 1
    if remaining > 1:
        prime_factors.append(remaining)

    for lower_bits in range(1, 1 << degree, 2):
        polynomial = (1 << degree) | lower_bits
        if _gf_power(2, order, polynomial, degree) != 1:
            continue
        if all(
            _gf_power(2, order // prime, polynomial, degree) != 1
            for prime in prime_factors
        ):
            return polynomial
    raise ValueError(f"No primitive binary polynomial found for degree {degree}.")


def _gf_power(value: int, exponent: int, polynomial: int, degree: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = _gf_multiply(result, value, polynomial, degree)
        value = _gf_multiply(value, value, polynomial, degree)
        exponent >>= 1
    return result


class BinaryBCHCode:
    """Shortened primitive binary BCH code with one- or two-bit correction."""

    def __init__(self, length: int, correction_capacity: int) -> None:
        if length < 2:
            raise ValueError("BCH code length must be at least two.")
        if correction_capacity not in (1, 2):
            raise ValueError("BCH correction_capacity must be 1 or 2.")
        degree = length.bit_length()
        field_order = (1 << degree) - 1
        if length > field_order:
            raise ValueError("BCH code length exceeds the selected field order.")

        polynomial = _primitive_polynomial(degree)
        field_powers = [1]
        for _ in range(1, field_order):
            field_powers.append(
                _gf_multiply(field_powers[-1], 2, polynomial, degree)
            )

        self.length = length
        self.correction_capacity = correction_capacity
        self._degree = degree
        self._columns = []
        for index in range(length):
            first_syndrome = field_powers[index]
            if correction_capacity == 1:
                column = first_syndrome
            else:
                third_syndrome = field_powers[(3 * index) % field_order]
                column = first_syndrome | (third_syndrome << degree)
            self._columns.append(column)

        self._redundancy = degree * correction_capacity
        self._parity_positions = self._independent_columns(self._redundancy)
        self.message_length = length - self._redundancy
        if self.message_length < 1:
            raise ValueError(
                f"Code length {length} is too short for correction capacity "
                f"{correction_capacity}; at least {self._redundancy + 1} bits are required."
            )
        self._parity_syndromes = self._build_parity_syndromes()
        self._error_patterns = self._build_error_patterns()

    def _independent_columns(self, count: int) -> list[int]:
        basis: dict[int, int] = {}
        selected = []
        for index, column in enumerate(self._columns):
            reduced = column
            for pivot in sorted(basis, reverse=True):
                if reduced & (1 << pivot):
                    reduced ^= basis[pivot]
            if reduced:
                pivot = reduced.bit_length() - 1
                basis[pivot] = reduced
                selected.append(index)
                if len(selected) == count:
                    return selected
        raise ValueError("BCH parity-check matrix does not have full rank.")

    def _build_parity_syndromes(self) -> dict[int, int]:
        syndromes = {0: 0}
        for parity_index, code_index in enumerate(self._parity_positions):
            column = self._columns[code_index]
            for syndrome, mask in list(syndromes.items()):
                syndromes[syndrome ^ column] = mask | (1 << parity_index)
        return syndromes

    def _build_error_patterns(self) -> dict[int, tuple[int, ...]]:
        patterns = {0: ()}
        for weight in range(1, self.correction_capacity + 1):
            for positions in combinations(range(self.length), weight):
                syndrome = 0
                for position in positions:
                    syndrome ^= self._columns[position]
                if syndrome in patterns:
                    raise ValueError("BCH syndromes are not unique within correction capacity.")
                patterns[syndrome] = positions
        return patterns

    def encode(self, message: np.ndarray) -> np.ndarray:
        message_bits = np.asarray(message, dtype=np.bool_)
        if message_bits.shape != (self.message_length,):
            raise ValueError(
                f"Message must have shape ({self.message_length},)."
            )

        codeword = np.zeros(self.length, dtype=np.bool_)
        data_positions = [
            index
            for index in range(self.length)
            if index not in self._parity_positions
        ]
        syndrome = 0
        for bit, position in zip(message_bits, data_positions):
            codeword[position] = bit
            if bit:
                syndrome ^= self._columns[position]

        parity_mask = self._parity_syndromes[syndrome]
        for parity_index, position in enumerate(self._parity_positions):
            codeword[position] = bool(parity_mask & (1 << parity_index))
        return codeword

    def correct(self, received: np.ndarray) -> tuple[np.ndarray, bool]:
        word = np.asarray(received, dtype=np.bool_)
        if word.shape != (self.length,):
            raise ValueError(f"Received word must have shape ({self.length},).")

        syndrome = 0
        for bit, column in zip(word, self._columns):
            if bit:
                syndrome ^= column
        error_positions = self._error_patterns.get(syndrome)
        if error_positions is None:
            return word.copy(), False

        corrected = word.copy()
        corrected[list(error_positions)] = ~corrected[list(error_positions)]
        return corrected, True


class BCHCodeOffsetFuzzyExtractor:
    """Enroll a fingerprint and reproduce it from a bounded-noise read."""

    def __init__(
        self,
        fingerprint: np.ndarray,
        correction_capacity: int,
        rng: np.random.Generator,
    ) -> None:
        enrollment = np.asarray(fingerprint, dtype=np.bool_)
        if enrollment.ndim != 1:
            raise ValueError("Enrollment fingerprint must be a one-dimensional bit vector.")
        self.code = BinaryBCHCode(len(enrollment), correction_capacity)
        secret = rng.integers(0, 2, size=self.code.message_length, dtype=np.uint8).astype(
            np.bool_
        )
        codeword = self.code.encode(secret)
        self._helper_data = np.logical_xor(enrollment, codeword)
        self.uncorrectable_reads = 0
        self.total_reads = 0

    @property
    def helper_data(self) -> np.ndarray:
        """Return a copy of the public enrollment helper data."""
        return self._helper_data.copy()

    def reproduce(self, observation: np.ndarray) -> tuple[np.ndarray, bool]:
        noisy_fingerprint = np.asarray(observation, dtype=np.bool_)
        if noisy_fingerprint.shape != self._helper_data.shape:
            raise ValueError(
                f"Observation must have shape {self._helper_data.shape}."
            )
        noisy_codeword = np.logical_xor(noisy_fingerprint, self._helper_data)
        codeword, success = self.code.correct(noisy_codeword)
        if not success:
            self.uncorrectable_reads += 1
        self.total_reads += 1
        return np.logical_xor(self._helper_data, codeword), success


__all__ = ["BCHCodeOffsetFuzzyExtractor", "BinaryBCHCode"]
