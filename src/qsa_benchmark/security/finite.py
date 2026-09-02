"""Exact finite-domain helpers for the QSA-FORMAL-CHECKS proof witnesses.

These routines are not cryptographic implementations. They support exact
algebraic checks, calibrated controls, and benchmark construction.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
from itertools import product
from typing import Callable, Iterable, Mapping, TypeAlias

Vector: TypeAlias = tuple[int, ...]
Distribution: TypeAlias = dict[Vector, Fraction]


def _validate_vector(value: Vector, modulus: int) -> None:
    if modulus < 2:
        raise ValueError("modulus must be at least two")
    if any((not isinstance(item, int)) or item < 0 or item >= modulus for item in value):
        raise ValueError("vector coordinate outside finite domain")


def add_mod(left: Vector, right: Vector, modulus: int) -> Vector:
    if len(left) != len(right):
        raise ValueError("vector lengths differ")
    _validate_vector(left, modulus); _validate_vector(right, modulus)
    return tuple((a + b) % modulus for a, b in zip(left, right, strict=True))


def sub_mod(left: Vector, right: Vector, modulus: int) -> Vector:
    if len(left) != len(right):
        raise ValueError("vector lengths differ")
    _validate_vector(left, modulus); _validate_vector(right, modulus)
    return tuple((a - b) % modulus for a, b in zip(left, right, strict=True))


def uniform_distribution(modulus: int, dimension: int) -> Distribution:
    if modulus < 2 or dimension < 1:
        raise ValueError("invalid finite domain")
    points = list(product(range(modulus), repeat=dimension))
    mass = Fraction(1, len(points))
    return {tuple(point): mass for point in points}


def translate_distribution(distribution: Mapping[Vector, Fraction], offset: Vector, modulus: int) -> Distribution:
    output: dict[Vector, Fraction] = defaultdict(Fraction)
    for point, probability in distribution.items():
        output[add_mod(point, offset, modulus)] += probability
    return dict(output)


def total_variation(left: Mapping[Vector, Fraction], right: Mapping[Vector, Fraction]) -> Fraction:
    support = set(left) | set(right)
    return Fraction(1, 2) * sum(abs(left.get(x, Fraction()) - right.get(x, Fraction())) for x in support)


def apply_public_bijection(distribution: Mapping[Vector, Fraction], mapping: Callable[[Vector], Vector]) -> Distribution:
    output: dict[Vector, Fraction] = defaultdict(Fraction)
    for point, probability in distribution.items():
        output[mapping(point)] += probability
    return dict(output)


def invert_public_bijection(image: Vector, domain: Iterable[Vector], mapping: Callable[[Vector], Vector]) -> Vector:
    preimages = [point for point in domain if mapping(point) == image]
    if len(preimages) != 1:
        raise ValueError("mapping is not a bijection on supplied domain")
    return preimages[0]


def checksum_zero_distribution(modulus: int, dimension: int) -> Distribution:
    if modulus < 2 or dimension < 2:
        raise ValueError("invalid checksum domain")
    support = [tuple(point) for point in product(range(modulus), repeat=dimension) if sum(point) % modulus == 0]
    mass = Fraction(1, len(support))
    return {point: mass for point in support}


def public_diffusion(value: Vector, modulus: int, rounds: int = 3) -> Vector:
    """Public exactly invertible modular diffusion; deliberately not a cipher."""
    _validate_vector(value, modulus)
    if not value or rounds < 1:
        raise ValueError("nonempty value and positive rounds required")
    state = list(value); n = len(state)
    for _ in range(rounds):
        for index in range(n):
            state[index] = (state[index] + state[index - 1]) % modulus
        state = state[1:] + state[:1]
    return tuple(state)


def public_diffusion_inverse(value: Vector, modulus: int, rounds: int = 3) -> Vector:
    _validate_vector(value, modulus)
    if not value or rounds < 1:
        raise ValueError("nonempty value and positive rounds required")
    state = list(value); n = len(state)
    for _ in range(rounds):
        state = state[-1:] + state[:-1]
        for index in range(n - 1, -1, -1):
            state[index] = (state[index] - state[index - 1]) % modulus
    return tuple(state)


def npcr(left: Vector, right: Vector) -> Fraction:
    if len(left) != len(right) or not left:
        raise ValueError("nonempty equal-length vectors required")
    return Fraction(sum(a != b for a, b in zip(left, right, strict=True)), len(left))


def uaci(left: Vector, right: Vector, modulus: int) -> Fraction:
    if len(left) != len(right) or not left:
        raise ValueError("nonempty equal-length vectors required")
    _validate_vector(left, modulus); _validate_vector(right, modulus)
    return Fraction(sum(abs(a - b) for a, b in zip(left, right, strict=True)), len(left) * (modulus - 1))
