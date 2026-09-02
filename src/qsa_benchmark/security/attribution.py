"""Exact identities supporting the QSA-FORMAL-CHECKS attribution results."""
from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import Hashable, Sequence, TypeVar

from .finite import Distribution, Vector, add_mod, sub_mod, translate_distribution, uniform_distribution

T = TypeVar("T", bound=Hashable)


def translated_uniform_distribution(modulus: int, offset: Vector) -> Distribution:
    return translate_distribution(uniform_distribution(modulus, len(offset)), offset, modulus)


def nonce_reuse_relation(ciphertext_left: Vector, ciphertext_right: Vector, modulus: int) -> Vector:
    return sub_mod(ciphertext_left, ciphertext_right, modulus)


def integrity_tamper(ciphertext: Vector, delta: Vector, modulus: int) -> Vector:
    if all(item == 0 for item in delta):
        raise ValueError("delta must be nonzero")
    return add_mod(ciphertext, delta, modulus)


def choose_public_acceptance_set_forgery(accepted: Sequence[T], challenge: T) -> T:
    """Choose an accepted fresh object whenever the public set has size >= 2."""
    unique = tuple(dict.fromkeys(accepted))
    if len(unique) < 2:
        raise ValueError("public acceptance set must contain at least two values")
    if challenge not in unique:
        raise ValueError("challenge is outside the public acceptance set")
    for candidate in unique:
        if candidate != challenge:
            return candidate
    raise AssertionError("unreachable")


def collision_probability(query_count: int, nonce_space_size: int) -> Fraction:
    if query_count < 0 or nonce_space_size < 1:
        raise ValueError("invalid collision parameters")
    if query_count > nonce_space_size:
        return Fraction(1, 1)
    numerator = prod(range(nonce_space_size - query_count + 1, nonce_space_size + 1))
    return Fraction(1, 1) - Fraction(numerator, nonce_space_size**query_count)
