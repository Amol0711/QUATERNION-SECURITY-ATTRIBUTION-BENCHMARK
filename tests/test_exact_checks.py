from __future__ import annotations

from fractions import Fraction

from qsa_benchmark.security import (
    checksum_counterexample_summary,
    npcr_common_mask_summary,
    public_diffusion,
    public_diffusion_inverse,
    uaci_coordinate_law,
)


def test_public_diffusion_is_exactly_invertible() -> None:
    value = tuple(range(16))
    encoded = public_diffusion(value, 256, rounds=3)
    assert public_diffusion_inverse(encoded, 256, rounds=3) == value


def test_checksum_distribution_has_uniform_low_order_marginals() -> None:
    result = checksum_counterexample_summary(modulus=4, dimension=4)
    assert result["single_coordinate_uniform"]
    assert result["pairwise_independent"]
    assert result["total_variation"] == Fraction(3, 4)


def test_common_additive_mask_preserves_npcr() -> None:
    result = npcr_common_mask_summary((1, 2, 3), (1, 9, 3), (7, 8, 9), 16)
    assert result["invariant"]
    assert result["baseline"] == Fraction(1, 3)


def test_uaci_coordinate_law() -> None:
    assert uaci_coordinate_law(1, 256) == {1: Fraction(255, 256), 255: Fraction(1, 256)}
