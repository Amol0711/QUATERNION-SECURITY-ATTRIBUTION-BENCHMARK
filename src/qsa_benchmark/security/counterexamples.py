"""Exact counterexample families and metric identities.

These routines are finite-domain research witnesses.  They are not
cryptographic implementations and make no computational-hardness claim.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import product
import random

from .finite import (
    Vector,
    add_mod,
    checksum_zero_distribution,
    npcr,
    public_diffusion,
    total_variation,
    uaci,
    uniform_distribution,
)


def _marginal(distribution: dict[Vector, Fraction], coordinate: int) -> dict[int, Fraction]:
    out: dict[int, Fraction] = {}
    for point, probability in distribution.items():
        out[point[coordinate]] = out.get(point[coordinate], Fraction()) + probability
    return out


def _pair_marginal(
    distribution: dict[Vector, Fraction], i: int, j: int
) -> dict[tuple[int, int], Fraction]:
    out: dict[tuple[int, int], Fraction] = {}
    for point, probability in distribution.items():
        key = (point[i], point[j])
        out[key] = out.get(key, Fraction()) + probability
    return out


def checksum_counterexample_summary(modulus: int = 4, dimension: int = 4) -> dict[str, object]:
    """Return the exact checksum-zero statistical counterexample (CEX-A)."""
    if dimension < 3:
        raise ValueError("pairwise-independence statement requires dimension >= 3")
    conditioned = checksum_zero_distribution(modulus, dimension)
    uniform = uniform_distribution(modulus, dimension)
    target_single = {value: Fraction(1, modulus) for value in range(modulus)}
    target_pair = {
        pair: Fraction(1, modulus * modulus)
        for pair in product(range(modulus), repeat=2)
    }
    return {
        "modulus": modulus,
        "dimension": dimension,
        "support_size": len(conditioned),
        "full_domain_size": modulus**dimension,
        "single_coordinate_uniform": all(
            _marginal(conditioned, i) == target_single for i in range(dimension)
        ),
        "pairwise_independent": all(
            _pair_marginal(conditioned, i, j) == target_pair
            for i in range(dimension)
            for j in range(i + 1, dimension)
        ),
        "total_variation": total_variation(conditioned, uniform),
        "checksum_distinguisher_advantage": Fraction(modulus - 1, modulus),
    }


def _fixed_pair(modulus: int = 256, dimension: int = 16, rounds: int = 3):
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    left = tuple((17 * i + 3) % modulus for i in range(dimension))
    right_list = list(left)
    right_list[dimension // 2] = (right_list[dimension // 2] + 1) % modulus
    right = tuple(right_list)
    return (
        left,
        right,
        public_diffusion(left, modulus, rounds),
        public_diffusion(right, modulus, rounds),
    )


def reuse_metric_pair_summary(
    modulus: int = 256, dimension: int = 16, rounds: int = 3
) -> dict[str, object]:
    """Return a same-mask *reuse* witness without one-shot posterior claims.

    This witness is used only to show that matched-mask differential metrics
    are functions of the transformed pair and the realized common mask.  It
    is intentionally separated from the one-shot side-information experiment.
    """
    left, right, encoded_left, encoded_right = _fixed_pair(modulus, dimension, rounds)
    mask = tuple((73 * i + 19) % modulus for i in range(dimension))
    cipher_left = add_mod(encoded_left, mask, modulus)
    cipher_right = add_mod(encoded_right, mask, modulus)
    return {
        "modulus": modulus,
        "dimension": dimension,
        "rounds": rounds,
        "same_realized_mask": True,
        "npcr": npcr(cipher_left, cipher_right),
        "uaci": uaci(cipher_left, cipher_right, modulus),
        "message_left": left,
        "message_right": right,
        "mask": mask,
        "cipher_left": cipher_left,
        "cipher_right": cipher_right,
    }


def one_shot_side_information_summary(
    modulus: int = 4, dimension: int = 3
) -> dict[str, object]:
    """Compute the exact CEX-B posterior under secret versus public mask.

    Messages and one-time masks are independent and uniform on Z_q^n.  For a
    fixed observed ciphertext, every message has exactly one compatible mask,
    so the posterior is uniform when the mask remains secret.  Publishing the
    realized mask yields exact recovery.
    """
    if modulus < 2 or dimension < 1:
        raise ValueError("invalid one-shot domain")
    domain = [tuple(x) for x in product(range(modulus), repeat=dimension)]
    observed = tuple((3 * i + 1) % modulus for i in range(dimension))
    compatible = []
    for message in domain:
        mask = tuple((c - m) % modulus for c, m in zip(observed, message, strict=True))
        assert add_mod(message, mask, modulus) == observed
        compatible.append((message, mask))
    posterior_mass = Fraction(1, len(compatible))
    public_mask = compatible[0][1]
    recovered = tuple((c - s) % modulus for c, s in zip(observed, public_mask, strict=True))
    return {
        "modulus": modulus,
        "dimension": dimension,
        "ciphertext": observed,
        "compatible_message_count": len(compatible),
        "secret_mask_exact_recovery_probability": posterior_mass,
        "public_mask_exact_recovery_probability": Fraction(1, 1),
        "public_mask": public_mask,
        "public_mask_recovered_message": recovered,
    }


def metric_equivalent_pair_summary(
    modulus: int = 256, dimension: int = 16, rounds: int = 3
) -> dict[str, object]:
    """Backward-compatible alias for the reuse metric witness."""
    return reuse_metric_pair_summary(modulus, dimension, rounds)


def npcr_common_mask_summary(left: Vector, right: Vector, mask: Vector, modulus: int) -> dict[str, object]:
    """Return the exact NPCR common-mask identity for additive masking."""
    if not left or len(left) != len(right) or len(left) != len(mask):
        raise ValueError("nonempty equal-length vectors required")
    masked_left = add_mod(left, mask, modulus)
    masked_right = add_mod(right, mask, modulus)
    return {
        "baseline": npcr(left, right),
        "masked": npcr(masked_left, masked_right),
        "invariant": npcr(left, right) == npcr(masked_left, masked_right),
    }


def uaci_coordinate_law(delta: int, modulus: int) -> dict[int, Fraction]:
    """Exact law of |U-(U+delta mod q)| for uniform U.

    The law makes UACI explicitly key-indexed under additive masking.  For
    delta=0 the difference is zero.  Otherwise the two possible differences
    are delta and q-delta, with probabilities (q-delta)/q and delta/q.
    """
    if modulus < 2 or delta < 0 or delta >= modulus:
        raise ValueError("invalid delta/modulus")
    if delta == 0:
        return {0: Fraction(1, 1)}
    out: dict[int, Fraction] = {}
    out[delta] = out.get(delta, Fraction()) + Fraction(modulus - delta, modulus)
    out[modulus - delta] = out.get(modulus - delta, Fraction()) + Fraction(delta, modulus)
    return dict(sorted(out.items()))


def exact_uaci_mask_distribution(left: Vector, right: Vector, modulus: int) -> dict[Fraction, Fraction]:
    """Enumerate the exact UACI distribution over all additive masks."""
    if not left or len(left) != len(right):
        raise ValueError("nonempty equal-length vectors required")
    counts: Counter[Fraction] = Counter()
    total = modulus ** len(left)
    for mask in product(range(modulus), repeat=len(left)):
        counts[uaci(add_mod(left, tuple(mask), modulus), add_mod(right, tuple(mask), modulus), modulus)] += 1
    return {value: Fraction(count, total) for value, count in sorted(counts.items())}


def fixed_uaci_key_distribution(sample_count: int = 20_000, seed: int = 11) -> dict[str, object]:
    """Return the fixed 20,000-mask byte-domain UACI distribution summary."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    modulus, dimension, rounds = 256, 16, 3
    _, _, encoded_left, encoded_right = _fixed_pair(modulus, dimension, rounds)
    rng = random.Random(seed)
    values: list[Fraction] = []
    for _ in range(sample_count):
        mask = tuple(rng.randrange(modulus) for _ in range(dimension))
        values.append(
            uaci(
                add_mod(encoded_left, mask, modulus),
                add_mod(encoded_right, mask, modulus),
                modulus,
            )
        )
    values.sort()
    def quantile_index(frac: float) -> int:
        return min(sample_count - 1, max(0, int(frac * sample_count)))
    ideal = Fraction(3346, 10_000)
    near = sum(abs(float(value) - float(ideal)) < 0.01 for value in values)
    return {
        "sample_count": sample_count,
        "seed": seed,
        "minimum": values[0],
        "p05": values[quantile_index(0.05)],
        "median": values[quantile_index(0.50)],
        "p95": values[quantile_index(0.95)],
        "maximum": values[-1],
        "within_0p01_of_0p3346": near,
        "distinct_values": len(set(values)),
    }
