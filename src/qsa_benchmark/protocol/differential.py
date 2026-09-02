from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping

import numpy as np
from scipy.stats import binom, norm

from .models import DifferentialDecision, PerturbationSpec
from .registry import method_policy, operation_regime

CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}


def perturbation_registry() -> dict[str, PerturbationSpec]:
    rows: dict[str, PerturbationSpec] = {}
    for location in ("CENTER", "QUARTER", "MAXGRAD", "MINGRAD"):
        for name, channel in CHANNEL_INDEX.items():
            pid = f"SP_{location}_{name}_UNIT"
            rows[pid] = PerturbationSpec(
                pid, "single_pixel", location.lower(), (channel,), 1, 1, "primary",
                f"one {name}-channel coordinate at {location.lower()} changed by one without wrap",
            )
    for magnitude in (17, 127):
        for name, channel in CHANNEL_INDEX.items():
            pid = f"SP_CENTER_{name}_MAG{magnitude}"
            rows[pid] = PerturbationSpec(
                pid, "single_pixel", "center", (channel,), magnitude, 1, "secondary",
                f"one center {name}-channel coordinate changed by {magnitude} without wrap",
            )
    for name, channel in CHANNEL_INDEX.items():
        pid = f"PATCH8_CENTER_{name}_UNIT"
        rows[pid] = PerturbationSpec(
            pid, "patch", "center_patch_8", (channel,), 1, 64, "secondary",
            f"center 8x8 patch in {name} changed by one without wrap",
        )
    rows["SPARSE16_RGB_UNIT"] = PerturbationSpec(
        "SPARSE16_RGB_UNIT", "sparse", "deterministic_sparse_16", (0, 1, 2), 1, 16, "secondary",
        "sixteen deterministic pixels, all RGB coordinates changed by one without wrap",
    )
    rows["CHECKER_LSB_ALL"] = PerturbationSpec(
        "CHECKER_LSB_ALL", "structured", "checkerboard", (0, 1, 2), 1, None, "secondary",
        "even-parity checkerboard pixels, all RGB coordinates changed by one without wrap",
    )
    rows["ROW_CENTER_ALL_UNIT"] = PerturbationSpec(
        "ROW_CENTER_ALL_UNIT", "structured", "center_row", (0, 1, 2), 1, None, "secondary",
        "center row, all RGB coordinates changed by one without wrap",
    )
    return rows


PERTURBATIONS = perturbation_registry()


def _luminance(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    return 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]


def _gradient_coordinate(image: np.ndarray, maximum: bool) -> tuple[int, int]:
    luminance = _luminance(image)
    gy, gx = np.gradient(luminance)
    magnitude = gx * gx + gy * gy
    if min(magnitude.shape) > 2:
        candidate = magnitude[1:-1, 1:-1]
        index = int(np.argmax(candidate) if maximum else np.argmin(candidate))
        row, column = np.unravel_index(index, candidate.shape)
        return int(row + 1), int(column + 1)
    index = int(np.argmax(magnitude) if maximum else np.argmin(magnitude))
    row, column = np.unravel_index(index, magnitude.shape)
    return int(row), int(column)


def _single_location(image: np.ndarray, location: str) -> tuple[int, int]:
    height, width = image.shape[:2]
    if location == "center":
        return height // 2, width // 2
    if location == "quarter":
        return height // 4, width // 4
    if location == "maxgrad":
        return _gradient_coordinate(image, True)
    if location == "mingrad":
        return _gradient_coordinate(image, False)
    raise ValueError(f"location does not identify one pixel: {location}")


def _change_without_wrap(values: np.ndarray, magnitude: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.int16)
    upward = source + magnitude
    downward = source - magnitude
    changed = np.where(upward <= 255, upward, downward)
    if np.any((changed < 0) | (changed > 255)):
        raise ValueError("declared magnitude cannot be applied without wrap")
    return changed.astype(np.uint8)


def _sparse_indices(height: int, width: int, count: int) -> np.ndarray:
    h = hashlib.sha256(f"QSA-PROTOCOL-SPARSE|{height}|{width}|{count}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return np.sort(rng.choice(height * width, size=min(count, height * width), replace=False))


def apply_perturbation(image: np.ndarray, perturbation_id: str) -> np.ndarray:
    try:
        spec = PERTURBATIONS[perturbation_id]
    except KeyError as exc:
        raise KeyError(f"unknown perturbation: {perturbation_id}") from exc
    source = np.asarray(image, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("perturbations require an RGB uint8 image")
    changed = source.copy()
    height, width = source.shape[:2]

    if spec.family == "single_pixel":
        row, column = _single_location(source, spec.location)
        for channel in spec.channels:
            changed[row, column, channel] = _change_without_wrap(
                source[row, column, channel : channel + 1], spec.magnitude
            )[0]
    elif spec.family == "patch":
        patch_h, patch_w = min(8, height), min(8, width)
        row0, col0 = (height - patch_h) // 2, (width - patch_w) // 2
        for channel in spec.channels:
            values = source[row0 : row0 + patch_h, col0 : col0 + patch_w, channel]
            changed[row0 : row0 + patch_h, col0 : col0 + patch_w, channel] = _change_without_wrap(values, spec.magnitude)
    elif spec.family == "sparse":
        indices = _sparse_indices(height, width, 16)
        rows, columns = np.unravel_index(indices, (height, width))
        for channel in spec.channels:
            changed[rows, columns, channel] = _change_without_wrap(source[rows, columns, channel], spec.magnitude)
    elif perturbation_id == "CHECKER_LSB_ALL":
        rows, columns = np.indices((height, width))
        mask = (rows + columns) % 2 == 0
        for channel in spec.channels:
            changed[..., channel][mask] = _change_without_wrap(source[..., channel][mask], 1)
    elif perturbation_id == "ROW_CENTER_ALL_UNIT":
        row = height // 2
        for channel in spec.channels:
            changed[row, :, channel] = _change_without_wrap(source[row, :, channel], 1)
    else:
        raise ValueError(f"unsupported perturbation family: {spec.family}")

    delta = changed != source
    if not np.any(delta):
        raise RuntimeError(f"perturbation made no change: {perturbation_id}")
    if spec.expected_pixel_count is not None:
        pixels = int(np.count_nonzero(np.any(delta, axis=2)))
        expected = min(spec.expected_pixel_count, height * width)
        if pixels != expected:
            raise RuntimeError(f"perturbation pixel-count mismatch for {perturbation_id}: {pixels} != {expected}")
    return changed


def npcr_uaci(left: bytes, right: bytes) -> tuple[float, float, int]:
    if len(left) != len(right) or not left:
        raise ValueError("differential metrics require equal nonempty bodies")
    a = np.frombuffer(left, dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(right, dtype=np.uint8).astype(np.int16)
    count = int(np.count_nonzero(a != b))
    return count / len(a), float(np.mean(np.abs(a - b)) / 255.0), count


def ideal_uaci_moments(q: int = 256) -> tuple[float, float]:
    if q < 2:
        raise ValueError("alphabet size must be at least two")
    mean = (q + 1) / (3.0 * q)
    variance = ((q + 1) * (q * q + 2)) / (18.0 * q * q * (q - 1))
    return mean, variance


def _npcr_lower_count(n: int, alpha: float, q: int) -> int:
    p = 1.0 - 1.0 / q
    candidate = int(binom.ppf(alpha, n, p))
    while candidate >= 0 and float(binom.cdf(candidate, n, p)) > alpha:
        candidate -= 1
    return candidate


def differential_decision(left: bytes, right: bytes, alpha: float, q: int = 256) -> DifferentialDecision:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    npcr, uaci, count = npcr_uaci(left, right)
    n = len(left)
    lower_count = _npcr_lower_count(n, alpha, q)
    npcr_p = float(binom.cdf(count, n, 1.0 - 1.0 / q))
    mean, variance = ideal_uaci_moments(q)
    standard_error = math.sqrt(variance / n)
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    lower, upper = mean - critical * standard_error, mean + critical * standard_error
    z_score = (uaci - mean) / standard_error
    uaci_p = float(min(1.0, 2.0 * norm.sf(abs(z_score))))
    return DifferentialDecision(
        n, npcr, uaci, npcr_p, uaci_p, count <= lower_count,
        not (lower <= uaci <= upper), lower_count, lower, upper,
    )


def holm_rejections(p_values: Iterable[float], alpha: float) -> list[bool]:
    values = [float(value) for value in p_values]
    if any(not (0.0 <= value <= 1.0) for value in values):
        raise ValueError("p-values must lie in [0,1]")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    rejected = [False] * len(values)
    for rank, index in enumerate(order):
        if values[index] <= alpha / (len(values) - rank):
            rejected[index] = True
        else:
            break
    return rejected


def pairwise_flip_probability(decisions: Iterable[bool]) -> float:
    values = [bool(value) for value in decisions]
    if len(values) < 2:
        return float("nan")
    ones = sum(values); zeros = len(values) - ones
    return (ones * zeros) / (len(values) * (len(values) - 1) // 2)


def planned_pair_counts(config: Mapping[str, object]) -> list[dict[str, object]]:
    corpus = config["corpus"]  # type: ignore[index]
    tiers = config["execution_tiers"]  # type: ignore[index]
    methods_all = list(config["methods"])  # type: ignore[index]
    perturbations = config["perturbations"]  # type: ignore[index]
    rows: list[dict[str, object]] = []
    for tier_id, tier_raw in tiers.items():  # type: ignore[union-attr]
        tier = dict(tier_raw)
        methods = methods_all if tier["methods"] == "all" else list(tier["methods"])
        images = list(corpus[str(tier["images"])])
        perturbation_ids = list(perturbations[str(tier["perturbations"])]) if isinstance(tier["perturbations"], str) else list(tier["perturbations"])
        for method_id in methods:
            policy = method_policy(str(method_id))
            for protocol_name in tier["protocols"]:
                if protocol_name == "P2_fresh_randomness" and not policy.p2_applicable:
                    continue
                for size in tier["sizes"]:
                    states = int(tier["state_repetitions_by_size"][str(size)])
                    pairs = len(images) * len(perturbation_ids) * int(tier["key_repetitions"]) * states
                    rows.append({
                        "tier_id": tier_id,
                        "method_id": method_id,
                        "protocol": protocol_name,
                        "operation_regime": operation_regime(str(method_id), str(protocol_name)),
                        "image_size": int(size),
                        "image_count": len(images),
                        "perturbation_count": len(perturbation_ids),
                        "key_repetitions": int(tier["key_repetitions"]),
                        "state_repetitions": states,
                        "planned_pairs": pairs,
                        "planned_encryptions": 2 * pairs,
                    })
    return rows
