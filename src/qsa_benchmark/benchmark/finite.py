from __future__ import annotations

import math
import warnings
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

COEFF_Q = 18
COEFF_SCALE = 1 << COEFF_Q
MAX_ELEMENTARY_ANGLE = math.pi / 4


def round_div_ties_away(numerator: np.ndarray | int, denominator: int) -> np.ndarray:
    n = np.asarray(numerator, dtype=np.int64)
    abs_n = np.abs(n)
    q = (abs_n + denominator // 2) // denominator
    return np.where(n < 0, -q, q).astype(np.int64)


def quantize_coefficient(value: float) -> int:
    return int(math.floor(value * COEFF_SCALE + 0.5) if value >= 0 else math.ceil(value * COEFF_SCALE - 0.5))


def _lift_pair(values: np.ndarray, i: int, j: int, a: int, b: int, inverse: bool) -> None:
    x = values[..., i]
    y = values[..., j]
    if inverse:
        x1 = x - round_div_ties_away(a * y, COEFF_SCALE)
        y0 = y - round_div_ties_away(b * x1, COEFF_SCALE)
        x0 = x1 - round_div_ties_away(a * y0, COEFF_SCALE)
        values[..., i] = x0
        values[..., j] = y0
    else:
        x1 = x + round_div_ties_away(a * y, COEFF_SCALE)
        y1 = y + round_div_ties_away(b * x1, COEFF_SCALE)
        x2 = x1 + round_div_ties_away(a * y1, COEFF_SCALE)
        values[..., i] = x2
        values[..., j] = y1


def angle_step(angle: float, i: int, j: int) -> dict[str, int]:
    a = quantize_coefficient(-math.tan(angle / 2.0))
    b = quantize_coefficient(math.sin(angle))
    return {"i": i, "j": j, "a": a, "b": b}


def split_angle(angle: float) -> list[float]:
    count = max(1, int(math.ceil(abs(angle) / MAX_ELEMENTARY_ANGLE)))
    return [angle / count] * count


def steps_from_rotation(matrix: np.ndarray) -> list[dict[str, int]]:
    rotation = Rotation.from_matrix(np.asarray(matrix, dtype=float))
    # XYZ factors act in order X, Y, Z on column vectors; the exact finite map
    # records its own ordered lifting sequence and needs no floating decoder.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        angles = rotation.as_euler("xyz", degrees=False)
    steps: list[dict[str, int]] = []
    for angle, pair in zip(angles, ((1, 2), (0, 2), (0, 1))):
        for part in split_angle(float(angle)):
            steps.append(angle_step(part, *pair))
    return steps


def apply_steps(values: np.ndarray, steps: Iterable[dict[str, int]], inverse: bool = False) -> np.ndarray:
    out = np.asarray(values, dtype=np.int64).copy()
    ordered = list(steps)
    if inverse:
        ordered = list(reversed(ordered))
    for step in ordered:
        _lift_pair(out, int(step["i"]), int(step["j"]), int(step["a"]), int(step["b"]), inverse)
    if np.any(out < np.iinfo(np.int32).min) or np.any(out > np.iinfo(np.int32).max):
        raise OverflowError("finite transform exceeds signed int32 range")
    return out


def center_image(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.int64) - 128


def restore_image(values: np.ndarray) -> np.ndarray:
    restored = np.asarray(values, dtype=np.int64) + 128
    if np.any(restored < 0) or np.any(restored > 255):
        # Tampered unauthenticated objects remain parseable but clip at the
        # image boundary; valid objects are required to stay exact.
        restored = np.clip(restored, 0, 255)
    return restored.astype(np.uint8)
