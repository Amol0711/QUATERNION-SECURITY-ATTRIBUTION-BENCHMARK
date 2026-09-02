from __future__ import annotations

import math

import numpy as np


def byte_entropy(payload: bytes) -> float:
    if not payload:
        return 0.0
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256).astype(float)
    probabilities = counts[counts > 0] / len(payload)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def adjacent_byte_correlation(payload: bytes) -> float:
    if len(payload) < 3:
        return 0.0
    values = np.frombuffer(payload, dtype=np.uint8).astype(float)
    if np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def object_metrics(ciphertext_view: bytes, object_bytes: bytes, raw_bytes: int) -> dict[str, float | int]:
    return {
        "ciphertext_view_bytes": len(ciphertext_view),
        "object_bytes": len(object_bytes),
        "raw_image_bytes": int(raw_bytes),
        "expansion_ratio": float(len(object_bytes) / raw_bytes),
        "ciphertext_entropy": byte_entropy(ciphertext_view),
        "adjacent_byte_correlation": adjacent_byte_correlation(ciphertext_view),
        "ciphertext_mean": float(np.mean(np.frombuffer(ciphertext_view, dtype=np.uint8))) if ciphertext_view else 0.0,
    }


def exact_psnr(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    if np.array_equal(reference, reconstruction):
        return math.inf
    error = np.mean((reference.astype(float) - reconstruction.astype(float)) ** 2)
    return float(10.0 * np.log10((255.0 ** 2) / error))
