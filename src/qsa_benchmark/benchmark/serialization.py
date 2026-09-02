from __future__ import annotations

import numpy as np


def image_to_bytes(image: np.ndarray) -> bytes:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8 image")
    return array.tobytes(order="C")


def bytes_to_image(payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    expected = int(np.prod(shape))
    if len(payload) != expected:
        raise ValueError("raw image payload length mismatch")
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()


def int32_to_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype="<i4")
    return array.tobytes(order="C")


def bytes_to_int32(payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    expected = int(np.prod(shape)) * 4
    if len(payload) != expected:
        raise ValueError("int32 payload length mismatch")
    return np.frombuffer(payload, dtype="<i4").reshape(shape).astype(np.int64)
