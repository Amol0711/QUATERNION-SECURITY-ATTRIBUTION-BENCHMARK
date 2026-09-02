from __future__ import annotations

import ctypes
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from .components import diffuse_bytes, invert_diffuse_bytes
from .models import RunContext
from .utils import derive_key


def _round_function(right: bytes, key: bytes, round_index: int, length: int) -> bytes:
    h = hashlib.shake_256()
    h.update(b"QSA-QUATERNION-FEISTEL-V1")
    h.update(round_index.to_bytes(2, "big"))
    h.update(key)
    h.update(right)
    return h.digest(length)


def quaternion_feistel_encrypt_reference(payload: bytes, context: RunContext, rounds: int = 6) -> tuple[bytes, int]:
    """Normative historical Python definition of B18."""
    original = len(payload)
    pad = (-len(payload)) % 16
    data = payload + bytes(pad)
    key = derive_key(context.master_key, "B18|QFEISTEL", 32)
    output = bytearray()
    for offset in range(0, len(data), 16):
        left = data[offset:offset + 8]
        right = data[offset + 8:offset + 16]
        for r in range(rounds):
            f = _round_function(right, key, r, 8)
            left, right = right, bytes(a ^ b for a, b in zip(left, f, strict=True))
        output.extend(left + right)
    return bytes(output), original


def quaternion_feistel_decrypt_reference(payload: bytes, original: int, context: RunContext, rounds: int = 6) -> bytes:
    if len(payload) % 16:
        raise ValueError("Feistel payload is not block aligned")
    key = derive_key(context.master_key, "B18|QFEISTEL", 32)
    output = bytearray()
    for offset in range(0, len(payload), 16):
        left = payload[offset:offset + 8]
        right = payload[offset + 8:offset + 16]
        for r in reversed(range(rounds)):
            previous_right = left
            f = _round_function(previous_right, key, r, 8)
            previous_left = bytes(a ^ b for a, b in zip(right, f, strict=True))
            left, right = previous_left, previous_right
        output.extend(left + right)
    if original < 0 or original > len(output):
        raise ValueError("invalid original length")
    return bytes(output[:original])


def _load_qfeistel_fast() -> ctypes.CDLL | None:
    path = Path(__file__).with_name("_qfeistel_fast.so")
    if not path.exists():
        return None
    try:
        library = ctypes.CDLL(str(path))
        byte_pointer = ctypes.POINTER(ctypes.c_ubyte)
        for name in ("qfeistel_encrypt_blocks", "qfeistel_decrypt_blocks"):
            function = getattr(library, name)
            function.argtypes = [
                byte_pointer, ctypes.c_size_t, byte_pointer, ctypes.c_int, byte_pointer,
            ]
            function.restype = ctypes.c_int
        return library
    except (OSError, AttributeError):
        return None


_QFEISTEL_FAST = _load_qfeistel_fast()


def _qfeistel_transform(data: bytes, key: bytes, rounds: int, decrypt: bool) -> bytes:
    if not data:
        return b""
    if len(data) % 16:
        raise ValueError("Feistel payload is not block aligned")
    if len(key) != 32:
        raise ValueError("Feistel key must be 32 bytes")
    if _QFEISTEL_FAST is None:
        raise RuntimeError("compiled quaternion-Feistel accelerator unavailable")
    input_buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    key_buffer = (ctypes.c_ubyte * len(key)).from_buffer_copy(key)
    output_buffer = (ctypes.c_ubyte * len(data))()
    function = (
        _QFEISTEL_FAST.qfeistel_decrypt_blocks
        if decrypt else _QFEISTEL_FAST.qfeistel_encrypt_blocks
    )
    status = function(input_buffer, len(data), key_buffer, int(rounds), output_buffer)
    if status != 0:
        raise RuntimeError(f"quaternion-Feistel accelerator failed with status {status}")
    return bytes(output_buffer)


def quaternion_feistel_encrypt(payload: bytes, context: RunContext, rounds: int = 6) -> tuple[bytes, int]:
    original = len(payload)
    pad = (-original) % 16
    data = payload + bytes(pad)
    if _QFEISTEL_FAST is None:
        return quaternion_feistel_encrypt_reference(payload, context, rounds)
    key = derive_key(context.master_key, "B18|QFEISTEL", 32)
    return _qfeistel_transform(data, key, rounds, decrypt=False), original


def quaternion_feistel_decrypt(payload: bytes, original: int, context: RunContext, rounds: int = 6) -> bytes:
    if len(payload) % 16:
        raise ValueError("Feistel payload is not block aligned")
    if original < 0 or original > len(payload):
        raise ValueError("invalid original length")
    if _QFEISTEL_FAST is None:
        return quaternion_feistel_decrypt_reference(payload, original, context, rounds)
    key = derive_key(context.master_key, "B18|QFEISTEL", 32)
    return _qfeistel_transform(payload, key, rounds, decrypt=True)[:original]


@lru_cache(maxsize=8)
def _cat_map_indices(size: int, rounds: int = 3) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    xx, yy = x.copy(), y.copy()
    for _ in range(rounds):
        xx, yy = (xx + yy) % size, (xx + 2 * yy) % size
    indices = (yy * size + xx).reshape(-1)
    indices.flags.writeable = False
    return indices


def chaos_pd_encrypt(image_bytes: bytes, shape: tuple[int, int, int], context: RunContext) -> bytes:
    height, width, channels = shape
    if height != width or channels != 3:
        raise ValueError("B19 requires a square RGB image")
    array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height * width, 3)
    indices = _cat_map_indices(height)
    permuted = array[indices].reshape(-1).tobytes()
    return diffuse_bytes(permuted, context, "B19|DIFF")


def chaos_pd_decrypt(payload: bytes, shape: tuple[int, int, int], context: RunContext) -> bytes:
    height, width, channels = shape
    if height != width or channels != 3:
        raise ValueError("B19 requires a square RGB image")
    diffused = invert_diffuse_bytes(payload, context, "B19|DIFF")
    array = np.frombuffer(diffused, dtype=np.uint8).reshape(height * width, 3)
    indices = _cat_map_indices(height)
    inverse = np.empty_like(indices)
    inverse[indices] = np.arange(len(indices))
    return array[inverse].reshape(-1).tobytes()
