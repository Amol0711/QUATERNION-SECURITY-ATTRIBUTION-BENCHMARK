from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import numpy as np

try:  # Optional exact execution accelerator; the scalar definition remains normative.
    from numba import njit
except Exception:  # pragma: no cover - exercised only in minimal environments
    njit = None

from .crypto import shake_stream, xor_bytes
from .models import RunContext
from .utils import derive_key


def public_shake_mask(payload: bytes, context: RunContext, label: str = "PUBLIC") -> bytes:
    return xor_bytes(payload, shake_stream(len(payload), context, label, public=True))


def keyed_shake_mask(payload: bytes, context: RunContext, label: str = "MASK") -> bytes:
    return xor_bytes(payload, shake_stream(len(payload), context, label, public=False))


def _chebyshev_mask_reference(length: int, seed_u64: int) -> np.ndarray:
    """Normative scalar byte sequence used by the legacy Chebyshev ablation.

    The function returns only the mask so that the optimized path can be checked
    byte-for-byte without conflating sequence generation and XOR execution.
    """
    if length < 0:
        raise ValueError("negative length")
    x = (int(seed_u64) + 1) / (2**64 + 2)
    out = np.empty(length, dtype=np.uint8)
    lower = 2**-53
    upper = 1.0 - 2**-53
    for index in range(length):
        y = 2.0 * x - 1.0
        y = 4.0 * y * y * y - 3.0 * y
        x = min(max((y + 1.0) / 2.0, lower), upper)
        out[index] = int(math.floor(x * 256.0)) & 0xFF
    return out


if njit is not None:
    @njit(cache=True, fastmath=False)
    def _chebyshev_mask_compiled(length: int, x0: float) -> np.ndarray:  # pragma: no cover - covered through wrapper
        out = np.empty(length, dtype=np.uint8)
        x = x0
        lower = 2.0 ** -53
        upper = 1.0 - lower
        for index in range(length):
            y = 2.0 * x - 1.0
            y = 4.0 * y * y * y - 3.0 * y
            x = (y + 1.0) / 2.0
            if x < lower:
                x = lower
            elif x > upper:
                x = upper
            out[index] = int(math.floor(x * 256.0)) & 0xFF
        return out
else:
    _chebyshev_mask_compiled = None


@lru_cache(maxsize=8)
def _cached_chebyshev_mask(length: int, seed_u64: int) -> bytes:
    x0 = (int(seed_u64) + 1) / (2**64 + 2)
    if _chebyshev_mask_compiled is None:
        mask = _chebyshev_mask_reference(length, seed_u64)
    else:
        mask = _chebyshev_mask_compiled(length, x0)
    return mask.tobytes()


def chebyshev_mask(payload: bytes, context: RunContext) -> bytes:
    # Deterministic public legacy-style sequence; it is intentionally not a cryptographic PRG.
    seed_material = hashlib.sha256(b"QSA-CHEBYSHEV-V1" + context.nonce + context.method_id.encode()).digest()
    seed_u64 = int.from_bytes(seed_material[:8], "big")
    mask = _cached_chebyshev_mask(len(payload), seed_u64)
    return xor_bytes(payload, mask)


def _rng_seed(context: RunContext, label: str, public: bool = False) -> int:
    h = hashlib.sha256()
    h.update(b"QSA-RNG-SEED-V1")
    h.update(label.encode())
    h.update(context.nonce)
    if public:
        h.update(context.method_id.encode())
    else:
        h.update(derive_key(context.master_key, label, 32))
    return int.from_bytes(h.digest()[:8], "big")


@lru_cache(maxsize=4)
def _cached_permutation_indices(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(length)
    indices.flags.writeable = False
    return indices


def permutation_indices(length: int, context: RunContext, label: str = "PERM", legacy: bool = False) -> np.ndarray:
    if length < 0:
        raise ValueError("negative length")
    seed = _rng_seed(context, label)
    # The benchmark distinguishes the legacy NumPy-seeded path by label and metadata;
    # both are deterministic exact permutations, not cryptographic assumptions.
    return _cached_permutation_indices(length, seed)


def permute_bytes(payload: bytes, context: RunContext, label: str = "PERM", legacy: bool = False) -> bytes:
    if not payload:
        return payload
    array = np.frombuffer(payload, dtype=np.uint8)
    indices = permutation_indices(len(array), context, label, legacy)
    return array[indices].tobytes()


def invert_permute_bytes(payload: bytes, context: RunContext, label: str = "PERM", legacy: bool = False) -> bytes:
    if not payload:
        return payload
    array = np.frombuffer(payload, dtype=np.uint8)
    indices = permutation_indices(len(array), context, label, legacy)
    inverse = np.empty_like(indices)
    inverse[indices] = np.arange(len(indices))
    return array[inverse].tobytes()


@lru_cache(maxsize=4)
def _cached_diffusion_keystream(length: int, nonce: bytes, master_key: bytes, method_id: str, label: str) -> np.ndarray:
    context = RunContext(
        master_key=master_key,
        nonce=nonce,
        seed=0,
        image_id="cache-neutral",
        method_id=method_id,
        run_id="cache-neutral",
    )
    key = np.frombuffer(shake_stream(length, context, label, public=False), dtype=np.uint8).astype(np.int64)
    key.flags.writeable = False
    return key


def diffusion_keystream(length: int, context: RunContext, label: str = "DIFF") -> np.ndarray:
    return _cached_diffusion_keystream(length, bytes(context.nonce), bytes(context.master_key), context.method_id, label)


def _diffuse_bytes_reference(payload: bytes, context: RunContext, label: str = "DIFF") -> bytes:
    if not payload:
        return payload
    source = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
    key = diffusion_keystream(len(source), context, label)
    forward = np.empty_like(source)
    previous = int(key[0])
    for index, value in enumerate(source):
        forward[index] = (int(value) + previous + int(key[index])) & 0xFF
        previous = int(forward[index])
    backward = forward.copy()
    following = int(key[-1])
    for index in range(len(backward) - 1, -1, -1):
        backward[index] = (int(backward[index]) + following + int(key[index])) & 0xFF
        following = int(backward[index])
    return backward.astype(np.uint8).tobytes()


def _invert_diffuse_bytes_reference(payload: bytes, context: RunContext, label: str = "DIFF") -> bytes:
    if not payload:
        return payload
    backward = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
    key = diffusion_keystream(len(backward), context, label)
    forward = np.empty_like(backward)
    following = int(key[-1])
    for index in range(len(backward) - 1, -1, -1):
        forward[index] = (int(backward[index]) - following - int(key[index])) & 0xFF
        following = int(backward[index])
    source = np.empty_like(forward)
    previous = int(key[0])
    for index, value in enumerate(forward):
        source[index] = (int(value) - previous - int(key[index])) & 0xFF
        previous = int(value)
    return source.astype(np.uint8).tobytes()


def diffuse_bytes(payload: bytes, context: RunContext, label: str = "DIFF") -> bytes:
    """Exact vectorization of the historical two-pass modular recurrence."""
    if not payload:
        return payload
    source = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
    key = diffusion_keystream(len(source), context, label)
    forward = (np.cumsum(source + key, dtype=np.int64) + int(key[0])) & 0xFF
    backward = (np.cumsum((forward + key)[::-1], dtype=np.int64)[::-1] + int(key[-1])) & 0xFF
    return backward.astype(np.uint8).tobytes()


def invert_diffuse_bytes(payload: bytes, context: RunContext, label: str = "DIFF") -> bytes:
    """Exact vectorized inverse of :func:`diffuse_bytes`."""
    if not payload:
        return payload
    backward = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
    key = diffusion_keystream(len(backward), context, label)
    forward = np.empty_like(backward)
    if len(backward) == 1:
        forward[0] = (backward[0] - 2 * key[0]) & 0xFF
    else:
        forward[:-1] = (backward[:-1] - backward[1:] - key[:-1]) & 0xFF
        forward[-1] = (backward[-1] - 2 * key[-1]) & 0xFF
    source = np.empty_like(forward)
    source[0] = (forward[0] - 2 * key[0]) & 0xFF
    if len(forward) > 1:
        source[1:] = (forward[1:] - forward[:-1] - key[1:]) & 0xFF
    return source.astype(np.uint8).tobytes()
