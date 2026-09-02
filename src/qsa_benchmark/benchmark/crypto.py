from __future__ import annotations

import hashlib
import hmac
from functools import lru_cache

import numpy as np

from cryptography.hazmat.primitives.ciphers.aead import AESGCM, AESGCMSIV, ChaCha20Poly1305

from .models import RunContext
from .utils import derive_key


def _check_nonce(nonce: bytes, expected: int) -> None:
    if len(nonce) != expected:
        raise ValueError(f"nonce must be exactly {expected} bytes")


def aes_gcm_encrypt(payload: bytes, aad: bytes, context: RunContext, label: str) -> tuple[bytes, bytes]:
    _check_nonce(context.nonce, 12)
    key = derive_key(context.master_key, f"{label}|AES-GCM", 32)
    combined = AESGCM(key).encrypt(context.nonce, payload, aad)
    return combined[:-16], combined[-16:]


def aes_gcm_decrypt(ciphertext: bytes, tag: bytes, aad: bytes, context: RunContext, label: str) -> bytes:
    _check_nonce(context.nonce, 12)
    if len(tag) != 16:
        raise ValueError("AES-GCM tag must be 16 bytes")
    key = derive_key(context.master_key, f"{label}|AES-GCM", 32)
    return AESGCM(key).decrypt(context.nonce, ciphertext + tag, aad)



def aes_gcm_siv_encrypt(payload: bytes, aad: bytes, context: RunContext, label: str) -> tuple[bytes, bytes]:
    """AES-256-GCM-SIV with a separated 16-byte tag."""
    _check_nonce(context.nonce, 12)
    key = derive_key(context.master_key, f"{label}|AES-GCM-SIV", 32)
    combined = AESGCMSIV(key).encrypt(context.nonce, payload, aad)
    return combined[:-16], combined[-16:]


def aes_gcm_siv_decrypt(ciphertext: bytes, tag: bytes, aad: bytes, context: RunContext, label: str) -> bytes:
    _check_nonce(context.nonce, 12)
    if len(tag) != 16:
        raise ValueError("AES-GCM-SIV tag must be 16 bytes")
    key = derive_key(context.master_key, f"{label}|AES-GCM-SIV", 32)
    return AESGCMSIV(key).decrypt(context.nonce, ciphertext + tag, aad)

def chacha_encrypt(payload: bytes, aad: bytes, context: RunContext, label: str) -> tuple[bytes, bytes]:
    _check_nonce(context.nonce, 12)
    key = derive_key(context.master_key, f"{label}|CHACHA20-POLY1305", 32)
    combined = ChaCha20Poly1305(key).encrypt(context.nonce, payload, aad)
    return combined[:-16], combined[-16:]


def chacha_decrypt(ciphertext: bytes, tag: bytes, aad: bytes, context: RunContext, label: str) -> bytes:
    _check_nonce(context.nonce, 12)
    if len(tag) != 16:
        raise ValueError("ChaCha20-Poly1305 tag must be 16 bytes")
    key = derive_key(context.master_key, f"{label}|CHACHA20-POLY1305", 32)
    return ChaCha20Poly1305(key).decrypt(context.nonce, ciphertext + tag, aad)


@lru_cache(maxsize=16)
def _cached_shake_stream(length: int, label: str, nonce: bytes, method_id: str, image_id: str, master_key: bytes, public: bool) -> bytes:
    h = hashlib.shake_256()
    h.update(b"QSA-BENCHMARK-SHAKE-V1")
    h.update(len(label).to_bytes(4, "big"))
    h.update(label.encode("utf-8"))
    h.update(nonce)
    if public:
        h.update(method_id.encode("utf-8"))
        h.update(image_id.encode("utf-8"))
    else:
        h.update(derive_key(master_key, f"{label}|SHAKE", 32))
    return h.digest(length)


def shake_stream(length: int, context: RunContext, label: str, public: bool = False) -> bytes:
    return _cached_shake_stream(
        int(length), label, bytes(context.nonce), context.method_id,
        context.image_id if public else "", bytes(context.master_key), bool(public),
    )


def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("xor length mismatch")
    if not left:
        return b""
    a = np.frombuffer(left, dtype=np.uint8)
    b = np.frombuffer(right, dtype=np.uint8)
    return np.bitwise_xor(a, b).tobytes()


def shake_hmac_encrypt(payload: bytes, aad: bytes, context: RunContext, label: str) -> tuple[bytes, bytes]:
    _check_nonce(context.nonce, 16)
    ciphertext = xor_bytes(payload, shake_stream(len(payload), context, label, public=False))
    key = derive_key(context.master_key, f"{label}|HMAC-SHA256", 32)
    tag = hmac.new(key, aad + ciphertext, hashlib.sha256).digest()
    return ciphertext, tag


def shake_hmac_decrypt(ciphertext: bytes, tag: bytes, aad: bytes, context: RunContext, label: str) -> bytes:
    _check_nonce(context.nonce, 16)
    if len(tag) != 32:
        raise ValueError("HMAC-SHA256 tag must be 32 bytes")
    key = derive_key(context.master_key, f"{label}|HMAC-SHA256", 32)
    expected = hmac.new(key, aad + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, tag):
        raise ValueError("authentication failed")
    return xor_bytes(ciphertext, shake_stream(len(ciphertext), context, label, public=False))
