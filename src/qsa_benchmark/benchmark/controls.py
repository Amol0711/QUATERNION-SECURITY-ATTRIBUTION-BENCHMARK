from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from .constructions import _check_method, _parts
from .crypto import aes_gcm_decrypt, aes_gcm_encrypt, aes_gcm_siv_decrypt, aes_gcm_siv_encrypt
from .envelope import associated_data, associated_data_from_header, encode_envelope, parse_envelope
from .models import ConstructionOutput, RunContext
from .plugins import ConstructionPlugin
from .serialization import bytes_to_image, image_to_bytes

PUBLIC_PRP_ROUNDS = 10
FIXED_BODY_PREFIX_LENGTH = 32
FIXED_BODY_PREFIX = bytes(FIXED_BODY_PREFIX_LENGTH)


def _xor(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("xor length mismatch")
    if not left:
        return b""
    return np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8),
        np.frombuffer(right, dtype=np.uint8),
    ).tobytes()


def _validation_public_material(context: RunContext, label: bytes, length: int) -> bytes:
    h = hashlib.shake_256()
    h.update(b"QSA-PUBLIC-MATERIAL-V1")
    h.update(len(label).to_bytes(2, "big")); h.update(label)
    h.update(len(context.nonce).to_bytes(2, "big")); h.update(context.nonce)
    h.update(context.seed.to_bytes(16, "big", signed=False))
    h.update(context.method_id.encode("ascii"))
    return h.digest(length)


def _public_material(context: RunContext, label: bytes, length: int) -> bytes:
    if context.public_material:
        if len(context.public_material) != length:
            raise ValueError(f"public material must be exactly {length} bytes")
        return bytes(context.public_material)
    return _validation_public_material(context, label, length)


def add_mod256(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("modular-add length mismatch")
    a = np.frombuffer(left, dtype=np.uint8).astype(np.uint16)
    b = np.frombuffer(right, dtype=np.uint8).astype(np.uint16)
    return ((a + b) & 0xFF).astype(np.uint8).tobytes()


def sub_mod256(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("modular-subtract length mismatch")
    a = np.frombuffer(left, dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(right, dtype=np.uint8).astype(np.int16)
    return ((a - b) & 0xFF).astype(np.uint8).tobytes()


def _round_function(key: bytes, round_index: int, right: bytes, output_length: int, total_length: int) -> bytes:
    h = hashlib.shake_256()
    h.update(b"QSA-PUBLIC-WIDEBLOCK-FEISTEL-V1")
    h.update(key); h.update(round_index.to_bytes(2, "big"))
    h.update(total_length.to_bytes(8, "big")); h.update(len(right).to_bytes(8, "big")); h.update(right)
    return h.digest(output_length)


def public_wideblock_prp(payload: bytes, key: bytes, rounds: int = PUBLIC_PRP_ROUNDS) -> bytes:
    if len(key) != 32:
        raise ValueError("public PRP key must be 32 bytes")
    if rounds <= 0 or rounds % 2:
        raise ValueError("round count must be positive and even")
    if len(payload) < 2:
        return payload
    split = len(payload) // 2
    left, right = payload[:split], payload[split:]
    for index in range(rounds):
        left, right = right, _xor(left, _round_function(key, index, right, len(left), len(payload)))
    return left + right


def invert_public_wideblock_prp(payload: bytes, key: bytes, rounds: int = PUBLIC_PRP_ROUNDS) -> bytes:
    if len(key) != 32:
        raise ValueError("public PRP key must be 32 bytes")
    if rounds <= 0 or rounds % 2:
        raise ValueError("round count must be positive and even")
    if len(payload) < 2:
        return payload
    split = len(payload) // 2
    left, right = payload[:split], payload[split:]
    for index in reversed(range(rounds)):
        previous_right = left
        previous_left = _xor(right, _round_function(key, index, previous_right, len(right), len(payload)))
        left, right = previous_left, previous_right
    return left + right


class PublicHighEntropyConstruction(ConstructionPlugin):
    method_id = "B04_public_high_entropy"
    display_name = "Public deterministic high-entropy mask control (corrected control)"
    family = "negative-control"
    benchmark_role = "deliberately insecure metric false-positive control"
    authenticated = False
    secure_control = False
    exact = True
    component_ids = ("public-shake256-mask-v2",)

    @staticmethod
    def _mask(length: int, nonce: bytes) -> bytes:
        h = hashlib.shake_256()
        h.update(b"QSA-B04-PUBLIC-MASK-V1")
        h.update(len(nonce).to_bytes(2, "big")); h.update(nonce)
        h.update(PublicHighEntropyConstruction.method_id.encode("ascii"))
        return h.digest(length)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        payload = image_to_bytes(image); ciphertext = _xor(payload, self._mask(len(payload), context.nonce))
        descriptor = {"construction": "public-shake256-v2", "mask_inputs": ["serialized_nonce", "method_id"], "publicly_recoverable": True}
        parts = _parts(self.method_id, tuple(image.shape), descriptor, context, protected=ciphertext)
        return ConstructionOutput(encode_envelope(parts), ciphertext)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        payload = _xor(parsed.protected_payload, self._mask(len(parsed.protected_payload), parsed.nonce))
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))


class PublicFreshPadConstruction(ConstructionPlugin):
    method_id = "B21_public_fresh_pad"
    display_name = "Published fresh additive pad control"
    family = "adversarial-control"
    benchmark_role = "exact ideal-appearance/confidentiality separation"
    authenticated = False
    secure_control = False
    exact = True
    component_ids = ("published-additive-pad",)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        payload = image_to_bytes(image); pad = _public_material(context, b"PUBLIC-PAD", len(payload))
        body = add_mod256(payload, pad)
        descriptor = {"construction": "published-additive-pad", "arithmetic": "Z_256-coordinatewise", "pad_length": len(pad), "publicly_recoverable": True}
        parts = _parts(self.method_id, tuple(image.shape), descriptor, context, public=pad, protected=body)
        return ConstructionOutput(encode_envelope(parts), body)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        if len(parsed.public_payload) != len(parsed.protected_payload):
            raise ValueError("public-pad length mismatch")
        payload = sub_mod256(parsed.protected_payload, parsed.public_payload)
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))


class PublicWideBlockPRPConstruction(ConstructionPlugin):
    method_id = "B22_public_wideblock_prp"
    display_name = "Published-key wide-block Feistel PRP control"
    family = "adversarial-control"
    benchmark_role = "same-context ideal-like appearance with public inversion"
    authenticated = False
    secure_control = False
    exact = True
    component_ids = ("published-key-wideblock-feistel-prp",)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        payload = image_to_bytes(image); key = _public_material(context, b"PUBLIC-PRP-KEY", 32)
        body = public_wideblock_prp(payload, key)
        descriptor = {"construction": "published-key-wideblock-feistel", "rounds": PUBLIC_PRP_ROUNDS, "key_length": len(key), "publicly_recoverable": True, "formal_status": "engineering instantiation; not a random-permutation proof"}
        parts = _parts(self.method_id, tuple(image.shape), descriptor, context, public=key, protected=body)
        return ConstructionOutput(encode_envelope(parts), body)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        if len(parsed.public_payload) != 32 or int(parsed.header["descriptor"].get("rounds", -1)) != PUBLIC_PRP_ROUNDS:
            raise ValueError("invalid public wide-block descriptor")
        payload = invert_public_wideblock_prp(parsed.protected_payload, parsed.public_payload)
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))


class SecureFixedHeaderConstruction(ConstructionPlugin):
    method_id = "B23_secure_fixed_header"
    display_name = "AES-256-GCM with fixed protected-body header"
    family = "adversarial-control"
    benchmark_role = "secure-but-nonuniform protected-body control"
    authenticated = True
    secure_control = True
    exact = True
    component_ids = ("fixed-protected-header", "aes-gcm-256")

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        payload = image_to_bytes(image)
        descriptor = {"construction": "aes-gcm-fixed-protected-header", "payload": "raw-rgb-u8", "fixed_prefix_length": FIXED_BODY_PREFIX_LENGTH}
        provisional = _parts(self.method_id, tuple(image.shape), descriptor, context, protected=bytes(FIXED_BODY_PREFIX_LENGTH + len(payload)), tag=bytes(16))
        aad = associated_data(provisional) + FIXED_BODY_PREFIX
        ciphertext, tag = aes_gcm_encrypt(payload, aad, context, self.method_id)
        body = FIXED_BODY_PREFIX + ciphertext
        final = _parts(self.method_id, tuple(image.shape), descriptor, context, protected=body, tag=tag)
        return ConstructionOutput(encode_envelope(final), body)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        prefix_length = int(parsed.header["descriptor"].get("fixed_prefix_length", -1))
        if prefix_length != FIXED_BODY_PREFIX_LENGTH or len(parsed.protected_payload) < prefix_length:
            raise ValueError("invalid fixed protected-body header")
        prefix, ciphertext = parsed.protected_payload[:prefix_length], parsed.protected_payload[prefix_length:]
        if prefix != FIXED_BODY_PREFIX:
            raise ValueError("fixed protected-body header check failed")
        local = replace(context, nonce=parsed.nonce)
        aad = associated_data_from_header(parsed.header_bytes, parsed.nonce, parsed.public_payload) + prefix
        payload = aes_gcm_decrypt(ciphertext, parsed.tag, aad, local, self.method_id)
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))


class AESGCMSIVConstruction(ConstructionPlugin):
    method_id = "B24_aes_gcm_siv"
    display_name = "Full-image AES-256-GCM-SIV"
    family = "standard-cryptography"
    benchmark_role = "nonce-misuse-resistant authenticated-encryption control"
    authenticated = True
    secure_control = True
    exact = True
    component_ids = ("aes-gcm-siv-256",)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        payload = image_to_bytes(image)
        descriptor = {"construction": "aes-gcm-siv-256", "payload": "raw-rgb-u8", "nonce_policy": "unique recommended; forced reuse studied separately"}
        provisional = _parts(self.method_id, tuple(image.shape), descriptor, context, protected=bytes(len(payload)), tag=bytes(16))
        aad = associated_data(provisional)
        ciphertext, tag = aes_gcm_siv_encrypt(payload, aad, context, self.method_id)
        final = _parts(self.method_id, tuple(image.shape), descriptor, context, protected=ciphertext, tag=tag)
        return ConstructionOutput(encode_envelope(final), ciphertext)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        local = replace(context, nonce=parsed.nonce)
        aad = associated_data_from_header(parsed.header_bytes, parsed.nonce, parsed.public_payload)
        payload = aes_gcm_siv_decrypt(parsed.protected_payload, parsed.tag, aad, local, self.method_id)
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))
