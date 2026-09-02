from __future__ import annotations

import hashlib
import hmac
import io
from dataclasses import replace
from typing import Any, Callable

import numpy as np
from PIL import Image

from .components import (
    chebyshev_mask,
    diffuse_bytes,
    invert_diffuse_bytes,
    invert_permute_bytes,
    keyed_shake_mask,
    permute_bytes,
    public_shake_mask,
)
from .crypto import (
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    chacha_decrypt,
    chacha_encrypt,
    shake_hmac_decrypt,
    shake_hmac_encrypt,
)
from .envelope import associated_data, associated_data_from_header, encode_envelope, parse_envelope
from .external import chaos_pd_decrypt, chaos_pd_encrypt, quaternion_feistel_decrypt, quaternion_feistel_encrypt
from .models import ConstructionOutput, EnvelopeParts, RunContext, TransformOutput
from .plugins import ConstructionPlugin, TransformPlugin
from .serialization import bytes_to_image, image_to_bytes
from .transforms import transform_from_descriptor
from .utils import derive_key


def _ctx(context: RunContext, nonce: bytes) -> RunContext:
    return replace(context, nonce=nonce)


def _metadata(context: RunContext) -> dict[str, Any]:
    if context.public_metadata is None:
        return {"image_id": context.image_id, "run_id": context.run_id, "benchmark": "QSA-BENCHMARK-V1"}
    metadata = dict(context.public_metadata)
    forbidden = {"image_id", "run_id", "pair_id", "key_index", "state_index", "perturbation_id"}
    overlap = forbidden.intersection(metadata)
    if overlap:
        raise ValueError(f"experiment identifiers cannot be serialized in the sanitized security view: {sorted(overlap)}")
    return metadata


def _check_method(parsed, method_id: str) -> None:
    if parsed.header["method_id"] != method_id:
        raise ValueError("method identifier mismatch")


def _parts(
    method_id: str,
    shape: tuple[int, int, int],
    descriptor: dict[str, Any],
    context: RunContext,
    public: bytes = b"",
    protected: bytes = b"",
    tag: bytes = b"",
) -> EnvelopeParts:
    return EnvelopeParts(method_id, shape, descriptor, _metadata(context), context.nonce, public, protected, tag)


class RawAEADConstruction(ConstructionPlugin):
    def __init__(self, method_id: str, display_name: str, primitive: str) -> None:
        self.method_id = method_id
        self.display_name = display_name
        self.family = "standard-cryptography"
        self.benchmark_role = "standard authenticated-encryption control"
        self.authenticated = True
        self.secure_control = True
        self.exact = True
        self.primitive = primitive
        self.component_ids = (primitive,)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        payload = image_to_bytes(image)
        descriptor = {"construction": self.primitive, "payload": "raw-rgb-u8"}
        tag_length = 16
        provisional = _parts(self.method_id, tuple(image.shape), descriptor, context, protected=bytes(len(payload)), tag=bytes(tag_length))
        aad = associated_data(provisional)
        if self.primitive == "aes-gcm-256":
            ciphertext, tag = aes_gcm_encrypt(payload, aad, context, self.method_id)
        else:
            ciphertext, tag = chacha_encrypt(payload, aad, context, self.method_id)
        final = _parts(self.method_id, tuple(image.shape), descriptor, context, protected=ciphertext, tag=tag)
        return ConstructionOutput(encode_envelope(final), ciphertext)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        local = _ctx(context, parsed.nonce)
        aad = associated_data_from_header(parsed.header_bytes, parsed.nonce, parsed.public_payload)
        if self.primitive == "aes-gcm-256":
            payload = aes_gcm_decrypt(parsed.protected_payload, parsed.tag, aad, local, self.method_id)
        else:
            payload = chacha_decrypt(parsed.protected_payload, parsed.tag, aad, local, self.method_id)
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))


class ShakeHMACConstruction(ConstructionPlugin):
    method_id = "B03_shake_hmac"
    display_name = "Keyed SHAKE-256 mask plus HMAC-SHA-256"
    family = "standard-cryptography"
    benchmark_role = "primitive-only stream-mask control"
    authenticated = True
    secure_control = True
    exact = True
    component_ids = ("keyed-shake256", "hmac-sha256")

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        payload = image_to_bytes(image)
        descriptor = {"construction": "shake256-hmac-sha256", "payload": "raw-rgb-u8"}
        provisional = _parts(self.method_id, tuple(image.shape), descriptor, context, protected=bytes(len(payload)), tag=bytes(32))
        aad = associated_data(provisional)
        ciphertext, tag = shake_hmac_encrypt(payload, aad, context, self.method_id)
        final = _parts(self.method_id, tuple(image.shape), descriptor, context, protected=ciphertext, tag=tag)
        return ConstructionOutput(encode_envelope(final), ciphertext)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        local = _ctx(context, parsed.nonce)
        aad = associated_data_from_header(parsed.header_bytes, parsed.nonce, parsed.public_payload)
        payload = shake_hmac_decrypt(parsed.protected_payload, parsed.tag, aad, local, self.method_id)
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))


class PublicHighEntropyConstruction(ConstructionPlugin):
    method_id = "B04_public_high_entropy"
    display_name = "Public deterministic high-entropy mask control"
    family = "negative-control"
    benchmark_role = "deliberately insecure metric false-positive control"
    authenticated = False
    secure_control = False
    exact = True
    component_ids = ("public-shake256-mask",)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        ciphertext = public_shake_mask(image_to_bytes(image), context, self.method_id)
        parts = _parts(self.method_id, tuple(image.shape), {"construction": "public-shake256", "publicly_recoverable": True}, context, protected=ciphertext)
        return ConstructionOutput(encode_envelope(parts), ciphertext)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        payload = public_shake_mask(parsed.protected_payload, _ctx(context, parsed.nonce), self.method_id)
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))


class PublicTransformConstruction(ConstructionPlugin):
    def __init__(self, method_id: str, display_name: str, family: str, role: str, transform: TransformPlugin) -> None:
        self.method_id = method_id
        self.display_name = display_name
        self.family = family
        self.benchmark_role = role
        self.authenticated = False
        self.secure_control = False
        self.exact = True
        self.transform = transform
        self.component_ids = (transform.component_id,)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        transformed = self.transform.forward(image, context)
        descriptor = {"construction": "public-transform", "transform": transformed.descriptor}
        parts = _parts(self.method_id, transformed.shape, descriptor, context, public=transformed.payload)
        return ConstructionOutput(encode_envelope(parts), transformed.payload)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        descriptor = parsed.header["descriptor"]["transform"]
        transform = transform_from_descriptor(descriptor)
        output = TransformOutput(parsed.public_payload, descriptor, tuple(parsed.header["image_shape"]))
        return transform.inverse(output, _ctx(context, parsed.nonce))


class LegacyAblationConstruction(ConstructionPlugin):
    def __init__(self, method_id: str, display_name: str, mode: str) -> None:
        self.method_id = method_id
        self.display_name = display_name
        self.family = "legacy-ablation"
        self.benchmark_role = f"{mode} ablation"
        self.authenticated = False
        self.secure_control = False
        self.exact = True
        self.mode = mode
        self.component_ids = (mode,)

    def _forward(self, payload: bytes, context: RunContext) -> bytes:
        if self.mode == "chebyshev-mask":
            return chebyshev_mask(payload, context)
        if self.mode == "permutation-only":
            return permute_bytes(payload, context, self.method_id)
        if self.mode == "diffusion-only":
            return diffuse_bytes(payload, context, self.method_id)
        raise ValueError(self.mode)

    def _inverse(self, payload: bytes, context: RunContext) -> bytes:
        if self.mode == "chebyshev-mask":
            return chebyshev_mask(payload, context)
        if self.mode == "permutation-only":
            return invert_permute_bytes(payload, context, self.method_id)
        if self.mode == "diffusion-only":
            return invert_diffuse_bytes(payload, context, self.method_id)
        raise ValueError(self.mode)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        ciphertext = self._forward(image_to_bytes(image), context)
        parts = _parts(self.method_id, tuple(image.shape), {"construction": self.mode}, context, protected=ciphertext)
        return ConstructionOutput(encode_envelope(parts), ciphertext)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        payload = self._inverse(parsed.protected_payload, _ctx(context, parsed.nonce))
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))


class TransformCryptoConstruction(ConstructionPlugin):
    def __init__(self, method_id: str, display_name: str, transform: TransformPlugin, primitive: str) -> None:
        self.method_id = method_id
        self.display_name = display_name
        self.family = "composition"
        self.benchmark_role = "transform-plus-standard-primitive attribution control"
        self.authenticated = True
        self.secure_control = True
        self.exact = True
        self.transform = transform
        self.primitive = primitive
        self.component_ids = (transform.component_id, primitive)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        transformed = self.transform.forward(image, context)
        descriptor = {"construction": self.primitive, "transform": transformed.descriptor, "payload": "signed-int32"}
        tag_len = 32 if self.primitive == "shake-hmac" else 16
        provisional = _parts(self.method_id, transformed.shape, descriptor, context, protected=bytes(len(transformed.payload)), tag=bytes(tag_len))
        aad = associated_data(provisional)
        if self.primitive == "shake-hmac":
            ciphertext, tag = shake_hmac_encrypt(transformed.payload, aad, context, self.method_id)
        else:
            ciphertext, tag = aes_gcm_encrypt(transformed.payload, aad, context, self.method_id)
        final = _parts(self.method_id, transformed.shape, descriptor, context, protected=ciphertext, tag=tag)
        return ConstructionOutput(encode_envelope(final), ciphertext)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        local = _ctx(context, parsed.nonce)
        aad = associated_data_from_header(parsed.header_bytes, parsed.nonce, parsed.public_payload)
        if self.primitive == "shake-hmac":
            payload = shake_hmac_decrypt(parsed.protected_payload, parsed.tag, aad, local, self.method_id)
        else:
            payload = aes_gcm_decrypt(parsed.protected_payload, parsed.tag, aad, local, self.method_id)
        descriptor = parsed.header["descriptor"]["transform"]
        transform = transform_from_descriptor(descriptor)
        return transform.inverse(TransformOutput(payload, descriptor, tuple(parsed.header["image_shape"])), local)


class TipR0Emulation(ConstructionPlugin):
    method_id = "B17_tip_r0_emulation"
    display_name = "TIP-R0 clean-room research emulation"
    family = "legacy-composition"
    benchmark_role = "legacy quaternion/sequence/permutation/diffusion composition"
    authenticated = True
    secure_control = False
    exact = True
    component_ids = ("caseii-curvature", "chebyshev", "shake", "legacy-permutation", "diffusion", "hmac")

    def __init__(self, transform: TransformPlugin) -> None:
        self.transform = transform

    def _forward(self, payload: bytes, context: RunContext) -> bytes:
        value = chebyshev_mask(payload, context)
        value = keyed_shake_mask(value, context, self.method_id)
        value = permute_bytes(value, context, self.method_id, legacy=True)
        return diffuse_bytes(value, context, self.method_id)

    def _inverse(self, payload: bytes, context: RunContext) -> bytes:
        value = invert_diffuse_bytes(payload, context, self.method_id)
        value = invert_permute_bytes(value, context, self.method_id, legacy=True)
        value = keyed_shake_mask(value, context, self.method_id)
        return chebyshev_mask(value, context)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        transformed = self.transform.forward(image, context)
        ciphertext = self._forward(transformed.payload, context)
        descriptor = {"construction": "tip-r0-clean-room", "transform": transformed.descriptor}
        provisional = _parts(self.method_id, transformed.shape, descriptor, context, protected=ciphertext, tag=bytes(32))
        aad = associated_data(provisional)
        key = derive_key(context.master_key, self.method_id + "|HMAC", 32)
        tag = hmac.new(key, aad + ciphertext, hashlib.sha256).digest()
        final = _parts(self.method_id, transformed.shape, descriptor, context, protected=ciphertext, tag=tag)
        return ConstructionOutput(encode_envelope(final), ciphertext)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        local = _ctx(context, parsed.nonce)
        aad = associated_data_from_header(parsed.header_bytes, parsed.nonce, parsed.public_payload)
        key = derive_key(local.master_key, self.method_id + "|HMAC", 32)
        expected = hmac.new(key, aad + parsed.protected_payload, hashlib.sha256).digest()
        if len(parsed.tag) != 32 or not hmac.compare_digest(expected, parsed.tag):
            raise ValueError("authentication failed")
        payload = self._inverse(parsed.protected_payload, local)
        descriptor = parsed.header["descriptor"]["transform"]
        return transform_from_descriptor(descriptor).inverse(
            TransformOutput(payload, descriptor, tuple(parsed.header["image_shape"])), local
        )


class ExternalQuaternionFeistel(ConstructionPlugin):
    method_id = "B18_external_quaternion_feistel"
    display_name = "External quaternion-Feistel clean-room emulation"
    family = "external-baseline"
    benchmark_role = "quaternion-Feistel structural reproduction"
    authenticated = False
    secure_control = False
    exact = True
    component_ids = ("quaternion-feistel-emulation",)

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        cipher, original = quaternion_feistel_encrypt(image_to_bytes(image), context)
        parts = _parts(self.method_id, tuple(image.shape), {"construction": "quaternion-feistel", "original_length": original}, context, protected=cipher)
        return ConstructionOutput(encode_envelope(parts), cipher)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        original = int(parsed.header["descriptor"]["original_length"])
        payload = quaternion_feistel_decrypt(parsed.protected_payload, original, _ctx(context, parsed.nonce))
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))


class ExternalChaosPD(ConstructionPlugin):
    method_id = "B19_external_chaos_pd"
    display_name = "External Fridrich-style chaos permutation-diffusion emulation"
    family = "external-baseline"
    benchmark_role = "chaotic-map permutation/diffusion structural reproduction"
    authenticated = False
    secure_control = False
    exact = True
    component_ids = ("cat-map-permutation", "two-pass-diffusion")

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        shape = tuple(image.shape)
        cipher = chaos_pd_encrypt(image_to_bytes(image), shape, context)
        parts = _parts(self.method_id, shape, {"construction": "fridrich-style-cat-map-diffusion"}, context, protected=cipher)
        return ConstructionOutput(encode_envelope(parts), cipher)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        shape = tuple(parsed.header["image_shape"])
        payload = chaos_pd_decrypt(parsed.protected_payload, shape, _ctx(context, parsed.nonce))
        return bytes_to_image(payload, shape)


class FullAEADExplicitPreview(ConstructionPlugin):
    method_id = "B20_full_aead_explicit_preview"
    display_name = "Full AES-GCM plus explicit public preview"
    family = "functionality-control"
    benchmark_role = "full-protection plus explicit-preview dominator"
    authenticated = True
    secure_control = True
    exact = True
    component_ids = ("explicit-preview", "aes-gcm-256")

    @staticmethod
    def _preview(image: np.ndarray) -> bytes:
        pil = Image.fromarray(image, mode="RGB")
        pil.thumbnail((24, 24), Image.Resampling.BILINEAR)
        buffer = io.BytesIO(); pil.save(buffer, format="PNG", optimize=False, compress_level=9)
        return buffer.getvalue()

    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput:
        payload = image_to_bytes(image)
        public = self._preview(image)
        descriptor = {"construction": "full-aead-explicit-preview", "preview_format": "PNG", "preview_max": [24, 24]}
        provisional = _parts(self.method_id, tuple(image.shape), descriptor, context, public=public, protected=bytes(len(payload)), tag=bytes(16))
        aad = associated_data(provisional)
        ciphertext, tag = aes_gcm_encrypt(payload, aad, context, self.method_id)
        final = _parts(self.method_id, tuple(image.shape), descriptor, context, public=public, protected=ciphertext, tag=tag)
        return ConstructionOutput(encode_envelope(final), ciphertext)

    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray:
        parsed = parse_envelope(object_bytes); _check_method(parsed, self.method_id)
        local = _ctx(context, parsed.nonce)
        aad = associated_data_from_header(parsed.header_bytes, parsed.nonce, parsed.public_payload)
        payload = aes_gcm_decrypt(parsed.protected_payload, parsed.tag, aad, local, self.method_id)
        return bytes_to_image(payload, tuple(parsed.header["image_shape"]))
