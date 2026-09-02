from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .models import EnvelopeParts, ParsedEnvelope
from .utils import canonical_json_bytes

MAGIC = b"QSB1"
VERSION = 1
PREFIX_LENGTH = 9
MAX_HEADER_LENGTH = 1 << 20
MAX_IMAGE_DIMENSION = 65535
MAX_IMAGE_PIXELS = 1 << 28
_ALLOWED_HEADER_KEYS = {
    "method_id",
    "image_shape",
    "descriptor",
    "metadata",
    "nonce_length",
    "public_length",
    "protected_length",
    "tag_length",
}


class EnvelopeFormatError(ValueError):
    """Canonical QSB1 parser rejection with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _reject(code: str, message: str) -> None:
    raise EnvelopeFormatError(code, message)


def _header(parts: EnvelopeParts) -> dict[str, Any]:
    return {
        "method_id": parts.method_id,
        "image_shape": list(parts.image_shape),
        "descriptor": parts.descriptor,
        "metadata": parts.metadata,
        "nonce_length": len(parts.nonce),
        "public_length": len(parts.public_payload),
        "protected_length": len(parts.protected_payload),
        "tag_length": len(parts.tag),
    }


def _validate_header_fields(
    header: Any,
) -> tuple[dict[str, Any], tuple[int, int, int], tuple[int, int, int, int]]:
    if not isinstance(header, dict) or set(header) != _ALLOWED_HEADER_KEYS:
        _reject("invalid_header_fields", "noncanonical header field set")

    method_id = header.get("method_id")
    if (
        type(method_id) is not str
        or not method_id
        or len(method_id) > 128
        or not method_id.isascii()
        or any(character.isspace() for character in method_id)
    ):
        _reject("invalid_method_id", "invalid method identifier")

    if not isinstance(header.get("descriptor"), dict):
        _reject("invalid_descriptor", "descriptor must be a JSON object")
    if not isinstance(header.get("metadata"), dict):
        _reject("invalid_metadata", "metadata must be a JSON object")

    shape_raw = header.get("image_shape")
    if not isinstance(shape_raw, list) or len(shape_raw) != 3:
        _reject("invalid_image_shape", "image shape must contain three dimensions")
    if any(
        type(value) is not int or value <= 0 or value > MAX_IMAGE_DIMENSION
        for value in shape_raw
    ):
        _reject("invalid_image_shape", "image dimensions must be positive bounded integers")
    if shape_raw[2] != 3 or shape_raw[0] * shape_raw[1] > MAX_IMAGE_PIXELS:
        _reject("invalid_image_shape", "invalid RGB image shape")
    shape = (int(shape_raw[0]), int(shape_raw[1]), int(shape_raw[2]))

    lengths: list[int] = []
    for name in ("nonce_length", "public_length", "protected_length", "tag_length"):
        value = header.get(name)
        if type(value) is not int or value < 0:
            _reject("invalid_component_length", f"invalid {name}")
        lengths.append(int(value))
    return header, shape, (lengths[0], lengths[1], lengths[2], lengths[3])


def _json_object_no_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _reject("duplicate_header_field", "duplicate JSON object field")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    _reject("invalid_header_json", f"non-finite JSON constant: {value}")


def associated_data_from_header(
    header_bytes: bytes, nonce: bytes, public_payload: bytes
) -> bytes:
    return (
        b"QSB1-AAD-v1"
        + len(header_bytes).to_bytes(4, "big")
        + header_bytes
        + nonce
        + public_payload
    )


def associated_data(parts: EnvelopeParts) -> bytes:
    header = _header(parts)
    _validate_header_fields(header)
    header_bytes = canonical_json_bytes(header)
    return associated_data_from_header(header_bytes, parts.nonce, parts.public_payload)


def encode_envelope(parts: EnvelopeParts) -> bytes:
    header = _header(parts)
    _validate_header_fields(header)
    header_bytes = canonical_json_bytes(header)
    if len(header_bytes) > MAX_HEADER_LENGTH:
        raise ValueError("header exceeds the QSB1 maximum length")
    prefix = MAGIC + bytes([VERSION]) + len(header_bytes).to_bytes(4, "big")
    return (
        prefix
        + header_bytes
        + parts.nonce
        + parts.public_payload
        + parts.protected_payload
        + parts.tag
    )


def parse_envelope(blob: bytes) -> ParsedEnvelope:
    if type(blob) is not bytes:
        _reject("invalid_input_type", "serialized envelope must be bytes")
    if len(blob) < PREFIX_LENGTH:
        _reject("truncated_prefix", "serialized envelope is shorter than the fixed prefix")
    if blob[:4] != MAGIC:
        _reject("invalid_magic", "invalid envelope magic")
    if blob[4] != VERSION:
        _reject("unsupported_version", "unsupported envelope version")

    header_len = int.from_bytes(blob[5:9], "big")
    if (
        header_len <= 0
        or header_len > MAX_HEADER_LENGTH
        or PREFIX_LENGTH + header_len > len(blob)
    ):
        _reject("invalid_header_length", "invalid header length")
    header_bytes = blob[PREFIX_LENGTH : PREFIX_LENGTH + header_len]
    try:
        header_text = header_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise EnvelopeFormatError(
            "invalid_header_encoding", "header must be ASCII JSON"
        ) from exc
    try:
        header = json.loads(
            header_text,
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except EnvelopeFormatError:
        raise
    except Exception as exc:
        raise EnvelopeFormatError("invalid_header_json", "invalid header JSON") from exc

    validated_header, _, lengths = _validate_header_fields(header)
    try:
        canonical = canonical_json_bytes(validated_header)
    except (TypeError, ValueError) as exc:
        raise EnvelopeFormatError(
            "invalid_header_json", "header contains unsupported JSON values"
        ) from exc
    if canonical != header_bytes:
        _reject("noncanonical_header_encoding", "noncanonical header encoding")

    expected = PREFIX_LENGTH + header_len + sum(lengths)
    if len(blob) != expected:
        _reject("envelope_length_mismatch", "envelope length mismatch or trailing bytes")

    position = PREFIX_LENGTH + header_len
    nonce = blob[position : position + lengths[0]]
    position += lengths[0]
    public = blob[position : position + lengths[1]]
    position += lengths[1]
    protected = blob[position : position + lengths[2]]
    position += lengths[2]
    tag = blob[position : position + lengths[3]]
    return ParsedEnvelope(validated_header, header_bytes, nonce, public, protected, tag)
