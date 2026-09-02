from __future__ import annotations

from .envelope import encode_envelope, parse_envelope
from .models import EnvelopeParts


def _flip(payload: bytes) -> bytes:
    if not payload:
        return payload
    changed = bytearray(payload)
    changed[len(changed) // 2] ^= 0x01
    return bytes(changed)


def tamper_object(blob: bytes) -> bytes:
    parsed = parse_envelope(blob)
    header = parsed.header
    public = parsed.public_payload
    protected = parsed.protected_payload
    nonce = parsed.nonce
    if public:
        public = _flip(public)
    elif protected:
        protected = _flip(protected)
    elif nonce:
        nonce = _flip(nonce)
    else:
        raise ValueError("object has no mutable field")
    return encode_envelope(EnvelopeParts(
        header["method_id"], tuple(header["image_shape"]), header["descriptor"], header["metadata"],
        nonce, public, protected, parsed.tag,
    ))
