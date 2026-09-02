from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qsa_benchmark.benchmark.envelope import parse_envelope
from qsa_benchmark.benchmark.controls import FIXED_BODY_PREFIX_LENGTH

from .registry import method_policy


@dataclass(frozen=True)
class ProtocolViews:
    method_id: str
    object_bytes: bytes
    header_bytes: bytes
    nonce: bytes
    public_payload: bytes
    protected_payload: bytes
    tag: bytes
    metric_body: bytes
    metric_body_source: str
    body_metric_domain: str
    image_shape: tuple[int, int, int]
    descriptor: dict[str, Any]
    metadata: dict[str, Any]


def extract_protocol_views(object_bytes: bytes, expected_method_id: str | None = None) -> ProtocolViews:
    parsed = parse_envelope(object_bytes)
    method_id = str(parsed.header["method_id"])
    if expected_method_id is not None and method_id != expected_method_id:
        raise ValueError("method identifier mismatch in protocol view")
    policy = method_policy(method_id)
    body = parsed.protected_payload if policy.metric_body_source == "protected_payload" else parsed.public_payload
    if not body:
        raise ValueError(f"empty metric body for {method_id}")
    metadata = dict(parsed.header["metadata"])
    forbidden = {"image_id", "run_id", "pair_id", "key_index", "state_index", "perturbation_id"}
    if forbidden.intersection(metadata):
        raise ValueError("experiment identifier leaked into adversary-visible metadata")
    if policy.body_metric_domain == "fixed_prefix_rgb_suffix":
        expected = int(parsed.header["descriptor"].get("fixed_prefix_length", -1))
        if expected != FIXED_BODY_PREFIX_LENGTH:
            raise ValueError("fixed-header domain declaration mismatch")
    return ProtocolViews(
        method_id=method_id,
        object_bytes=object_bytes,
        header_bytes=parsed.header_bytes,
        nonce=parsed.nonce,
        public_payload=parsed.public_payload,
        protected_payload=parsed.protected_payload,
        tag=parsed.tag,
        metric_body=body,
        metric_body_source=policy.metric_body_source,
        body_metric_domain=policy.body_metric_domain,
        image_shape=tuple(int(v) for v in parsed.header["image_shape"]),
        descriptor=dict(parsed.header["descriptor"]),
        metadata=metadata,
    )


def body_projections(views: ProtocolViews) -> dict[str, tuple[bytes, bool, str]]:
    """Return body projection -> (bytes, inferentially eligible, semantic domain)."""
    h, w, channels = views.image_shape
    if channels != 3:
        raise ValueError("the protocol requires RGB images")
    raw_length = h * w * 3
    body = views.metric_body
    domain = views.body_metric_domain
    projections: dict[str, tuple[bytes, bool, str]] = {
        "aggregate_full_body": (body, domain != "int32_serialized", domain)
    }
    if domain == "rgb_u8":
        if len(body) != raw_length:
            raise ValueError(f"RGB body length mismatch for {views.method_id}")
        for name, offset in (("R", 0), ("G", 1), ("B", 2)):
            projections[f"channel_{name}"] = (body[offset::3], True, "rgb_channel_u8")
    elif domain == "fixed_prefix_rgb_suffix":
        if len(body) != FIXED_BODY_PREFIX_LENGTH + raw_length:
            raise ValueError(f"fixed-header body length mismatch for {views.method_id}")
        suffix = body[FIXED_BODY_PREFIX_LENGTH:]
        projections["aggregate_cryptographic_suffix"] = (suffix, True, "rgb_u8")
        for name, offset in (("R", 0), ("G", 1), ("B", 2)):
            projections[f"suffix_channel_{name}"] = (suffix[offset::3], True, "rgb_channel_u8")
    return projections
