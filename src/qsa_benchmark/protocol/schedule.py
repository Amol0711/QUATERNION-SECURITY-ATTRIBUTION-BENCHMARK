from __future__ import annotations

import hashlib

from qsa_benchmark.benchmark.models import RunContext
from qsa_benchmark.benchmark.utils import derive_key

from .models import ContextPair, DifferentialProtocol
from .registry import method_policy, operation_regime


SCHEDULE_DOMAIN_SEPARATOR = b"QSA-PROTOCOL-SCHEDULE-V1"
METHOD_KEY_LABEL_TEMPLATE = "QSA|METHOD|{method_id}|KEY|{key_index}"
SCHEDULE_LABEL_FIELDS = (
    "protocol_id",
    "master_seed",
    "method_id",
    "image_id",
    "perturbation_id",
    "pair_ordinal",
    "key_index",
    "state_index",
)
P1_DERIVATION_LABELS = {
    "nonce": "P1-NONCE",
    "seed": "P1-SEED",
    "public_material": "P1-PUBLIC-MATERIAL",
}
P2_DERIVATION_LABELS = {
    "left_nonce": "P2-NONCE-L",
    "right_nonce": "P2-NONCE-R",
    "left_seed": "P2-SEED-L",
    "right_seed": "P2-SEED-R",
    "left_public_material": "P2-PUBLIC-MATERIAL-L",
    "right_public_material": "P2-PUBLIC-MATERIAL-R",
}
PUBLIC_MATERIAL_LENGTH_RULES: dict[str, str | int] = {
    "B21_public_fresh_pad": "payload_length",
    "B22_public_wideblock_prp": 32,
    "default": 0,
}


def schedule_registry() -> dict[str, object]:
    """Return the deterministic context-derivation contract."""

    return {
        "domain_separator_ascii": SCHEDULE_DOMAIN_SEPARATOR.decode("ascii"),
        "method_key_label_template": METHOD_KEY_LABEL_TEMPLATE,
        "method_key_length_bytes": 32,
        "schedule_label_fields": list(SCHEDULE_LABEL_FIELDS),
        "p1_derivation_labels": dict(P1_DERIVATION_LABELS),
        "p2_derivation_labels": dict(P2_DERIVATION_LABELS),
        "public_material_length_rules": dict(PUBLIC_MATERIAL_LENGTH_RULES),
        "execution_label_in_serialized_object": False,
    }


def _xof(label: str, length: int, *values: object) -> bytes:
    h = hashlib.shake_256()
    h.update(SCHEDULE_DOMAIN_SEPARATOR)
    label_blob = label.encode("utf-8")
    h.update(len(label_blob).to_bytes(2, "big"))
    h.update(label_blob)
    for value in values:
        blob = str(value).encode("utf-8")
        h.update(len(blob).to_bytes(4, "big"))
        h.update(blob)
    return h.digest(length)


def _seed(label: str, *values: object) -> int:
    return int.from_bytes(_xof(label, 8, *values), "big")


def derive_fixed_reference_bytes(
    *,
    domain_ascii: str,
    method_id: str,
    ordinal: int,
    input_sha256: str,
    length: int,
) -> bytes:
    """Derive deterministic public bytes for frozen reference contexts.

    The derivation is domain separated from the experimental pair schedule.
    It binds each value to the construction identifier, its registered
    ordinal, and the canonical input digest.  The resulting keys, nonces,
    seeds, and public material are published test values rather than secrets.
    """

    if not domain_ascii or not domain_ascii.isascii():
        raise ValueError("fixed-reference domain must be nonempty ASCII")
    if not method_id or not method_id.isascii():
        raise ValueError("fixed-reference method identifier must be nonempty ASCII")
    if ordinal <= 0:
        raise ValueError("fixed-reference ordinal must be positive")
    if length < 0:
        raise ValueError("fixed-reference derivation length must be nonnegative")
    if len(input_sha256) != 64:
        raise ValueError("fixed-reference input digest must be lowercase SHA-256 hex")
    try:
        input_digest = bytes.fromhex(input_sha256)
    except ValueError as exc:
        raise ValueError("fixed-reference input digest must be lowercase SHA-256 hex") from exc
    if input_digest.hex() != input_sha256:
        raise ValueError("fixed-reference input digest must be lowercase SHA-256 hex")

    h = hashlib.shake_256()
    h.update(b"QSA-FIXED-REFERENCE-CONTEXT-V1")
    for blob in (domain_ascii.encode("ascii"), method_id.encode("ascii")):
        h.update(len(blob).to_bytes(2, "big"))
        h.update(blob)
    h.update(int(ordinal).to_bytes(4, "big"))
    h.update(input_digest)
    return h.digest(length)


def public_material_length(method_id: str, payload_length: int) -> int:
    rule = PUBLIC_MATERIAL_LENGTH_RULES.get(
        method_id, PUBLIC_MATERIAL_LENGTH_RULES["default"]
    )
    return payload_length if rule == "payload_length" else int(rule)


def build_context_pair(
    *,
    master_key: bytes,
    master_seed: int,
    method_id: str,
    protocol: DifferentialProtocol | str,
    image_id: str,
    perturbation_id: str,
    pair_ordinal: int,
    key_index: int,
    state_index: int,
    payload_length: int,
    protocol_id: str = "QSA-PROTOCOL-V1",
) -> ContextPair:
    protocol = DifferentialProtocol(protocol)
    policy = method_policy(method_id)
    if protocol is DifferentialProtocol.P2_FRESH_RANDOMNESS and not policy.p2_applicable:
        raise ValueError(f"P2 is not applicable to {method_id}")
    if min(pair_ordinal, key_index, state_index) < 0:
        raise ValueError("schedule indices must be nonnegative")
    if payload_length <= 0:
        raise ValueError("payload length must be positive")

    method_key = derive_key(
        master_key,
        METHOD_KEY_LABEL_TEMPLATE.format(method_id=method_id, key_index=key_index),
        32,
    )
    schedule_label = (
        protocol_id,
        master_seed,
        method_id,
        image_id,
        perturbation_id,
        pair_ordinal,
        key_index,
        state_index,
    )
    nonce_length = policy.nonce_length
    material_length = public_material_length(method_id, payload_length)

    if protocol is DifferentialProtocol.P1_COMMON_CONTEXT:
        nonce_left = nonce_right = _xof(
            P1_DERIVATION_LABELS["nonce"], nonce_length, *schedule_label
        )
        seed_left = seed_right = _seed(
            P1_DERIVATION_LABELS["seed"], *schedule_label
        )
        material = (
            _xof(
                P1_DERIVATION_LABELS["public_material"],
                material_length,
                *schedule_label,
            )
            if material_length
            else b""
        )
        material_left = material_right = material
        code = "P1-CC"
    else:
        nonce_left = _xof(
            P2_DERIVATION_LABELS["left_nonce"], nonce_length, *schedule_label
        )
        nonce_right = _xof(
            P2_DERIVATION_LABELS["right_nonce"], nonce_length, *schedule_label
        )
        seed_left = _seed(
            P2_DERIVATION_LABELS["left_seed"], *schedule_label
        )
        seed_right = _seed(
            P2_DERIVATION_LABELS["right_seed"], *schedule_label
        )
        material_left = (
            _xof(
                P2_DERIVATION_LABELS["left_public_material"],
                material_length,
                *schedule_label,
            )
            if material_length
            else b""
        )
        material_right = (
            _xof(
                P2_DERIVATION_LABELS["right_public_material"],
                material_length,
                *schedule_label,
            )
            if material_length
            else b""
        )
        if (
            nonce_left == nonce_right
            or seed_left == seed_right
            or (material_length and material_left == material_right)
        ):
            raise AssertionError("P2 context derivation collision")
        code = "P2-FR"

    public_metadata = {
        "benchmark": "QSA-EXPERIMENT",
        "metadata_profile": "security-view-v2",
        "object_semantics": "complete-public-object",
        "protocol": code,
        "schema": 2,
    }
    run_base = (
        f"{protocol_id}|{code}|{method_id}|{image_id}|{perturbation_id}"
        f"|p{pair_ordinal}|k{key_index}|s{state_index}"
    )
    left = RunContext(
        method_key,
        nonce_left,
        seed_left,
        image_id,
        method_id,
        run_base + "|L",
        public_metadata,
        material_left,
        protocol_id,
    )
    right = RunContext(
        method_key,
        nonce_right,
        seed_right,
        image_id,
        method_id,
        run_base + "|R",
        public_metadata,
        material_right,
        protocol_id,
    )
    return ContextPair(
        protocol=protocol,
        left=left,
        right=right,
        same_master_key=left.master_key == right.master_key,
        same_nonce=left.nonce == right.nonce,
        same_seed=left.seed == right.seed,
        same_public_material=left.public_material == right.public_material,
        operation_regime=operation_regime(method_id, protocol.value),
    )
