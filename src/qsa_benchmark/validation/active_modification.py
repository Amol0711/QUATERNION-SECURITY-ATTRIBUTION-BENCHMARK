from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import numpy as np

from qsa_benchmark.benchmark.envelope import EnvelopeFormatError, encode_envelope, parse_envelope
from qsa_benchmark.benchmark.models import EnvelopeParts, ParsedEnvelope, RunContext
from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES, make_method
from qsa_benchmark.benchmark.utils import canonical_json_bytes
from qsa_benchmark.protocol.registry import method_policy
from qsa_benchmark.protocol.schedule import derive_fixed_reference_bytes, public_material_length

CONFIG_RELATIVE_PATH = "configs/reference/active_modification_inputs.json"
CONFIG_SCHEMA_RELATIVE_PATH = "configs/reference/active_modification_inputs.schema.json"
SCHEDULE_ID = "QSA-ACTIVE-MODIFICATION-SCHEDULE-V1"


@dataclass(frozen=True)
class ActiveMutation:
    mutation_id: str
    target_component: str
    mutation_class: str
    parser_level: bool
    object_bytes: bytes


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def reencode_envelope(
    parsed: ParsedEnvelope,
    *,
    header: dict[str, Any] | None = None,
    nonce: bytes | None = None,
    public: bytes | None = None,
    protected: bytes | None = None,
    tag: bytes | None = None,
) -> bytes:
    selected = parsed.header if header is None else header
    return encode_envelope(
        EnvelopeParts(
            str(selected["method_id"]),
            tuple(int(value) for value in selected["image_shape"]),
            dict(selected["descriptor"]),
            dict(selected["metadata"]),
            parsed.nonce if nonce is None else nonce,
            parsed.public_payload if public is None else public,
            parsed.protected_payload if protected is None else protected,
            parsed.tag if tag is None else tag,
        )
    )


def _flip(blob: bytes) -> bytes:
    if not blob:
        raise ValueError("cannot mutate an empty component")
    output = bytearray(blob)
    output[len(output) // 2] ^= 1
    return bytes(output)


def _mutated_header(parsed: ParsedEnvelope, target: str) -> dict[str, Any]:
    header = json.loads(json.dumps(parsed.header, allow_nan=False))
    if target == "descriptor":
        descriptor = header["descriptor"]
        if descriptor:
            key = sorted(descriptor)[0]
            value = descriptor[key]
            if isinstance(value, bool):
                descriptor[key] = not value
            elif isinstance(value, int):
                descriptor[key] = value + 1
            elif isinstance(value, float):
                descriptor[key] = value + 1.0
            elif isinstance(value, str):
                descriptor[key] = value + "-mut"
            elif isinstance(value, list):
                descriptor[key] = list(value) + [0]
            elif isinstance(value, dict):
                descriptor[key]["mutation"] = 1
            else:
                descriptor["mutation"] = 1
        else:
            descriptor["mutation"] = 1
    elif target == "metadata":
        metadata = header["metadata"]
        if type(metadata.get("schema")) is int:
            metadata["schema"] += 1
        else:
            metadata["mutation"] = 1
    else:
        raise ValueError(f"unknown header mutation target: {target}")
    return header


def generate_active_mutations(first: bytes, donor: bytes) -> list[ActiveMutation]:
    """Generate every registered byte-distinct modification for one object pair."""

    primary = parse_envelope(first)
    secondary = parse_envelope(donor)
    mutations: list[ActiveMutation] = []
    seen: dict[bytes, str] = {}

    def add(
        mutation_id: str,
        target_component: str,
        mutation_class: str,
        parser_level: bool,
        modified: bytes,
    ) -> None:
        if modified == first:
            raise AssertionError(f"{mutation_id} did not change the serialized object")
        previous = seen.get(modified)
        if previous is not None:
            raise AssertionError(f"{previous} and {mutation_id} produced the same object")
        seen[modified] = mutation_id
        mutations.append(
            ActiveMutation(
                mutation_id=mutation_id,
                target_component=target_component,
                mutation_class=mutation_class,
                parser_level=parser_level,
                object_bytes=modified,
            )
        )

    if primary.nonce:
        add("nonce_bit_flip", "nonce", "component_bit_flip", False, reencode_envelope(primary, nonce=_flip(primary.nonce)))
    if primary.public_payload:
        add("public_payload_bit_flip", "public_payload", "component_bit_flip", False, reencode_envelope(primary, public=_flip(primary.public_payload)))
    if primary.protected_payload:
        add("protected_payload_bit_flip", "protected_payload", "component_bit_flip", False, reencode_envelope(primary, protected=_flip(primary.protected_payload)))
    if primary.tag:
        add("tag_bit_flip", "tag", "component_bit_flip", False, reencode_envelope(primary, tag=_flip(primary.tag)))
    add("descriptor_mutation", "descriptor", "header_semantic_mutation", False, reencode_envelope(primary, header=_mutated_header(primary, "descriptor")))
    add("metadata_mutation", "metadata", "header_semantic_mutation", False, reencode_envelope(primary, header=_mutated_header(primary, "metadata")))

    if primary.protected_payload and len(primary.protected_payload) == len(secondary.protected_payload) and primary.protected_payload != secondary.protected_payload:
        add("protected_payload_splice", "protected_payload", "same_length_splice", False, reencode_envelope(primary, protected=secondary.protected_payload))
    elif primary.public_payload and len(primary.public_payload) == len(secondary.public_payload) and primary.public_payload != secondary.public_payload:
        add("public_payload_splice", "public_payload", "same_length_splice", False, reencode_envelope(primary, public=secondary.public_payload))
    else:
        raise AssertionError("no byte-distinct same-length payload splice is available")

    add("trailing_byte_append", "framing", "noncanonical_length", True, first + b"\x00")
    add("one_byte_truncation", "framing", "noncanonical_length", True, first[:-1])
    return mutations


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def generate_donor_image(base: np.ndarray) -> np.ndarray:
    if base.shape != (16, 16, 3) or base.dtype != np.uint8:
        raise ValueError("active-modification base input must be 16-by-16 RGB uint8")
    y, x, channel = np.indices(base.shape)
    pattern = (17 * y + 29 * x + 43 * channel + 91) % 256
    shifted = np.roll(np.roll(base, 3, axis=0), 5, axis=1)
    return np.bitwise_xor(shifted, pattern.astype(np.uint8))


def load_active_modification_config(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = _load_json(root / CONFIG_RELATIVE_PATH)
    schema = _load_json(root / CONFIG_SCHEMA_RELATIVE_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(config)
    method_order = list(EXTENDED_METHOD_FACTORIES)
    if config["method_order"] != method_order:
        raise ValueError("active-modification method order differs from executable registry")
    base_path = root / config["input_pair"]["base_path"]
    base_bytes = base_path.read_bytes()
    if _sha256(base_bytes) != config["input_pair"]["base_sha256"]:
        raise ValueError("active-modification base input digest mismatch")
    base = np.frombuffer(base_bytes, dtype=np.uint8).reshape(config["input_pair"]["shape"]).copy()
    donor = generate_donor_image(base)
    if _sha256(donor.tobytes(order="C")) != config["input_pair"]["donor_sha256"]:
        raise ValueError("active-modification donor input digest mismatch")
    return config


def load_active_input_pair(repo_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(repo_root).resolve()
    config = load_active_modification_config(root)
    data = (root / config["input_pair"]["base_path"]).read_bytes()
    base = np.frombuffer(data, dtype=np.uint8).reshape(config["input_pair"]["shape"]).copy()
    return base, generate_donor_image(base)


def derive_active_context(
    config: Mapping[str, Any],
    *,
    method_id: str,
    ordinal: int,
    role: str,
    input_sha256: str,
    payload_length: int,
) -> RunContext:
    if role not in {"base", "donor"}:
        raise ValueError("active-modification role must be base or donor")
    if config["method_order"][ordinal - 1] != method_id:
        raise ValueError("active-modification method ordinal mismatch")
    profile = config["context_profile"]
    nonce = derive_fixed_reference_bytes(
        domain_ascii=profile["nonce_domain_ascii"] + "|" + role.upper(),
        method_id=method_id,
        ordinal=ordinal,
        input_sha256=input_sha256,
        length=method_policy(method_id).nonce_length,
    )
    seed = int.from_bytes(
        derive_fixed_reference_bytes(
            domain_ascii=profile["seed_domain_ascii"] + "|" + role.upper(),
            method_id=method_id,
            ordinal=ordinal,
            input_sha256=input_sha256,
            length=8,
        ),
        "big",
    )
    material_length = public_material_length(method_id, payload_length)
    public_material = derive_fixed_reference_bytes(
        domain_ascii=profile["public_material_domain_ascii"] + "|" + role.upper(),
        method_id=method_id,
        ordinal=ordinal,
        input_sha256=input_sha256,
        length=material_length,
    )
    image_id = profile["base_image_id"] if role == "base" else profile["donor_image_id"]
    return RunContext(
        master_key=bytes.fromhex(profile["master_key_hex"]),
        nonce=nonce,
        seed=seed,
        image_id=image_id,
        method_id=method_id,
        run_id=profile["run_id_template"].format(method_id=method_id, role=role),
        public_metadata=dict(profile["public_metadata"]),
        public_material=public_material,
        protocol_id=profile["protocol_id"],
    )


def _byte_difference_count(left: bytes, right: bytes) -> int:
    common = sum(a != b for a, b in zip(left, right))
    return common + abs(len(left) - len(right))


def build_active_modification_rows(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    config = load_active_modification_config(root)
    base, donor = load_active_input_pair(root)
    base_digest = _sha256(base.tobytes(order="C"))
    donor_digest = _sha256(donor.tobytes(order="C"))
    rows: list[dict[str, Any]] = []
    case_ordinal = 0

    for method_ordinal, method_id in enumerate(config["method_order"], start=1):
        method = make_method(method_id, profile="extended")
        base_context = derive_active_context(config, method_id=method_id, ordinal=method_ordinal, role="base", input_sha256=base_digest, payload_length=base.nbytes)
        donor_context = derive_active_context(config, method_id=method_id, ordinal=method_ordinal, role="donor", input_sha256=donor_digest, payload_length=donor.nbytes)
        first = method.encrypt(base, base_context).object_bytes
        second = method.encrypt(donor, donor_context).object_bytes
        base_parsed = parse_envelope(first)
        if not np.array_equal(method.decrypt(first, base_context), base):
            raise AssertionError(f"base round trip failed for {method_id}")
        if not np.array_equal(method.decrypt(second, donor_context), donor):
            raise AssertionError(f"donor round trip failed for {method_id}")

        for method_mutation_ordinal, mutation in enumerate(generate_active_mutations(first, second), start=1):
            case_ordinal += 1
            accepted = False
            changed: bool | None = None
            exception_class = ""
            parser_code = ""
            failure_stage = "none"
            try:
                recovered = method.decrypt(mutation.object_bytes, base_context)
                accepted = True
                changed = not np.array_equal(recovered, base)
            except EnvelopeFormatError as exc:
                exception_class = type(exc).__name__
                parser_code = exc.code
                failure_stage = "parser"
            except Exception as exc:  # cryptographic integrity and construction checks
                exception_class = type(exc).__name__
                failure_stage = "integrity_or_release"

            authenticated = bool(method.authenticated)
            if mutation.parser_level:
                expected_outcome = "parser_reject"
            elif authenticated:
                expected_outcome = "integrity_reject"
            elif accepted and changed:
                expected_outcome = "accepted_plaintext_changed"
            elif accepted:
                expected_outcome = "accepted_plaintext_unchanged"
            else:
                expected_outcome = "unexpected_reject"

            context_core = {
                "method_id": method_id,
                "protocol_id": base_context.protocol_id,
                "run_id": base_context.run_id,
                "image_id": base_context.image_id,
                "nonce_hex": base_context.nonce.hex(),
                "seed": int(base_context.seed),
                "public_material_sha256": _sha256(base_context.public_material),
                "public_metadata": dict(base_context.public_metadata or {}),
            }
            rows.append({
                "case_id": f"AM{case_ordinal:03d}_{method_id}_{mutation.mutation_id}",
                "case_ordinal": case_ordinal,
                "method_ordinal": method_ordinal,
                "method_mutation_ordinal": method_mutation_ordinal,
                "method_id": method_id,
                "authenticated": authenticated,
                "mutation_id": mutation.mutation_id,
                "target_component": mutation.target_component,
                "mutation_class": mutation.mutation_class,
                "parser_level": mutation.parser_level,
                "expected_outcome": expected_outcome,
                "expected_accepted": accepted,
                "expected_plaintext_changed": "NA" if changed is None else changed,
                "failure_stage": failure_stage,
                "expected_exception_class": exception_class,
                "expected_parser_code": parser_code,
                "base_input_sha256": base_digest,
                "donor_input_sha256": donor_digest,
                "context_sha256": _sha256(canonical_json_bytes(context_core)),
                "original_object_sha256": _sha256(first),
                "mutated_object_sha256": _sha256(mutation.object_bytes),
                "original_object_bytes": len(first),
                "mutated_object_bytes": len(mutation.object_bytes),
                "byte_difference_count": _byte_difference_count(first, mutation.object_bytes),
                "nonce_bytes": len(base_parsed.nonce),
                "public_payload_bytes": len(base_parsed.public_payload),
                "protected_payload_bytes": len(base_parsed.protected_payload),
                "tag_bytes": len(base_parsed.tag),
                "byte_distinct": mutation.object_bytes != first,
            })

    _validate_active_rows(rows, config["expected_counts"])
    return rows


def active_modification_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    authenticated = [row for row in rows if bool(row["authenticated"])]
    unauthenticated = [row for row in rows if not bool(row["authenticated"])]
    return {
        "total_cases": len(rows),
        "authenticated_cases": len(authenticated),
        "authenticated_rejections": sum(not bool(row["expected_accepted"]) for row in authenticated),
        "unauthenticated_cases": len(unauthenticated),
        "unauthenticated_acceptances": sum(bool(row["expected_accepted"]) for row in unauthenticated),
        "unauthenticated_rejections": sum(not bool(row["expected_accepted"]) for row in unauthenticated),
        "unauthenticated_plaintext_changed": sum(row["expected_plaintext_changed"] is True for row in unauthenticated),
        "unauthenticated_plaintext_unchanged": sum(row["expected_plaintext_changed"] is False for row in unauthenticated),
        "parser_length_violations": sum(bool(row["parser_level"]) for row in rows),
    }


def _validate_active_rows(rows: list[Mapping[str, Any]], expected: Mapping[str, Any]) -> None:
    if [int(row["case_ordinal"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise AssertionError("active-modification case ordering is not contiguous")
    if len({str(row["case_id"]) for row in rows}) != len(rows):
        raise AssertionError("active-modification case identifiers are not unique")
    if len({str(row["mutated_object_sha256"]) for row in rows}) != len(rows):
        raise AssertionError("active-modification objects are not globally unique")
    if not all(bool(row["byte_distinct"]) and int(row["byte_difference_count"]) > 0 for row in rows):
        raise AssertionError("active-modification schedule contains a no-op")
    counts = active_modification_counts(rows)
    if counts != dict(expected):
        raise AssertionError(f"active-modification count mismatch: {counts} != {dict(expected)}")
    for row in rows:
        if row["parser_level"]:
            if row["expected_outcome"] != "parser_reject" or row["expected_parser_code"] != "envelope_length_mismatch":
                raise AssertionError(f"invalid parser outcome for {row['case_id']}")
        elif row["authenticated"]:
            if row["expected_outcome"] != "integrity_reject" or row["expected_accepted"]:
                raise AssertionError(f"invalid authenticated outcome for {row['case_id']}")
        elif row["expected_outcome"] not in {"accepted_plaintext_changed", "accepted_plaintext_unchanged"}:
            raise AssertionError(f"invalid unauthenticated outcome for {row['case_id']}")


def active_modification_summary(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_active_modification_config(root)
    rows = build_active_modification_rows(root)
    return {
        "schedule_id": config["schedule_id"],
        "counts": active_modification_counts(rows),
        "method_count": len(config["method_order"]),
        "base_input_sha256": config["input_pair"]["base_sha256"],
        "donor_input_sha256": config["input_pair"]["donor_sha256"],
        "ordered_case_digest": _sha256(canonical_json_bytes([{key: row[key] for key in ("case_id", "mutated_object_sha256", "expected_outcome")} for row in rows])),
    }
