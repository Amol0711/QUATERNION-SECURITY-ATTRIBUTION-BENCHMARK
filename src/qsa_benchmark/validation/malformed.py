from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import jsonschema

from qsa_benchmark.benchmark.envelope import (
    MAGIC,
    MAX_HEADER_LENGTH,
    PREFIX_LENGTH,
    VERSION,
    EnvelopeFormatError,
    parse_envelope,
)
from qsa_benchmark.benchmark.utils import canonical_json_bytes, sha256_file

MANIFEST_ID = "QSA-MALFORMED-OBJECT-MANIFEST-V1"
SCHEMA_VERSION = 1
BASE_OBJECT_RELATIVE_PATH = "reference/known_answers/objects/B20_full_aead_explicit_preview.qsb"
MANIFEST_RELATIVE_PATH = "reference/malformed_objects/manifest.json"
VECTOR_DIRECTORY_RELATIVE_PATH = "reference/malformed_objects/vectors"

_SOURCE_AUTHORITY_PATHS = (
    "pyproject.toml",
    "src/qsa_benchmark/benchmark/envelope.py",
    "src/qsa_benchmark/benchmark/models.py",
    "src/qsa_benchmark/benchmark/utils.py",
    "src/qsa_benchmark/validation/malformed.py",
    "scripts/generate_reference_assets.py",
    BASE_OBJECT_RELATIVE_PATH,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _human_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError(f"unsafe repository-relative path: {relative}")
    result = (root / Path(*pure.parts)).resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"path escapes repository root: {relative}")
    return result


def _prefix(header: bytes, *, version: int = VERSION, declared_length: int | None = None) -> bytes:
    length = len(header) if declared_length is None else declared_length
    return MAGIC + bytes([version]) + int(length).to_bytes(4, "big")


def _payload(parsed: Any) -> bytes:
    return parsed.nonce + parsed.public_payload + parsed.protected_payload + parsed.tag


def _with_raw_header(parsed: Any, header_bytes: bytes) -> bytes:
    return _prefix(header_bytes) + header_bytes + _payload(parsed)


def _with_header(parsed: Any, update: Callable[[dict[str, Any]], None], *, canonical: bool = True) -> bytes:
    header = json.loads(json.dumps(parsed.header, allow_nan=False))
    update(header)
    if canonical:
        encoded = canonical_json_bytes(header)
    else:
        encoded = json.dumps(header, ensure_ascii=True, separators=(", ", ": "), sort_keys=False, allow_nan=False).encode("ascii")
    return _with_raw_header(parsed, encoded)


def _raw_component_offsets(base: bytes) -> dict[str, tuple[int, int]]:
    parsed = parse_envelope(base)
    position = PREFIX_LENGTH + len(parsed.header_bytes)
    result: dict[str, tuple[int, int]] = {}
    for name, component in (
        ("nonce", parsed.nonce),
        ("public_payload", parsed.public_payload),
        ("protected_payload", parsed.protected_payload),
        ("tag", parsed.tag),
    ):
        result[name] = (position, position + len(component))
        position += len(component)
    return result


def _remove_component_byte(base: bytes, component: str) -> bytes:
    start, end = _raw_component_offsets(base)[component]
    if start == end:
        raise ValueError(f"base object has no {component}")
    index = start + (end - start) // 2
    return base[:index] + base[index + 1 :]


def _case_specs(base: bytes) -> list[tuple[str, str, str, str, bytes]]:
    parsed = parse_envelope(base)
    canonical = parsed.header_bytes
    header = parsed.header
    payload = _payload(parsed)

    duplicate_header = (
        b'{"method_id":"duplicate",' + canonical[1:]
    )
    reordered = json.dumps(header, ensure_ascii=True, separators=(",", ":"), sort_keys=False, allow_nan=False).encode("ascii")
    nonfinite = canonical.replace(b'"schema":1', b'"schema":NaN', 1)
    if nonfinite == canonical:
        raise AssertionError("unable to construct non-finite JSON vector")

    specs: list[tuple[str, str, str, str, bytes]] = [
        ("empty_object", "fixed_prefix", "truncated_prefix", "zero-byte object", b""),
        ("truncated_fixed_prefix", "fixed_prefix", "truncated_prefix", "eight-byte prefix fragment", base[: PREFIX_LENGTH - 1]),
        ("invalid_magic", "fixed_prefix", "invalid_magic", "invalid four-byte magic", b"BAD!" + base[4:]),
        ("unsupported_version", "fixed_prefix", "unsupported_version", "unsupported format version", base[:4] + bytes([VERSION + 1]) + base[5:]),
        ("zero_header_length", "header_framing", "invalid_header_length", "zero declared header length", base[:5] + (0).to_bytes(4, "big") + base[9:]),
        ("oversized_header_length", "header_framing", "invalid_header_length", "declared header exceeds the parser maximum", base[:5] + (MAX_HEADER_LENGTH + 1).to_bytes(4, "big") + base[9:]),
        ("declared_header_beyond_object", "header_framing", "invalid_header_length", "declared header extends beyond available bytes", base[:5] + (len(base) + 1).to_bytes(4, "big") + base[9:]),
        ("invalid_header_encoding", "header_json", "invalid_header_encoding", "header contains a non-ASCII byte", _with_raw_header(parsed, bytes([0xFF]) + canonical[1:])),
        ("malformed_header_json", "header_json", "invalid_header_json", "header is not valid JSON", _with_raw_header(parsed, b"{" + canonical[1:-1])),
        ("nonfinite_header_json", "header_json", "invalid_header_json", "header contains a non-finite JSON constant", _with_raw_header(parsed, nonfinite)),
        ("header_json_array", "header_semantics", "invalid_header_fields", "top-level header is a JSON array", _with_raw_header(parsed, b"[]")),
        ("duplicate_header_field", "header_json", "duplicate_header_field", "header repeats method_id", _with_raw_header(parsed, duplicate_header)),
        ("missing_header_field", "header_semantics", "invalid_header_fields", "header omits tag_length", _with_header(parsed, lambda item: item.pop("tag_length"))),
        ("unexpected_header_field", "header_semantics", "invalid_header_fields", "header adds an unsupported field", _with_header(parsed, lambda item: item.__setitem__("unexpected", 1))),
        ("noncanonical_whitespace", "canonical_encoding", "noncanonical_header_encoding", "header uses noncanonical whitespace", _with_raw_header(parsed, json.dumps(header, ensure_ascii=True, indent=1, sort_keys=True, allow_nan=False).encode("ascii"))),
        ("noncanonical_key_order", "canonical_encoding", "noncanonical_header_encoding", "header keys are not in canonical order", _with_raw_header(parsed, reordered)),
        ("empty_method_id", "header_semantics", "invalid_method_id", "method identifier is empty", _with_header(parsed, lambda item: item.__setitem__("method_id", ""))),
        ("nonnumeric_method_id", "header_semantics", "invalid_method_id", "method identifier has the wrong JSON type", _with_header(parsed, lambda item: item.__setitem__("method_id", 20))),
        ("whitespace_method_id", "header_semantics", "invalid_method_id", "method identifier contains whitespace", _with_header(parsed, lambda item: item.__setitem__("method_id", "B20 invalid"))),
        ("descriptor_not_object", "header_semantics", "invalid_descriptor", "descriptor is not a JSON object", _with_header(parsed, lambda item: item.__setitem__("descriptor", []))),
        ("metadata_not_object", "header_semantics", "invalid_metadata", "metadata is not a JSON object", _with_header(parsed, lambda item: item.__setitem__("metadata", []))),
        ("shape_wrong_rank", "header_semantics", "invalid_image_shape", "image shape has the wrong rank", _with_header(parsed, lambda item: item.__setitem__("image_shape", [16, 16]))),
        ("shape_zero_dimension", "header_semantics", "invalid_image_shape", "image shape contains a zero dimension", _with_header(parsed, lambda item: item.__setitem__("image_shape", [0, 16, 3]))),
        ("shape_boolean_dimension", "header_semantics", "invalid_image_shape", "image shape contains a Boolean dimension", _with_header(parsed, lambda item: item.__setitem__("image_shape", [True, 16, 3]))),
        ("shape_exceeds_bound", "header_semantics", "invalid_image_shape", "image dimension exceeds the parser bound", _with_header(parsed, lambda item: item.__setitem__("image_shape", [65536, 16, 3]))),
        ("negative_nonce_length", "component_lengths", "invalid_component_length", "nonce length is negative", _with_header(parsed, lambda item: item.__setitem__("nonce_length", -1))),
        ("boolean_tag_length", "component_lengths", "invalid_component_length", "tag length is Boolean", _with_header(parsed, lambda item: item.__setitem__("tag_length", True))),
        ("truncated_nonce", "component_framing", "envelope_length_mismatch", "one nonce byte is removed", _remove_component_byte(base, "nonce")),
        ("truncated_public_payload", "component_framing", "envelope_length_mismatch", "one public-payload byte is removed", _remove_component_byte(base, "public_payload")),
        ("truncated_protected_payload", "component_framing", "envelope_length_mismatch", "one protected-payload byte is removed", _remove_component_byte(base, "protected_payload")),
        ("truncated_tag", "component_framing", "envelope_length_mismatch", "one tag byte is removed", _remove_component_byte(base, "tag")),
        ("trailing_byte", "component_framing", "envelope_length_mismatch", "one trailing byte is appended", base + b"\x00"),
        ("missing_all_components", "component_framing", "envelope_length_mismatch", "all declared components are absent", _prefix(canonical) + canonical),
    ]
    if reordered == canonical:
        # The canonical order is sorted. Reverse insertion order guarantees a distinct order.
        reverse_header = {key: header[key] for key in reversed(list(header))}
        replacement = json.dumps(reverse_header, ensure_ascii=True, separators=(",", ":"), sort_keys=False, allow_nan=False).encode("ascii")
        specs[15] = ("noncanonical_key_order", "canonical_encoding", "noncanonical_header_encoding", "header keys are not in canonical order", _with_raw_header(parsed, replacement))
    return specs


def _manifest_schema() -> dict[str, Any]:
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    case = {
        "type": "object",
        "additionalProperties": False,
        "required": ["case_id", "category", "description", "expected_error_code", "expected_exception_class", "object_bytes", "object_path", "ordinal", "sha256"],
        "properties": {
            "case_id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
            "category": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 1},
            "expected_error_code": {"type": "string", "minLength": 1},
            "expected_exception_class": {"const": "EnvelopeFormatError"},
            "object_bytes": {"type": "integer", "minimum": 0},
            "object_path": {"type": "string", "pattern": "^reference/malformed_objects/vectors/[0-9]{2}_[a-z0-9_]+\\.qsb$"},
            "ordinal": {"type": "integer", "minimum": 1},
            "sha256": digest,
        },
    }
    source = {
        "type": "object", "additionalProperties": False,
        "required": ["path", "sha256"],
        "properties": {"path": {"type": "string", "minLength": 1}, "sha256": digest},
    }
    return {
        "$id": "urn:qsa:schema:malformed-object-manifest:v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["base_object", "case_count", "case_order_sha256", "cases", "manifest_id", "parser_contract", "schema_version", "source_authorities", "vector_set_sha256"],
        "properties": {
            "base_object": {
                "type": "object", "additionalProperties": False,
                "required": ["method_id", "object_bytes", "path", "sha256"],
                "properties": {
                    "method_id": {"const": "B20_full_aead_explicit_preview"},
                    "object_bytes": {"type": "integer", "minimum": 1},
                    "path": {"const": BASE_OBJECT_RELATIVE_PATH},
                    "sha256": digest,
                },
            },
            "case_count": {"type": "integer", "minimum": 1},
            "case_order_sha256": digest,
            "cases": {"type": "array", "minItems": 1, "items": case},
            "manifest_id": {"const": MANIFEST_ID},
            "parser_contract": {
                "type": "object", "additionalProperties": False,
                "required": ["format_magic_ascii", "format_version", "rejection_type", "stable_error_codes"],
                "properties": {
                    "format_magic_ascii": {"const": "QSB1"},
                    "format_version": {"const": 1},
                    "rejection_type": {"const": "EnvelopeFormatError"},
                    "stable_error_codes": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string"}},
                },
            },
            "schema_version": {"const": SCHEMA_VERSION},
            "source_authorities": {"type": "array", "minItems": 1, "items": source},
            "vector_set_sha256": digest,
        },
    }


def build_malformed_object_files(repo_root: str | Path) -> dict[str, bytes]:
    root = Path(repo_root).resolve()
    base = _safe_path(root, BASE_OBJECT_RELATIVE_PATH).read_bytes()
    parsed = parse_envelope(base)
    if parsed.header["method_id"] != "B20_full_aead_explicit_preview":
        raise ValueError("malformed-object base method differs from the public contract")
    files: dict[str, bytes] = {}
    cases: list[dict[str, Any]] = []
    codes: set[str] = set()
    seen_payloads: set[bytes] = set()
    for ordinal, (case_id, category, expected_code, description, payload) in enumerate(_case_specs(base), start=1):
        if payload in seen_payloads:
            raise AssertionError(f"duplicate malformed vector bytes: {case_id}")
        seen_payloads.add(payload)
        try:
            parse_envelope(payload)
        except EnvelopeFormatError as exc:
            if exc.code != expected_code:
                raise AssertionError(f"{case_id} produced {exc.code}, expected {expected_code}") from exc
        except Exception as exc:
            raise AssertionError(f"{case_id} produced unexpected {type(exc).__name__}") from exc
        else:
            raise AssertionError(f"{case_id} was accepted by the canonical parser")
        relative = f"{VECTOR_DIRECTORY_RELATIVE_PATH}/{ordinal:02d}_{case_id}.qsb"
        files[relative] = payload
        cases.append({
            "case_id": case_id,
            "category": category,
            "description": description,
            "expected_error_code": expected_code,
            "expected_exception_class": "EnvelopeFormatError",
            "object_bytes": len(payload),
            "object_path": relative,
            "ordinal": ordinal,
            "sha256": _sha256(payload),
        })
        codes.add(expected_code)
    case_order = [{"case_id": case["case_id"], "expected_error_code": case["expected_error_code"], "sha256": case["sha256"]} for case in cases]
    vector_set = [{"object_path": case["object_path"], "sha256": case["sha256"]} for case in cases]
    manifest = {
        "base_object": {"method_id": parsed.header["method_id"], "object_bytes": len(base), "path": BASE_OBJECT_RELATIVE_PATH, "sha256": _sha256(base)},
        "case_count": len(cases),
        "case_order_sha256": _sha256(canonical_json_bytes(case_order)),
        "cases": cases,
        "manifest_id": MANIFEST_ID,
        "parser_contract": {"format_magic_ascii": MAGIC.decode("ascii"), "format_version": VERSION, "rejection_type": "EnvelopeFormatError", "stable_error_codes": sorted(codes)},
        "schema_version": SCHEMA_VERSION,
        "source_authorities": [{"path": path, "sha256": sha256_file(root / path)} for path in _SOURCE_AUTHORITY_PATHS],
        "vector_set_sha256": _sha256(canonical_json_bytes(vector_set)),
    }
    schema = _manifest_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)
    files[MANIFEST_RELATIVE_PATH] = _human_json_bytes(manifest)
    return dict(sorted(files.items()))


def write_malformed_object_files(repo_root: str | Path) -> dict[str, bytes]:
    root = Path(repo_root).resolve()
    files = build_malformed_object_files(root)
    reference_root = _safe_path(root, "reference/malformed_objects")
    expected = {Path(relative) for relative in files}
    if reference_root.is_dir():
        for path in sorted(reference_root.rglob("*"), reverse=True):
            if path.is_file() and path.relative_to(root) not in expected:
                path.unlink()
        for path in sorted(reference_root.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    for relative, payload in files.items():
        target = _safe_path(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return files


def load_malformed_object_manifest(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest = json.loads(_safe_path(root, MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(_manifest_schema()).validate(manifest)
    return manifest


def check_malformed_object_files(repo_root: str | Path) -> list[str]:
    root = Path(repo_root).resolve()
    expected = build_malformed_object_files(root)
    errors: list[str] = []
    for relative, payload in expected.items():
        path = _safe_path(root, relative)
        if not path.is_file():
            errors.append(f"missing malformed-object file: {relative}")
        elif path.read_bytes() != payload:
            errors.append(f"malformed-object file differs from frozen executable reference: {relative}")
    reference_root = _safe_path(root, "reference/malformed_objects")
    actual = {path.relative_to(root).as_posix() for path in reference_root.rglob("*") if path.is_file()} if reference_root.is_dir() else set()
    for unexpected in sorted(actual - set(expected)):
        errors.append(f"unexpected malformed-object file: {unexpected}")
    return errors


def verify_malformed_object_assets(repo_root: str | Path) -> list[str]:
    root = Path(repo_root).resolve()
    errors = check_malformed_object_files(root)
    try:
        manifest = load_malformed_object_manifest(root)
        for source in manifest["source_authorities"]:
            if sha256_file(root / source["path"]) != source["sha256"]:
                errors.append(f"malformed-object source authority changed: {source['path']}")
        for case in manifest["cases"]:
            payload = _safe_path(root, case["object_path"]).read_bytes()
            if len(payload) != case["object_bytes"] or _sha256(payload) != case["sha256"]:
                errors.append(f"malformed-object vector identity mismatch: {case['case_id']}")
                continue
            try:
                parse_envelope(payload)
            except EnvelopeFormatError as exc:
                if exc.code != case["expected_error_code"]:
                    errors.append(f"malformed-object error-code mismatch: {case['case_id']}")
            except Exception as exc:
                errors.append(f"malformed-object exception mismatch for {case['case_id']}: {type(exc).__name__}")
            else:
                errors.append(f"malformed-object vector was accepted: {case['case_id']}")
    except Exception as exc:
        errors.append(f"malformed-object verification failed: {type(exc).__name__}: {exc}")
    return errors


def malformed_object_summary(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest = load_malformed_object_manifest(root)
    categories: dict[str, int] = {}
    for case in manifest["cases"]:
        categories[case["category"]] = categories.get(case["category"], 0) + 1
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_file(root / MANIFEST_RELATIVE_PATH),
        "case_count": manifest["case_count"],
        "category_counts": dict(sorted(categories.items())),
        "stable_error_code_count": len(manifest["parser_contract"]["stable_error_codes"]),
        "vector_set_sha256": manifest["vector_set_sha256"],
    }
