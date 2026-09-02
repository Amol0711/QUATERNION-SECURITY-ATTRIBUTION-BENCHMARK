from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import numpy as np
from PIL import Image

from qsa_benchmark.benchmark.crypto import (
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    shake_hmac_decrypt,
    shake_hmac_encrypt,
)
from qsa_benchmark.benchmark.datasets import build_corpora
from qsa_benchmark.benchmark.models import RunContext, TransformOutput
from qsa_benchmark.benchmark.transforms import CaseIITransform
from qsa_benchmark.benchmark.utils import canonical_json_bytes

CONFIG_PATH = "configs/protocol/primitive_scaling.json"
SCHEMA_PATH = "configs/protocol/primitive_scaling.schema.json"
ROUND_TRIP_FIELDS = (
    "run_label", "case_ordinal", "primitive", "image_id", "image_size",
    "body_representation", "body_bytes", "body_length_ratio", "input_sha256",
    "body_sha256", "recovered_body_sha256", "recovered_image_sha256", "round_trip_exact",
)
TIMING_FIELDS = (
    "run_label", "primitive", "image_id", "image_size", "body_representation",
    "direction", "repetition", "body_bytes", "elapsed_ns",
)
AGGREGATE_FIELDS = (
    "run_label", "primitive", "image_size", "image_id", "direction",
    "raw_body_bytes", "expanded_body_bytes", "body_length_ratio",
    "raw_primitive_median_ns", "expanded_primitive_median_ns", "primitive_time_ratio",
    "rho_times_raw_prediction_ns", "linear_scaling_relative_residual",
)
SUMMARY_FIELDS = (
    "run_label", "primitive", "direction", "configurations", "median_body_ratio",
    "median_time_ratio", "median_linear_scaling_relative_residual",
    "p95_linear_scaling_relative_residual", "maximum_linear_scaling_relative_residual",
)


def load_primitive_config(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    payload = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    schema = json.loads((root / SCHEMA_PATH).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(payload)
    if len(payload["primitives"]) != 2 or len(payload["body_representations"]) != 2:
        raise ValueError("primitive-scaling factorization must contain two primitives and two body representations")
    expected = payload["expected_counts"]
    cases = len(payload["primitives"]) * len(payload["image_ids"]) * len(payload["sizes"]) * len(payload["body_representations"])
    if cases != expected["round_trip_cases_per_execution"]:
        raise ValueError("primitive-scaling round-trip denominator is inconsistent")
    if cases * 2 * int(payload["timing_repetitions"]) != expected["timing_records_per_execution"]:
        raise ValueError("primitive-scaling timing denominator is inconsistent")
    return payload


def _write_rows(path: Path, fields: tuple[str, ...], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("true" if value is True else "false" if value is False else value) for key, value in row.items()})


def _derive_bytes(domain: str, *values: object, length: int) -> bytes:
    h = hashlib.shake_256()
    h.update(b"QSA-PRIMITIVE-SCALING-CONTEXT-V1")
    h.update(domain.encode("ascii"))
    for value in values:
        encoded = str(value).encode("utf-8")
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
    return h.digest(length)


def _context(config: Mapping[str, Any], primitive: str, image_id: str, size: int, body_id: str) -> RunContext:
    nonce_length = next(int(item["nonce_bytes"]) for item in config["primitives"] if item["id"] == primitive)
    seed = int.from_bytes(_derive_bytes("seed", primitive, image_id, size, body_id, length=8), "big")
    return RunContext(
        master_key=bytes.fromhex(str(config["master_key_hex"])),
        nonce=_derive_bytes("nonce", primitive, image_id, size, body_id, length=nonce_length),
        seed=seed,
        image_id=image_id,
        method_id="primitive_scaling",
        run_id="primitive_scaling",
        public_metadata={"body_representation": body_id, "image_size": int(size)},
        public_material=b"",
        protocol_id=str(config["protocol_id"]),
    )


def _load_images(repo_root: Path, config: Mapping[str, Any], workspace: Path) -> dict[tuple[str, int], np.ndarray]:
    images: dict[tuple[str, int], np.ndarray] = {}
    for size_value in config["sizes"]:
        size = int(size_value)
        generated = workspace / "data" / f"{size}x{size}"
        records = build_corpora(generated, size=(size, size))
        by_id = {record.image_id: record for record in records}
        for image_id in config["image_ids"]:
            path = Path(by_id[image_id].path)
            images[(image_id, size)] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return images


def _body(image: np.ndarray, body_id: str, context: RunContext) -> tuple[bytes, object | None]:
    if body_id == "raw_rgb_uint8":
        return image.tobytes(order="C"), None
    if body_id == "caseii_int32_le":
        transform = CaseIITransform(False)
        output = transform.forward(image, context)
        return output.payload, output
    raise ValueError(f"unsupported body representation: {body_id}")


def _recover_image(payload: bytes, body_id: str, image: np.ndarray, descriptor: object | None, context: RunContext) -> np.ndarray:
    if body_id == "raw_rgb_uint8":
        return np.frombuffer(payload, dtype=np.uint8).reshape(image.shape).copy()
    if body_id == "caseii_int32_le":
        if not isinstance(descriptor, TransformOutput):
            raise TypeError("Case-II descriptor is missing")
        output = TransformOutput(payload, descriptor.descriptor, descriptor.shape)
        return CaseIITransform(False).inverse(output, context)
    raise ValueError(body_id)


def _encrypt(primitive: str, payload: bytes, aad: bytes, context: RunContext) -> tuple[bytes, bytes]:
    if primitive == "AES-256-GCM":
        return aes_gcm_encrypt(payload, aad, context, "primitive-scaling")
    if primitive == "SHAKE-256/HMAC-SHA-256":
        return shake_hmac_encrypt(payload, aad, context, "primitive-scaling")
    raise ValueError(primitive)


def _decrypt(primitive: str, ciphertext: bytes, tag: bytes, aad: bytes, context: RunContext) -> bytes:
    if primitive == "AES-256-GCM":
        return aes_gcm_decrypt(ciphertext, tag, aad, context, "primitive-scaling")
    if primitive == "SHAKE-256/HMAC-SHA-256":
        return shake_hmac_decrypt(ciphertext, tag, aad, context, "primitive-scaling")
    raise ValueError(primitive)


def execute_primitive_scaling(
    repo_root: str | Path,
    output_root: str | Path,
    *,
    run_label: str,
    timing: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_primitive_config(root)
    images = _load_images(root, config, output / "workspace")
    round_trip_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    case_ordinal = 0

    for primitive_entry in config["primitives"]:
        primitive = str(primitive_entry["id"])
        for image_id in config["image_ids"]:
            for size_value in config["sizes"]:
                size = int(size_value)
                image = images[(image_id, size)]
                raw_length = int(image.nbytes)
                for body_entry in config["body_representations"]:
                    body_id = str(body_entry["id"])
                    case_ordinal += 1
                    context = _context(config, primitive, image_id, size, body_id)
                    payload, descriptor = _body(image, body_id, context)
                    aad = canonical_json_bytes({"primitive": primitive, "image_id": image_id, "image_size": size, "body_representation": body_id})
                    ciphertext, tag = _encrypt(primitive, payload, aad, context)
                    recovered_body = _decrypt(primitive, ciphertext, tag, aad, context)
                    recovered_image = _recover_image(recovered_body, body_id, image, descriptor, context)
                    exact = recovered_body == payload and np.array_equal(recovered_image, image)
                    round_trip_rows.append({
                        "run_label": run_label,
                        "case_ordinal": case_ordinal,
                        "primitive": primitive,
                        "image_id": image_id,
                        "image_size": size,
                        "body_representation": body_id,
                        "body_bytes": len(payload),
                        "body_length_ratio": format(len(payload) / raw_length, ".17g"),
                        "input_sha256": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
                        "body_sha256": hashlib.sha256(payload).hexdigest(),
                        "recovered_body_sha256": hashlib.sha256(recovered_body).hexdigest(),
                        "recovered_image_sha256": hashlib.sha256(recovered_image.tobytes(order="C")).hexdigest(),
                        "round_trip_exact": bool(exact),
                    })
                    if not exact:
                        raise RuntimeError(f"primitive round trip failed for {primitive}, {image_id}, {size}, {body_id}")
                    if timing:
                        repetitions = int(config["timing_repetitions"])
                        for repetition in range(repetitions):
                            start = time.perf_counter_ns()
                            timed_ciphertext, timed_tag = _encrypt(primitive, payload, aad, context)
                            elapsed = time.perf_counter_ns() - start
                            timing_rows.append({"run_label":run_label,"primitive":primitive,"image_id":image_id,"image_size":size,"body_representation":body_id,"direction":"encrypt","repetition":repetition,"body_bytes":len(payload),"elapsed_ns":elapsed})
                            start = time.perf_counter_ns()
                            timed_recovered = _decrypt(primitive, timed_ciphertext, timed_tag, aad, context)
                            elapsed = time.perf_counter_ns() - start
                            if timed_recovered != payload:
                                raise RuntimeError("timed primitive recovery mismatch")
                            timing_rows.append({"run_label":run_label,"primitive":primitive,"image_id":image_id,"image_size":size,"body_representation":body_id,"direction":"decrypt","repetition":repetition,"body_bytes":len(payload),"elapsed_ns":elapsed})

    expected = config["expected_counts"]
    if len(round_trip_rows) != int(expected["round_trip_cases_per_execution"]):
        raise RuntimeError("primitive round-trip row count mismatch")
    _write_rows(output / "primitive_round_trip_records.csv", ROUND_TRIP_FIELDS, round_trip_rows)

    aggregate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    if timing:
        if len(timing_rows) != int(expected["timing_records_per_execution"]):
            raise RuntimeError("primitive timing row count mismatch")
        _write_rows(output / "primitive_timing_records.csv", TIMING_FIELDS, timing_rows)
        groups: dict[tuple[str, str, int, str], dict[str, list[int]]] = {}
        for row in timing_rows:
            key = (str(row["primitive"]), str(row["image_id"]), int(row["image_size"]), str(row["direction"]))
            groups.setdefault(key, {}).setdefault(str(row["body_representation"]), []).append(int(row["elapsed_ns"]))
        for (primitive, image_id, size, direction), values in sorted(groups.items()):
            raw_times = values["raw_rgb_uint8"]
            expanded_times = values["caseii_int32_le"]
            raw_bytes = size * size * 3
            expanded_bytes = raw_bytes * 4
            raw_median = float(statistics.median(raw_times))
            expanded_median = float(statistics.median(expanded_times))
            predicted = 4.0 * raw_median
            residual = abs(expanded_median - predicted) / max(expanded_median, predicted, 1.0)
            aggregate_rows.append({
                "run_label":run_label,"primitive":primitive,"image_size":size,"image_id":image_id,"direction":direction,
                "raw_body_bytes":raw_bytes,"expanded_body_bytes":expanded_bytes,"body_length_ratio":4,
                "raw_primitive_median_ns":format(raw_median,".17g"),"expanded_primitive_median_ns":format(expanded_median,".17g"),
                "primitive_time_ratio":format(expanded_median/max(raw_median,1.0),".17g"),
                "rho_times_raw_prediction_ns":format(predicted,".17g"),"linear_scaling_relative_residual":format(residual,".17g"),
            })
        if len(aggregate_rows) != int(expected["timing_aggregate_rows_per_execution"]):
            raise RuntimeError("primitive timing aggregate row count mismatch")
        _write_rows(output / "primitive_scaling_aggregate_records.csv", AGGREGATE_FIELDS, aggregate_rows)
        for primitive in [str(item["id"]) for item in config["primitives"]]:
            for direction in ("decrypt", "encrypt"):
                subset = [row for row in aggregate_rows if row["primitive"] == primitive and row["direction"] == direction]
                ratios = [float(row["primitive_time_ratio"]) for row in subset]
                residuals = sorted(float(row["linear_scaling_relative_residual"]) for row in subset)
                q95_index = max(0, min(len(residuals)-1, int(np.ceil(0.95*len(residuals)))-1))
                summary_rows.append({
                    "run_label":run_label,"primitive":primitive,"direction":direction,"configurations":len(subset),
                    "median_body_ratio":4,"median_time_ratio":format(statistics.median(ratios),".17g"),
                    "median_linear_scaling_relative_residual":format(statistics.median(residuals),".17g"),
                    "p95_linear_scaling_relative_residual":format(residuals[q95_index],".17g"),
                    "maximum_linear_scaling_relative_residual":format(max(residuals),".17g"),
                })
        _write_rows(output / "primitive_scaling_summary.csv", SUMMARY_FIELDS, summary_rows)

    manifest = {
        "protocol_id": config["protocol_id"],
        "run_label": run_label,
        "round_trip_cases": len(round_trip_rows),
        "round_trip_cases_exact": sum(bool(row["round_trip_exact"]) for row in round_trip_rows),
        "timing_enabled": bool(timing),
        "timing_records": len(timing_rows),
        "timing_aggregate_rows": len(aggregate_rows),
    }
    (output / "primitive_scaling_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    return manifest
