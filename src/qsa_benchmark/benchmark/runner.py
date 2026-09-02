from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .config import BenchmarkConfig
from .datasets import build_corpora, load_manifest, write_manifests
from .instrumentation import measure
from .metrics import exact_psnr, object_metrics
from .models import DatasetRecord, RunContext
from .registry import make_method, method_registry
from .tamper import tamper_object
from .utils import canonical_json_bytes, deterministic_seed, environment_metadata, sha256_file

NONCE_LENGTHS = {"B03_shake_hmac": 16, "B15_geometry_shake_hmac": 16}


def _nonce(method_id: str, benchmark_id: str, image_id: str, repetition: int, seed: int) -> bytes:
    length = NONCE_LENGTHS.get(method_id, 12)
    h = hashlib.shake_256(); h.update(b"QSA-BENCHMARK-NONCE-V1")
    for value in (benchmark_id, method_id, image_id, str(repetition), str(seed)):
        blob = value.encode("utf-8"); h.update(len(blob).to_bytes(4, "big")); h.update(blob)
    return h.digest(length)


def prepare_data(config: BenchmarkConfig, repo_root: Path) -> Path:
    size = tuple(int(value) for value in config.payload["dataset"]["image_size"])
    corpus_root = repo_root / "data/generated" / f"{size[0]}x{size[1]}"
    manifest_dir = repo_root / "data/manifests" / f"{size[0]}x{size[1]}"
    records = build_corpora(corpus_root, size=size)
    write_manifests(records, manifest_dir, size=size)
    return manifest_dir / "dataset_manifest.csv"


def select_records(config: BenchmarkConfig, manifest_path: Path) -> list[DatasetRecord]:
    records = load_manifest(manifest_path)
    splits = set(config.payload["dataset"]["splits"])
    corpora = set(config.payload["dataset"]["corpora"])
    limit = int(config.payload["dataset"]["max_records_per_corpus"])
    selected: list[DatasetRecord] = []
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record.split not in splits or record.corpus not in corpora:
            continue
        if counts[record.corpus] >= limit:
            continue
        selected.append(record); counts[record.corpus] += 1
    if not selected:
        raise ValueError("configuration selected no dataset records")
    return selected


def _load_image(record: DatasetRecord, repo_root: Path) -> np.ndarray:
    path = Path(record.path)
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists() or sha256_file(path) != record.sha256:
        raise ValueError(f"dataset hash mismatch: {record.image_id}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _write_dict_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0])); writer.writeheader(); writer.writerows(materialized)


def summarize_results(rows: list[dict[str, Any]], tamper_rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], public_recoverable: bool) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tamper_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_method[row["method_id"]].append(row)
    for row in tamper_rows: tamper_by[row["method_id"]].append(row)
    methods: dict[str, Any] = {}
    for method_id in sorted(by_method):
        group = by_method[method_id]; tamper = tamper_by.get(method_id, [])
        methods[method_id] = {
            **metadata[method_id],
            "runs": len(group),
            "all_exact": all(bool(row["exact_recovery"]) for row in group),
            "median_encrypt_ms": float(np.median([row["encrypt_ns"] for row in group]) / 1e6),
            "median_decrypt_ms": float(np.median([row["decrypt_ns"] for row in group]) / 1e6),
            "median_entropy": float(np.median([row["ciphertext_entropy"] for row in group])),
            "median_adjacent_correlation": float(np.median([row["adjacent_byte_correlation"] for row in group])),
            "median_expansion_ratio": float(np.median([row["expansion_ratio"] for row in group])),
            "tamper_accept_count": sum(bool(row["tamper_accepted"]) for row in tamper),
            "tamper_probe_count": len(tamper),
        }
    authenticated = [mid for mid, item in methods.items() if item["authenticated"]]
    unauthenticated = [mid for mid, item in methods.items() if not item["authenticated"]]
    conditions = {
        "all_20_methods_present": len(methods) == 20,
        "all_round_trips_exact": all(item["all_exact"] for item in methods.values()),
        "authenticated_methods_reject_all_tampering": all(methods[mid]["tamper_accept_count"] == 0 for mid in authenticated),
        "unauthenticated_methods_expose_malleability": all(methods[mid]["tamper_accept_count"] > 0 for mid in unauthenticated),
        "public_high_entropy_control_is_publicly_recoverable": bool(public_recoverable),
        "unique_nonces_within_method": len({(row["method_id"], row["nonce_hex"]) for row in rows}) == len(rows),
    }
    validation = {"conditions": conditions, "passed": all(conditions.values()), "decision": "PASS" if all(conditions.values()) else "FAIL"}
    return {"method_count": len(methods), "run_count": len(rows), "tamper_probe_count": len(tamper_rows), "methods": methods, "validation": validation}


def run_benchmark(config: BenchmarkConfig, repo_root: Path) -> dict[str, Path]:
    size = tuple(int(value) for value in config.payload["dataset"]["image_size"] )
    manifest_path = repo_root / "data/manifests" / f"{size[0]}x{size[1]}" / "dataset_manifest.csv"
    if not manifest_path.exists(): manifest_path = prepare_data(config, repo_root)
    records = select_records(config, manifest_path)
    output_root = config.output_root; output_root.mkdir(parents=True, exist_ok=True)
    objects_dir = output_root / "objects"; recon_dir = output_root / "reconstructions"
    if config.payload["outputs"]["write_objects"]: objects_dir.mkdir(parents=True, exist_ok=True)
    if config.payload["outputs"]["write_reconstructions"]: recon_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []; tamper_rows: list[dict[str, Any]] = []
    nonce_registry: set[tuple[str, bytes]] = set()
    metadata = {item["method_id"]: item for item in method_registry()}
    public_recoverable = True
    for record in records:
        image = _load_image(record, repo_root)
        for method_id in config.methods:
            method = make_method(method_id)
            for repetition in range(int(config.payload["execution"]["repetitions"])):
                seed = deterministic_seed(config.master_seed, config.benchmark_id, record.image_id, method_id, repetition)
                nonce = _nonce(method_id, config.benchmark_id, record.image_id, repetition, seed)
                if (method_id, nonce) in nonce_registry: raise RuntimeError("nonce collision inside configured benchmark")
                nonce_registry.add((method_id, nonce))
                run_id = f"{config.benchmark_id}|{record.image_id}|{method_id}|r{repetition}"
                context = RunContext(config.master_key, nonce, seed, record.image_id, method_id, run_id)
                encrypted = measure(lambda: method.encrypt(image, context))
                decrypted = measure(lambda: method.decrypt(encrypted.value.object_bytes, context))
                exact = bool(np.array_equal(image, decrypted.value))
                metrics = object_metrics(encrypted.value.ciphertext_view, encrypted.value.object_bytes, image.nbytes)
                result_rows.append({
                    "benchmark_id": config.benchmark_id, "run_id": run_id, "image_id": record.image_id,
                    "split": record.split, "corpus": record.corpus, "method_id": method_id,
                    "method_family": method.family, "authenticated": method.authenticated,
                    "secure_control": method.secure_control, "repetition": repetition, "seed": seed,
                    "nonce_hex": nonce.hex(), "exact_recovery": exact,
                    "reconstruction_psnr_db": "inf" if exact else exact_psnr(image, decrypted.value),
                    "encrypt_ns": encrypted.elapsed_ns, "decrypt_ns": decrypted.elapsed_ns,
                    "encrypt_peak_tracemalloc_bytes": encrypted.peak_tracemalloc_bytes,
                    "decrypt_peak_tracemalloc_bytes": decrypted.peak_tracemalloc_bytes,
                    "encrypt_rss_delta_bytes": encrypted.rss_after_bytes - encrypted.rss_before_bytes,
                    "decrypt_rss_delta_bytes": decrypted.rss_after_bytes - decrypted.rss_before_bytes,
                    "object_sha256": hashlib.sha256(encrypted.value.object_bytes).hexdigest(), **metrics,
                })
                if method_id == "B04_public_high_entropy":
                    alternate = RunContext(bytes(32), nonce, seed, record.image_id, method_id, run_id)
                    public_recoverable &= np.array_equal(image, method.decrypt(encrypted.value.object_bytes, alternate))
                if config.payload["outputs"]["write_objects"]:
                    (objects_dir / f"{record.image_id}__{method_id}__r{repetition}.qsb").write_bytes(encrypted.value.object_bytes)
                if config.payload["outputs"]["write_reconstructions"]:
                    Image.fromarray(decrypted.value, mode="RGB").save(recon_dir / f"{record.image_id}__{method_id}__r{repetition}.png")
                if config.payload["execution"]["tamper_probe"]:
                    tampered = tamper_object(encrypted.value.object_bytes)
                    accepted = False; changed = False; error_type = ""
                    try:
                        tampered_image = method.decrypt(tampered, context)
                        accepted = True; changed = not np.array_equal(image, tampered_image)
                    except Exception as exc:  # controlled negative test
                        error_type = type(exc).__name__
                    tamper_rows.append({"run_id": run_id, "image_id": record.image_id, "method_id": method_id, "authenticated": method.authenticated, "tamper_accepted": accepted, "accepted_plaintext_changed": changed, "rejection_error_type": error_type})
    result_path = output_root / "benchmark_results.csv"; tamper_path = output_root / "tamper_results.csv"
    _write_dict_rows(result_path, result_rows); _write_dict_rows(tamper_path, tamper_rows)
    nondeterministic = {"encrypt_ns", "decrypt_ns", "encrypt_peak_tracemalloc_bytes", "decrypt_peak_tracemalloc_bytes", "encrypt_rss_delta_bytes", "decrypt_rss_delta_bytes"}
    deterministic_rows = [{key: value for key, value in row.items() if key not in nondeterministic} for row in result_rows]
    deterministic_path = output_root / "deterministic_results.csv"; _write_dict_rows(deterministic_path, deterministic_rows)
    summary = summarize_results(result_rows, tamper_rows, metadata, public_recoverable)
    summary_path = output_root / "benchmark_summary.json"; summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    validation_path = output_root / "validation.json"; validation_path.write_text(json.dumps(summary["validation"], indent=2, sort_keys=True) + "\n")
    fingerprint_payload = {"benchmark_id": config.benchmark_id, "dataset_manifest_sha256": sha256_file(manifest_path), "deterministic_rows": deterministic_rows, "tamper_rows": tamper_rows, "validation": summary["validation"]}
    fingerprint = hashlib.sha256(canonical_json_bytes(fingerprint_payload)).hexdigest()
    fingerprint_path = output_root / "deterministic_fingerprint.json"; fingerprint_path.write_text(json.dumps({"sha256": fingerprint, "run_count": len(deterministic_rows)}, indent=2, sort_keys=True) + "\n")
    registry_path = output_root / "method_registry.json"; registry_path.write_text(json.dumps(method_registry(), indent=2, sort_keys=True) + "\n")
    environment = {**environment_metadata(), "numpy": np.__version__, "cryptography": importlib.metadata.version("cryptography"), "pillow": importlib.metadata.version("pillow"), "scikit_image": importlib.metadata.version("scikit-image"), "scipy": importlib.metadata.version("scipy"), "psutil": importlib.metadata.version("psutil"), "pyyaml": importlib.metadata.version("PyYAML"), "jsonschema": importlib.metadata.version("jsonschema"), "config_sha256": sha256_file(config.path), "manifest_sha256": sha256_file(manifest_path), "cwd": os.getcwd(), "platform_release": platform.release(), "executable": sys.executable}
    environment_path = output_root / "environment.json"; environment_path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    execution_path = output_root / "execution_manifest.json"
    outputs = [result_path, deterministic_path, tamper_path, summary_path, validation_path, fingerprint_path, registry_path, environment_path]
    execution = {"benchmark_id": config.benchmark_id, "stages": ["prepare-data", "execute-constructions", "tamper-calibration", "summarize", "validate"], "inputs": {"config": str(config.path.relative_to(repo_root)), "config_sha256": sha256_file(config.path), "dataset_manifest": str(manifest_path.relative_to(repo_root)), "dataset_manifest_sha256": sha256_file(manifest_path)}, "outputs": {path.name: sha256_file(path) for path in outputs}}
    execution_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
    return {"results": result_path, "tamper": tamper_path, "deterministic_results": deterministic_path, "fingerprint": fingerprint_path, "summary": summary_path, "validation": validation_path, "registry": registry_path, "environment": environment_path, "execution_manifest": execution_path, "dataset_manifest": manifest_path}
