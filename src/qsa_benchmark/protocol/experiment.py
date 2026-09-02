from __future__ import annotations

import argparse
import cProfile
import csv
import gc
import hashlib
import io
import json
import math
import os
import pstats
import resource
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import bdtr
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from qsa_benchmark.attacks.attacks import affine_diffusion_recovery, recover_permutation
from qsa_benchmark.attacks.metrics import reconstruction_metrics
from qsa_benchmark.benchmark.envelope import EnvelopeFormatError, parse_envelope
from qsa_benchmark.benchmark.metrics import adjacent_byte_correlation, byte_entropy
from qsa_benchmark.benchmark.models import RunContext
from qsa_benchmark.benchmark.registry import make_method
from qsa_benchmark.benchmark.serialization import bytes_to_image
from qsa_benchmark.benchmark.utils import canonical_json_bytes, derive_key, sha256_file
from qsa_benchmark.validation.active_modification import (
    generate_active_mutations,
    reencode_envelope,
)

from .config import ProtocolConfig, load_protocol_config, repository_root
from .differential import (
    PERTURBATIONS,
    apply_perturbation,
    holm_rejections,
    ideal_uaci_moments,
    npcr_uaci,
    pairwise_flip_probability,
    planned_pair_counts,
)
from .leakage import leakage_entropy_rows
from .models import DifferentialProtocol
from .registry import POLICIES, method_policy, operation_regime, protocol_method_registry
from .schedule import build_context_pair, public_material_length
from .timing import applicable_stages, fit_serial_cost_model
from .views import body_projections, extract_protocol_views

EXPERIMENT_VERSION = "QSA-EXPERIMENT-V1"
FORBIDDEN_KEYS = {
    "image_id", "run_id", "pair_id", "key_index", "state_index", "perturbation_id"
}
CSV_FLOAT_FORMAT = "%.17g"


def _reported_attack_error_type(exc: Exception) -> str:
    """Preserve the public experiment table's historical exception class."""
    if isinstance(exc, EnvelopeFormatError):
        return "ValueError"
    return type(exc).__name__


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> int:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not materialized:
            path.write_text("", encoding="utf-8")
            return 0
        fieldnames = list(materialized[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in materialized:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
                elif isinstance(value, float):
                    if math.isnan(value):
                        value = "NA"
                    elif math.isinf(value):
                        value = "inf" if value > 0 else "-inf"
                    else:
                        value = format(value, ".17g")
                elif isinstance(value, np.generic):
                    value = value.item()
                out[key] = value
            writer.writerow(out)
    return len(materialized)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_csv_sha256(path: Path, *, excluded_columns: set[str] | None = None) -> str:
    excluded = excluded_columns or set()
    rows = read_rows(path)
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append({key: row[key] for key in sorted(row) if key not in excluded})
    normalized.sort(key=lambda row: canonical_json_bytes(row))
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def recursive_forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if str(key) in FORBIDDEN_KEYS:
                found.append(path)
            found.extend(recursive_forbidden_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(recursive_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def _ensure_size_data(repo_root: Path, size: int) -> Path:
    from qsa_benchmark.benchmark.datasets import build_corpora, write_manifests

    manifest_dir = repo_root / "data/manifests" / f"{size}x{size}"
    manifest_path = manifest_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        corpus_root = repo_root / "data/generated" / f"{size}x{size}"
        records = build_corpora(corpus_root, size=(size, size))
        write_manifests(records, manifest_dir, size=(size, size))
    return manifest_path


def load_manifest(repo_root: Path) -> tuple[list[dict[str, str]], dict[tuple[str, int], np.ndarray], dict[tuple[str, int], str]]:
    rows: list[dict[str, str]] = []
    for size in (256, 512):
        rows.extend(read_rows(_ensure_size_data(repo_root, size)))
    if len(rows) != 48:
        raise RuntimeError(f"expected 48 standard-corpus rows, found {len(rows)}")
    images: dict[tuple[str, int], np.ndarray] = {}
    corpus: dict[tuple[str, int], str] = {}
    for row in rows:
        path = repo_root / row["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise RuntimeError(f"manifest digest mismatch for {row['image_id']} {row['height']}: {digest}")
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        size = int(row["height"])
        if image.shape != (size, size, 3):
            raise RuntimeError(f"shape mismatch for {row['image_id']} {size}: {image.shape}")
        key = (row["image_id"], size)
        images[key] = image
        corpus[key] = row["corpus"]
    return rows, images, corpus


def load_timing_images(repo_root: Path, config: Mapping[str, Any]) -> dict[tuple[str, int], np.ndarray]:
    _, standard, _ = load_manifest(repo_root)
    output: dict[tuple[str, int], np.ndarray] = {}
    manifest_96 = read_rows(_ensure_size_data(repo_root, 96))
    paths_96 = {row["image_id"]: repo_root / row["path"] for row in manifest_96}
    for image_id in config["corpus"]["timing_panel"]:
        for size_value in config["timing"]["sizes"]:
            size = int(size_value)
            if size in {256, 512}:
                output[(image_id, size)] = standard[(image_id, size)]
            elif size == 96:
                path = paths_96[image_id]
                output[(image_id, size)] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            else:
                raise RuntimeError(f"unsupported timing size: {size}")
    return output

def stable_ordinal(*values: object) -> int:
    h = hashlib.sha256()
    h.update(b"QSA-EXPERIMENT-ORDINAL-V1")
    for value in values:
        blob = str(value).encode("utf-8")
        h.update(len(blob).to_bytes(4, "big")); h.update(blob)
    return int.from_bytes(h.digest()[:4], "big")


def _protocol_enum(name: str) -> DifferentialProtocol:
    if name == "P1_common_context":
        return DifferentialProtocol.P1_COMMON_CONTEXT
    if name == "P2_fresh_randomness":
        return DifferentialProtocol.P2_FRESH_RANDOMNESS
    raise ValueError(name)


def _metric_raw(left: bytes, right: bytes) -> tuple[int, int, float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("equal nonempty projections required")
    a = np.frombuffer(left, dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(right, dtype=np.uint8).astype(np.int16)
    changed = int(np.count_nonzero(a != b))
    abs_sum = int(np.sum(np.abs(a - b), dtype=np.int64))
    n = len(a)
    return n, changed, changed / n, abs_sum / (255.0 * n)


def differential_schedule_for_method(config: ProtocolConfig, method_id: str) -> Iterable[dict[str, Any]]:
    payload = config.payload
    policy = method_policy(method_id)
    for tier_index, (tier_id, tier) in enumerate(payload["execution_tiers"].items()):
        methods = payload["methods"] if tier["methods"] == "all" else tier["methods"]
        if method_id not in methods:
            continue
        images = payload["corpus"][tier["images"]]
        perturbations = (
            payload["perturbations"][tier["perturbations"]]
            if isinstance(tier["perturbations"], str)
            else tier["perturbations"]
        )
        for protocol_index, protocol_name in enumerate(tier["protocols"]):
            if protocol_name == "P2_fresh_randomness" and not policy.p2_applicable:
                continue
            for size in tier["sizes"]:
                states = int(tier["state_repetitions_by_size"][str(size)])
                for image_index, image_id in enumerate(images):
                    for perturbation_index, perturbation_id in enumerate(perturbations):
                        for key_index in range(int(tier["key_repetitions"])):
                            for state_index in range(states):
                                ordinal = stable_ordinal(
                                    tier_id, method_id, protocol_name, size, image_id,
                                    perturbation_id, key_index, state_index,
                                )
                                yield {
                                    "tier_id": tier_id,
                                    "tier_index": tier_index,
                                    "method_id": method_id,
                                    "protocol": protocol_name,
                                    "protocol_index": protocol_index,
                                    "image_size": int(size),
                                    "image_id": image_id,
                                    "perturbation_id": perturbation_id,
                                    "key_index": key_index,
                                    "state_index": state_index,
                                    "pair_ordinal": ordinal,
                                    "image_index": image_index,
                                    "perturbation_index": perturbation_index,
                                }


def run_differential_shard(repo_root: Path, run_root: Path, method_id: str) -> dict[str, Any]:
    config = load_protocol_config()
    if method_id not in config.methods:
        raise KeyError(method_id)
    _, images, corpus = load_manifest(repo_root)
    method = make_method(method_id, profile="extended")
    pair_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    perturbation_cache: dict[tuple[str, int, str], np.ndarray] = {}
    start = time.time()
    for index, spec in enumerate(differential_schedule_for_method(config, method_id)):
        key = (spec["image_id"], spec["image_size"])
        image = images[key]
        pkey = (spec["image_id"], spec["image_size"], spec["perturbation_id"])
        changed = perturbation_cache.get(pkey)
        if changed is None:
            changed = apply_perturbation(image, spec["perturbation_id"])
            perturbation_cache[pkey] = changed
        pair = build_context_pair(
            master_key=config.master_key,
            master_seed=config.master_seed,
            method_id=method_id,
            protocol=_protocol_enum(spec["protocol"]),
            image_id=spec["image_id"],
            perturbation_id=spec["perturbation_id"],
            pair_ordinal=spec["pair_ordinal"],
            key_index=spec["key_index"],
            state_index=spec["state_index"],
            payload_length=image.nbytes,
            protocol_id=config.protocol_id,
        )
        left = method.encrypt(image, pair.left)
        right = method.encrypt(changed, pair.right)
        left_views = extract_protocol_views(left.object_bytes, method_id)
        right_views = extract_protocol_views(right.object_bytes, method_id)
        header_paths = recursive_forbidden_paths(parse_envelope(left.object_bytes).header)
        header_paths += recursive_forbidden_paths(parse_envelope(right.object_bytes).header)
        if header_paths:
            raise RuntimeError(f"forbidden object metadata in {method_id}: {header_paths}")
        left_projections = body_projections(left_views)
        right_projections = body_projections(right_views)
        if set(left_projections) != set(right_projections):
            raise RuntimeError(f"projection mismatch for {method_id}")
        pair_id = (
            f"{spec['tier_id']}|{method_id}|{spec['protocol']}|{spec['image_size']}|"
            f"{spec['image_id']}|{spec['perturbation_id']}|k{spec['key_index']}|s{spec['state_index']}"
        )
        agg_left = left_projections["aggregate_full_body"][0]
        agg_right = right_projections["aggregate_full_body"][0]
        n, changed_count, npcr, uaci = _metric_raw(agg_left, agg_right)
        pair_rows.append({
            "pair_id": pair_id,
            "tier_id": spec["tier_id"],
            "method_id": method_id,
            "protocol": spec["protocol"],
            "protocol_code": "P1-CC" if spec["protocol"].startswith("P1") else "P2-FR",
            "operation_regime": pair.operation_regime,
            "corpus": corpus[key],
            "image_id": spec["image_id"],
            "image_size": spec["image_size"],
            "perturbation_id": spec["perturbation_id"],
            "key_index": spec["key_index"],
            "state_index": spec["state_index"],
            "pair_ordinal": spec["pair_ordinal"],
            "same_master_key": pair.same_master_key,
            "same_nonce": pair.same_nonce,
            "same_seed": pair.same_seed,
            "same_public_material": pair.same_public_material,
            "body_metric_domain": left_views.body_metric_domain,
            "metric_body_source": left_views.metric_body_source,
            "sample_count": n,
            "changed_count": changed_count,
            "npcr": npcr,
            "uaci": uaci,
            "left_object_bytes": len(left.object_bytes),
            "right_object_bytes": len(right.object_bytes),
            "left_metric_body_bytes": len(left_views.metric_body),
            "right_metric_body_bytes": len(right_views.metric_body),
            "left_object_sha256": hashlib.sha256(left.object_bytes).hexdigest(),
            "right_object_sha256": hashlib.sha256(right.object_bytes).hexdigest(),
            "left_metric_body_sha256": hashlib.sha256(left_views.metric_body).hexdigest(),
            "right_metric_body_sha256": hashlib.sha256(right_views.metric_body).hexdigest(),
            "metadata_recursive_clean": True,
        })
        for projection_id in left_projections:
            left_body, eligible_left, semantic_left = left_projections[projection_id]
            right_body, eligible_right, semantic_right = right_projections[projection_id]
            if eligible_left != eligible_right or semantic_left != semantic_right:
                raise RuntimeError("projection metadata mismatch")
            sample_count, count, p_npcr, p_uaci = _metric_raw(left_body, right_body)
            projection_rows.append({
                "pair_id": pair_id,
                "tier_id": spec["tier_id"],
                "method_id": method_id,
                "protocol": spec["protocol"],
                "protocol_code": "P1-CC" if spec["protocol"].startswith("P1") else "P2-FR",
                "operation_regime": pair.operation_regime,
                "corpus": corpus[key],
                "image_id": spec["image_id"],
                "image_size": spec["image_size"],
                "perturbation_id": spec["perturbation_id"],
                "key_index": spec["key_index"],
                "state_index": spec["state_index"],
                "projection_id": projection_id,
                "semantic_domain": semantic_left,
                "inferential_eligible": bool(eligible_left),
                "sample_count": sample_count,
                "changed_count": count,
                "npcr": p_npcr,
                "uaci": p_uaci,
            })
        if (index + 1) % 250 == 0:
            print(f"[{method_id}] differential {index + 1} pairs", flush=True)
    shard = run_root / "differential/shards"
    pair_path = shard / f"{method_id}__pairs.csv"
    projection_path = shard / f"{method_id}__projections_raw.csv"
    write_rows(pair_path, pair_rows)
    write_rows(projection_path, projection_rows)
    manifest = {
        "method_id": method_id,
        "pair_rows": len(pair_rows),
        "projection_rows": len(projection_rows),
        "elapsed_seconds": time.time() - start,
        "pair_sha256": sha256_file(pair_path),
        "projection_sha256": sha256_file(projection_path),
    }
    write_json(shard / f"{method_id}__manifest.json", manifest)
    return manifest


def _apply_vectorized_inference(frame: pd.DataFrame, alpha: float = 0.01, q: int = 256) -> pd.DataFrame:
    out = frame.copy()
    eligible = out["inferential_eligible"].astype(str).str.lower().isin({"true", "1"})
    n = out["sample_count"].astype(int).to_numpy()
    count = out["changed_count"].astype(int).to_numpy()
    npcr_p = np.full(len(out), np.nan, dtype=float)
    uaci_p = np.full(len(out), np.nan, dtype=float)
    npcr_reject = np.zeros(len(out), dtype=bool)
    uaci_reject = np.zeros(len(out), dtype=bool)
    lower_count = np.full(len(out), -1, dtype=int)
    uaci_lower = np.full(len(out), np.nan, dtype=float)
    uaci_upper = np.full(len(out), np.nan, dtype=float)
    if eligible.any():
        idx = np.flatnonzero(eligible.to_numpy())
        p = 1.0 - 1.0 / q
        npcr_p[idx] = bdtr(count[idx], n[idx], p)
        # Exact lower critical count by unique sample count.
        from scipy.stats import binom
        cache: dict[int, int] = {}
        for nn in np.unique(n[idx]):
            candidate = int(binom.ppf(alpha, int(nn), p))
            while candidate >= 0 and float(binom.cdf(candidate, int(nn), p)) > alpha:
                candidate -= 1
            cache[int(nn)] = candidate
        lower_count[idx] = np.asarray([cache[int(nn)] for nn in n[idx]], dtype=int)
        npcr_reject[idx] = count[idx] <= lower_count[idx]
        mean, variance = ideal_uaci_moments(q)
        se = np.sqrt(variance / n[idx])
        critical = float(norm.ppf(1.0 - alpha / 2.0))
        lower = mean - critical * se
        upper = mean + critical * se
        vals = out.iloc[idx]["uaci"].astype(float).to_numpy()
        z = (vals - mean) / se
        uaci_p[idx] = np.minimum(1.0, 2.0 * norm.sf(np.abs(z)))
        uaci_lower[idx] = lower
        uaci_upper[idx] = upper
        uaci_reject[idx] = (vals < lower) | (vals > upper)
    out["alpha"] = alpha
    out["npcr_p_value"] = npcr_p
    out["uaci_p_value"] = uaci_p
    out["npcr_lower_count"] = lower_count
    out["uaci_lower"] = uaci_lower
    out["uaci_upper"] = uaci_upper
    out["npcr_raw_reject"] = npcr_reject
    out["uaci_raw_reject"] = uaci_reject
    out["npcr_holm_reject"] = False
    out["uaci_holm_reject"] = False
    grouping = ["method_id", "protocol", "image_size", "projection_id"]
    for _, indices in out.loc[eligible].groupby(grouping, sort=True).groups.items():
        loc = list(indices)
        nrej = holm_rejections(out.loc[loc, "npcr_p_value"].astype(float), alpha)
        urej = holm_rejections(out.loc[loc, "uaci_p_value"].astype(float), alpha)
        out.loc[loc, "npcr_holm_reject"] = nrej
        out.loc[loc, "uaci_holm_reject"] = urej
    out["joint_holm_pass"] = eligible & ~(out["npcr_holm_reject"] | out["uaci_holm_reject"])
    return out


def merge_differential(repo_root: Path, run_root: Path) -> dict[str, Any]:
    config = load_protocol_config()
    shard = run_root / "differential/shards"
    pair_frames: list[pd.DataFrame] = []
    projection_frames: list[pd.DataFrame] = []
    for method_id in config.methods:
        pair_path = shard / f"{method_id}__pairs.csv"
        projection_path = shard / f"{method_id}__projections_raw.csv"
        if not pair_path.exists() or not projection_path.exists():
            raise RuntimeError(f"missing differential shard for {method_id}")
        pair_frames.append(pd.read_csv(pair_path, keep_default_na=False))
        projection_frames.append(pd.read_csv(projection_path, keep_default_na=False))
    pairs = pd.concat(pair_frames, ignore_index=True)
    projections = pd.concat(projection_frames, ignore_index=True)
    pairs = pairs.sort_values(["tier_id", "method_id", "protocol", "image_size", "image_id", "perturbation_id", "key_index", "state_index"], kind="mergesort")
    projections = projections.sort_values(["tier_id", "method_id", "protocol", "image_size", "image_id", "perturbation_id", "key_index", "state_index", "projection_id"], kind="mergesort")
    inferred = _apply_vectorized_inference(projections, alpha=float(config.payload["statistics"]["primary_alpha"]))
    pair_path = run_root / "differential_pair_records.csv"
    projection_path = run_root / "differential_projection_records.csv"
    pairs.to_csv(pair_path, index=False, float_format=CSV_FLOAT_FORMAT)
    inferred.to_csv(projection_path, index=False, float_format=CSV_FLOAT_FORMAT, na_rep="NA")
    eligible = inferred[inferred["inferential_eligible"].astype(str).str.lower().isin({"true", "1"})].copy()
    group_cols = ["tier_id", "method_id", "protocol", "protocol_code", "operation_regime", "image_size", "projection_id", "semantic_domain"]
    summary_rows: list[dict[str, Any]] = []
    mean0, var0 = ideal_uaci_moments(256)
    for keys, group in eligible.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        n_values = group["sample_count"].astype(int).to_numpy()
        joint = group["joint_holm_pass"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
        npcr_pass = ~group["npcr_holm_reject"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
        uaci_pass = ~group["uaci_holm_reject"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
        row.update({
            "record_count": len(group),
            "image_count": group["image_id"].nunique(),
            "perturbation_count": group["perturbation_id"].nunique(),
            "mean_npcr": float(group["npcr"].astype(float).mean()),
            "std_npcr": float(group["npcr"].astype(float).std(ddof=1)) if len(group) > 1 else 0.0,
            "mean_uaci": float(group["uaci"].astype(float).mean()),
            "std_uaci": float(group["uaci"].astype(float).std(ddof=1)) if len(group) > 1 else 0.0,
            "ideal_uaci_mean": mean0,
            "mean_ideal_uaci_standard_deviation": float(np.mean(np.sqrt(var0 / n_values))),
            "npcr_pass_probability": float(np.mean(npcr_pass)),
            "uaci_pass_probability": float(np.mean(uaci_pass)),
            "joint_pass_probability": float(np.mean(joint)),
            "joint_pairwise_flip_probability": pairwise_flip_probability(joint),
        })
        summary_rows.append(row)
    summary_path = run_root / "differential_summary.csv"
    write_rows(summary_path, summary_rows)
    planned = sum(int(row["planned_pairs"]) for row in planned_pair_counts(config.payload))
    if len(pairs) != planned:
        raise RuntimeError(f"differential pair count {len(pairs)} != planned {planned}")
    if len(inferred) != 99072:
        raise RuntimeError(f"projection count {len(inferred)} != 99072")
    if len(summary_rows) != 346:
        raise RuntimeError(f"summary count {len(summary_rows)} != 346")
    manifest = {
        "pair_rows": len(pairs),
        "projection_rows": len(inferred),
        "summary_rows": len(summary_rows),
        "pair_canonical_sha256": canonical_csv_sha256(pair_path),
        "projection_canonical_sha256": canonical_csv_sha256(projection_path),
        "summary_canonical_sha256": canonical_csv_sha256(summary_path),
    }
    write_json(run_root / "differential_manifest.json", manifest)
    return manifest


def _context_for_object(config: ProtocolConfig, method_id: str, image_id: str, size: int, ordinal: int, purpose: str) -> RunContext:
    policy = method_policy(method_id)
    protocol = DifferentialProtocol.P2_FRESH_RANDOMNESS if policy.p2_applicable else DifferentialProtocol.P1_COMMON_CONTEXT
    pair = build_context_pair(
        master_key=config.master_key,
        master_seed=config.master_seed,
        method_id=method_id,
        protocol=protocol,
        image_id=image_id,
        perturbation_id=purpose,
        pair_ordinal=stable_ordinal(method_id, image_id, size, ordinal, purpose),
        key_index=0,
        state_index=ordinal,
        payload_length=size * size * 3,
        protocol_id=config.protocol_id,
    )
    return pair.left


def run_leakage(repo_root: Path, run_root: Path) -> dict[str, Any]:
    config = load_protocol_config()
    manifest_rows, images, corpus = load_manifest(repo_root)
    entropy_rows = leakage_entropy_rows(images, corpus, config.methods)
    entropy_map = {(row["method_id"], row["image_id"], int(row["image_size"])): row for row in entropy_rows}
    records: list[dict[str, Any]] = []
    objects = run_root / "leakage_objects"
    objects.mkdir(parents=True, exist_ok=True)
    for method_id in config.methods:
        method = make_method(method_id, profile="extended")
        policy = method_policy(method_id)
        for ordinal, ((image_id, size), image) in enumerate(sorted(images.items(), key=lambda item: (item[0][1], item[0][0]))):
            context = _context_for_object(config, method_id, image_id, size, ordinal, "LEAKAGE")
            encrypted = method.encrypt(image, context)
            views = extract_protocol_views(encrypted.object_bytes, method_id)
            parsed = parse_envelope(encrypted.object_bytes)
            forbidden = recursive_forbidden_paths(parsed.header)
            if forbidden:
                raise RuntimeError(f"forbidden header keys in {method_id}: {forbidden}")
            public_exact: bool | str = "NA"
            if policy.publicly_invertible:
                attacker = replace(context, master_key=hashlib.sha256(b"QSA-UNRELATED-KEY").digest())
                recovered = method.decrypt(encrypted.object_bytes, attacker)
                public_exact = bool(np.array_equal(image, recovered))
                if not public_exact:
                    raise RuntimeError(f"declared public inverse failed for {method_id}")
            base = dict(entropy_map[(method_id, image_id, size)])
            base.update({
                "object_bytes": len(encrypted.object_bytes),
                "header_bytes": len(views.header_bytes),
                "nonce_bytes": len(views.nonce),
                "public_payload_bytes": len(views.public_payload),
                "protected_payload_bytes": len(views.protected_payload),
                "tag_bytes": len(views.tag),
                "metric_body_bytes": len(views.metric_body),
                "metric_body_source": views.metric_body_source,
                "body_metric_domain": views.body_metric_domain,
                "object_sha256": hashlib.sha256(encrypted.object_bytes).hexdigest(),
                "header_sha256": hashlib.sha256(views.header_bytes).hexdigest(),
                "public_payload_sha256": hashlib.sha256(views.public_payload).hexdigest(),
                "protected_payload_sha256": hashlib.sha256(views.protected_payload).hexdigest(),
                "tag_sha256": hashlib.sha256(views.tag).hexdigest(),
                "recursive_metadata_clean": True,
                "public_wrong_key_exact_recovery": public_exact,
                "observed_post_object_recovery_bits": 0.0 if public_exact is True else "NA",
            })
            records.append(base)
            if image_id == "astronaut" and size == 256:
                (objects / f"{method_id}.qsb").write_bytes(encrypted.object_bytes)
    records_path = run_root / "leakage_from_object_records.csv"
    write_rows(records_path, records)
    summary_rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(records)
    for method_id, group in frame.groupby("method_id", sort=True):
        policy = method_policy(method_id)
        summary_rows.append({
            "method_id": method_id,
            "record_count": len(group),
            "publicly_invertible": policy.publicly_invertible,
            "all_recursive_metadata_clean": bool(group["recursive_metadata_clean"].astype(bool).all()),
            "public_wrong_key_exact_count": int(sum(value is True for value in group["public_wrong_key_exact_recovery"])),
            "median_object_bytes": float(group["object_bytes"].astype(float).median()),
            "median_protected_payload_bytes": float(group["protected_payload_bytes"].astype(float).median()),
            "minimum_prechallenge_L_class_bits": float(pd.to_numeric(group["prechallenge_L_class_bits"], errors="coerce").min()),
            "minimum_descriptor_orbit_bits": float(pd.to_numeric(group["descriptor_only_orbit_bits"], errors="coerce").min()),
            "permitted_functionality": "|".join(policy.permitted_functionality),
            "post_object_recovery_class": policy.post_object_recovery,
        })
    summary_path = run_root / "leakage_from_object_summary.csv"
    write_rows(summary_path, summary_rows)
    if len(records) != 1152 or len(summary_rows) != 24:
        raise RuntimeError(f"leakage counts invalid: {len(records)}, {len(summary_rows)}")
    manifest = {
        "record_rows": len(records),
        "summary_rows": len(summary_rows),
        "record_canonical_sha256": canonical_csv_sha256(records_path),
        "summary_canonical_sha256": canonical_csv_sha256(summary_path),
        "representative_objects": len(list(objects.glob("*.qsb"))),
    }
    write_json(run_root / "leakage_manifest.json", manifest)
    return manifest


def _object_feature(blob: bytes, method_id: str) -> np.ndarray:
    views = extract_protocol_views(blob, method_id)
    body = views.metric_body
    values = np.frombuffer(body, dtype=np.uint8)
    hist = np.bincount(values, minlength=256).astype(float)
    hist /= max(len(values), 1)
    # Fixed 64-position quantile sample preserves coarse serialized structure.
    if len(values):
        indices = np.linspace(0, len(values) - 1, 64).astype(int)
        sample = values[indices].astype(float) / 255.0
    else:
        sample = np.zeros(64)
    scalar = np.asarray([
        math.log1p(len(blob)), math.log1p(len(views.header_bytes)), math.log1p(len(views.nonce)),
        math.log1p(len(views.public_payload)), math.log1p(len(views.protected_payload)),
        math.log1p(len(views.tag)), byte_entropy(body) / 8.0,
        (adjacent_byte_correlation(body) + 1.0) / 2.0,
    ])
    return np.concatenate([hist, sample, scalar])


def _target_lowres(image: np.ndarray, size: int = 8) -> np.ndarray:
    reduced = Image.fromarray(image, mode="RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(reduced, dtype=np.uint8).reshape(-1).astype(float) / 255.0


def _psnr_u8(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = reference.astype(float)
    pred = prediction.astype(float)
    mse = float(np.mean((ref - pred) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def run_attacks(repo_root: Path, run_root: Path) -> dict[str, Any]:
    config = load_protocol_config()
    _, images, _ = load_manifest(repo_root)
    representative_manifest: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    structured_rows: list[dict[str, Any]] = []
    learning_manifest: list[dict[str, Any]] = []
    learned_rows: list[dict[str, Any]] = []
    distinguisher_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    object_root = run_root / "attack_objects"
    object_root.mkdir(parents=True, exist_ok=True)

    representative_image = images[("astronaut", 256)]
    second_image = images[("coffee", 256)]
    representative: dict[str, tuple[bytes, RunContext]] = {}
    public_exact: dict[str, bool] = {}

    # Representative complete objects, public recovery, active tampering, and reuse.
    for method_index, method_id in enumerate(config.methods):
        method = make_method(method_id, profile="extended")
        policy = method_policy(method_id)
        context = _context_for_object(config, method_id, "astronaut", 256, method_index, "ATTACK-REP")
        second_context = _context_for_object(config, method_id, "coffee", 256, method_index + 100, "ATTACK-REP-SECOND")
        first = method.encrypt(representative_image, context)
        second = method.encrypt(second_image, second_context)
        representative[method_id] = (first.object_bytes, context)
        object_path = object_root / f"{method_id}.qsb"
        object_path.write_bytes(first.object_bytes)
        views = extract_protocol_views(first.object_bytes, method_id)
        recovered_exact: bool | str = "NA"
        if policy.publicly_invertible:
            attacker = replace(context, master_key=hashlib.sha256(b"QSA-PUBLIC-ATTACKER").digest())
            recovered_exact = bool(np.array_equal(method.decrypt(first.object_bytes, attacker), representative_image))
            if not recovered_exact:
                raise RuntimeError(f"public recovery failed for {method_id}")
            public_exact[method_id] = True
        else:
            public_exact[method_id] = False
        representative_manifest.append({
            "method_id": method_id,
            "image_id": "astronaut",
            "image_size": 256,
            "object_path": str(object_path.relative_to(run_root)),
            "object_bytes": len(first.object_bytes),
            "protected_payload_bytes": len(views.protected_payload),
            "object_sha256": hashlib.sha256(first.object_bytes).hexdigest(),
            "protected_payload_sha256": hashlib.sha256(views.protected_payload).hexdigest(),
            "recursive_metadata_clean": not recursive_forbidden_paths(parse_envelope(first.object_bytes).header),
            "public_wrong_key_exact_recovery": recovered_exact,
        })
        for mutation in generate_active_mutations(first.object_bytes, second.object_bytes):
            mutation_id = mutation.mutation_id
            mutated = mutation.object_bytes
            accepted = False
            changed = False
            error_type = ""
            try:
                decoded = method.decrypt(mutated, context)
                accepted = True
                changed = not np.array_equal(decoded, representative_image)
            except Exception as exc:  # expected for authenticated or malformed objects
                error_type = _reported_attack_error_type(exc)
            active_rows.append({
                "method_id": method_id,
                "mutation_id": mutation_id,
                "authenticated": method.authenticated,
                "accepted": accepted,
                "accepted_plaintext_changed": changed,
                "error_type": error_type,
                "regime": "fresh_object_modification",
            })

        # Common-context pair for the forced-reuse relation record.
        pair = build_context_pair(
            master_key=config.master_key, master_seed=config.master_seed, method_id=method_id,
            protocol=DifferentialProtocol.P1_COMMON_CONTEXT, image_id="reuse_pair",
            perturbation_id="FORCED-REUSE", pair_ordinal=stable_ordinal(method_id, "reuse"),
            key_index=0, state_index=0, payload_length=representative_image.nbytes,
            protocol_id=config.protocol_id,
        )
        a = method.encrypt(representative_image, pair.left)
        b = method.encrypt(second_image, pair.right)
        va = extract_protocol_views(a.object_bytes, method_id)
        vb = extract_protocol_views(b.object_bytes, method_id)
        xor_accuracy: float | str = "NA"
        exact_second = False
        if len(va.protected_payload) == len(vb.protected_payload) == representative_image.nbytes:
            cdelta = np.bitwise_xor(
                np.frombuffer(va.protected_payload, dtype=np.uint8),
                np.frombuffer(vb.protected_payload, dtype=np.uint8),
            )
            pdelta = np.bitwise_xor(representative_image.reshape(-1), second_image.reshape(-1))
            xor_accuracy = float(np.mean(cdelta == pdelta))
            recovered = np.bitwise_xor(cdelta, representative_image.reshape(-1)).reshape(second_image.shape)
            exact_second = bool(np.array_equal(recovered, second_image))
        relation_exposed = bool(exact_second or (isinstance(xor_accuracy, float) and xor_accuracy == 1.0))
        reuse_rows.append({
            "method_id": method_id,
            "operation_regime": pair.operation_regime,
            "protected_lengths_equal": len(va.protected_payload) == len(vb.protected_payload),
            "raw_plaintext_xor_accuracy": xor_accuracy,
            "raw_plaintext_second_exact": exact_second,
            "relation_exposed": relation_exposed,
            "publicly_invertible": policy.publicly_invertible,
            "interpretation": "raw-byte XOR witness; structured-transform relations are reported separately",
        })

    # Query-structured recovery controls on the fixed 96x96 timing image.
    timing_images = load_timing_images(repo_root, config.payload)
    target = timing_images[("astronaut", 96)]
    ctx = _context_for_object(config, "B13_permutation_only", "astronaut", 96, 0, "STRUCTURED")
    mapping, accuracy, queries = recover_permutation(target.shape, ctx)
    structured_rows.append({
        "method_id": "B13_permutation_only", "attack": "chosen_plaintext_position_code",
        "queries": queries, "exact": accuracy == 1.0, "accuracy": accuracy,
        "notes": "two-query base-256 position code at 27,648 byte positions",
    })
    for method_id in ("B14_diffusion_only", "B19_external_chaos_pd"):
        ctx = _context_for_object(config, method_id, "astronaut", 96, 0, "STRUCTURED")
        recovered = affine_diffusion_recovery(method_id, target, ctx)
        exact = bool(np.array_equal(recovered, target))
        structured_rows.append({
            "method_id": method_id, "attack": "known_zero_affine_difference",
            "queries": 2, "exact": exact, "accuracy": float(np.mean(recovered == target)),
            "notes": "zero-reference cancellation followed by exact inverse difference",
        })

    # Deterministic learning corpus: all 24 standard 256x256 images per method.
    image_ids = list(config.payload["corpus"]["image_ids"])
    split_test = {"astronaut", "coffee", "chelsea", "retina", "syn_checker", "syn_lowrank"}
    for method_index, method_id in enumerate(config.methods):
        method = make_method(method_id, profile="extended")
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        labels: list[str] = []
        blobs: list[bytes] = []
        for image_index, image_id in enumerate(image_ids):
            image = images[(image_id, 256)]
            context = _context_for_object(config, method_id, image_id, 256, image_index, "LEARNING")
            encrypted = method.encrypt(image, context)
            features.append(_object_feature(encrypted.object_bytes, method_id))
            targets.append(_target_lowres(image))
            labels.append(image_id)
            blobs.append(encrypted.object_bytes)
            learning_manifest.append({
                "method_id": method_id, "image_id": image_id, "image_size": 256,
                "split": "test" if image_id in split_test else "train",
                "object_sha256": hashlib.sha256(encrypted.object_bytes).hexdigest(),
                "object_bytes": len(encrypted.object_bytes),
            })
        X = np.vstack(features); Y = np.vstack(targets)
        train_idx = np.asarray([i for i, label in enumerate(labels) if label not in split_test])
        test_idx = np.asarray([i for i, label in enumerate(labels) if label in split_test])
        model = Ridge(alpha=10.0, fit_intercept=True)
        model.fit(X[train_idx], Y[train_idx])
        predictions = np.clip(model.predict(X[test_idx]), 0.0, 1.0)
        for idx, pred in zip(test_idx, predictions, strict=True):
            target_u8 = np.rint(Y[idx].reshape(8, 8, 3) * 255.0).astype(np.uint8)
            pred_u8 = np.rint(pred.reshape(8, 8, 3) * 255.0).astype(np.uint8)
            metrics = reconstruction_metrics(target_u8, pred_u8)
            learned_rows.append({
                "method_id": method_id,
                "image_id": labels[idx],
                "split": "test",
                "attack": "ridge_object_feature_to_8x8_rgb",
                "evaluation_size": 8,
                "psnr_db": metrics["psnr_db"],
                "ssim": metrics["ssim"],
                "mean_delta_e": metrics["mean_delta_e"],
                "ycbcr_psnr_db": metrics["ycbcr_psnr_db"],
                "exact": bool(np.array_equal(target_u8, pred_u8)),
                "feature_count": X.shape[1],
                "training_objects": len(train_idx),
            })

        # Real-vs-ideal protected-body distinguisher with fixed features and split.
        real_features: list[np.ndarray] = []
        ideal_features: list[np.ndarray] = []
        for object_index, blob in enumerate(blobs):
            views = extract_protocol_views(blob, method_id)
            body = views.metric_body
            real_features.append(_object_feature(blob, method_id))
            seed = hashlib.shake_256(
                b"QSA-IDEAL-BODY" + method_id.encode() + object_index.to_bytes(4, "big")
            ).digest(len(body))
            # Preserve the real envelope structure but replace only the registered metric body.
            parsed = parse_envelope(blob)
            if method_policy(method_id).metric_body_source == "protected_payload":
                ideal_blob = reencode_envelope(parsed, protected=seed)
            else:
                ideal_blob = reencode_envelope(parsed, public=seed)
            ideal_features.append(_object_feature(ideal_blob, method_id))
        DX = np.vstack(real_features + ideal_features)
        Dy = np.asarray([1] * len(real_features) + [0] * len(ideal_features))
        indices = np.arange(len(Dy))
        train, test = train_test_split(indices, test_size=0.35, random_state=20260807 + method_index, stratify=Dy)
        clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=20260807 + method_index)
        clf.fit(DX[train], Dy[train])
        scores = clf.predict_proba(DX[test])[:, 1]
        auc = float(roc_auc_score(Dy[test], scores))
        distinguisher_rows.append({
            "method_id": method_id,
            "train_samples": len(train),
            "test_samples": len(test),
            "test_auc": auc,
            "symmetrized_advantage": float(2.0 * abs(auc - 0.5)),
            "feature_count": DX.shape[1],
            "ideal_body_rule": "same registered metric-body length; deterministic SHAKE-256 ideal bytes",
        })

    # Build the primary matrix from measured, nonaggregated evidence.
    active_by = defaultdict(list)
    for row in active_rows: active_by[row["method_id"]].append(row)
    reuse_by = {row["method_id"]: row for row in reuse_rows}
    structured_by = {row["method_id"]: row for row in structured_rows}
    learned_by = defaultdict(list)
    for row in learned_rows: learned_by[row["method_id"]].append(row)
    dist_by = {row["method_id"]: row for row in distinguisher_rows}
    metadata = {row["method_id"]: row for row in protocol_method_registry()}
    for method_id in config.methods:
        method = make_method(method_id, profile="extended")
        policy = method_policy(method_id)
        active = active_by[method_id]
        accepted_changed = sum(bool(r["accepted"]) and bool(r["accepted_plaintext_changed"]) for r in active)
        all_modifications_rejected = all(not bool(r["accepted"]) for r in active) if method.authenticated else False
        learned_psnr = max(float(r["psnr_db"]) for r in learned_by[method_id])
        struct = structured_by.get(method_id)
        matrix_rows.append({
            "method_id": method_id,
            "display_name": metadata[method_id]["display_name"],
            "family": metadata[method_id]["family"],
            "correct_use_confidentiality_class": (
                "STANDARD_OR_MISUSE_RESISTANT_CONTROL" if method.secure_control
                else "ZERO_BY_PUBLIC_INVERSION" if policy.publicly_invertible
                else "NOT_ASSERTED"
            ),
            "exact_public_inversion_applicable": policy.publicly_invertible,
            "exact_public_inversion_observed": public_exact[method_id],
            "structured_recovery_applicable": struct is not None,
            "structured_recovery_exact": bool(struct["exact"]) if struct is not None else "NA",
            "structured_recovery_accuracy": float(struct["accuracy"]) if struct is not None else "NA",
            "learned_reconstruction_best_psnr_db_8x8": learned_psnr,
            "protected_body_distinguisher_auc": float(dist_by[method_id]["test_auc"]),
            "protected_body_distinguisher_advantage": float(dist_by[method_id]["symmetrized_advantage"]),
            "authenticated": method.authenticated,
            "active_modification_probe_count": len(active),
            "active_accepted_changed_count": accepted_changed,
            "all_registered_modifications_rejected": all_modifications_rejected,
            "forced_reuse_relation_exposed": bool(reuse_by[method_id]["relation_exposed"]),
            "forced_reuse_raw_xor_accuracy": reuse_by[method_id]["raw_plaintext_xor_accuracy"],
            "permitted_preview_leakage": method_id == "B20_full_aead_explicit_preview",
            "permitted_functionality": "|".join(policy.permitted_functionality),
            "operation_regime_reference": policy.p1_semantics,
            "primary_scalar_score": "FORBIDDEN_NOT_COMPUTED",
        })

    paths = {
        "representative": run_root / "representative_object_manifest.csv",
        "active": run_root / "active_tamper_records.csv",
        "reuse": run_root / "forced_reuse_records.csv",
        "structured": run_root / "structured_recovery_records.csv",
        "learning_manifest": run_root / "learning_object_manifest.csv",
        "learned": run_root / "learned_reconstruction_records.csv",
        "distinguisher": run_root / "protected_body_distinguishing_records.csv",
        "matrix": run_root / "attack_leakage_regime_matrix.csv",
    }
    write_rows(paths["representative"], representative_manifest)
    write_rows(paths["active"], active_rows)
    write_rows(paths["reuse"], reuse_rows)
    write_rows(paths["structured"], structured_rows)
    write_rows(paths["learning_manifest"], learning_manifest)
    write_rows(paths["learned"], learned_rows)
    write_rows(paths["distinguisher"], distinguisher_rows)
    write_rows(paths["matrix"], matrix_rows)
    if len(representative_manifest) != 24 or len(reuse_rows) != 24 or len(structured_rows) != 3:
        raise RuntimeError("attack core counts invalid")
    if len(learning_manifest) != 576 or len(learned_rows) != 144 or len(distinguisher_rows) != 24 or len(matrix_rows) != 24:
        raise RuntimeError("attack learning/matrix counts invalid")
    manifest = {
        "representative_rows": len(representative_manifest),
        "active_rows": len(active_rows),
        "forced_reuse_rows": len(reuse_rows),
        "structured_rows": len(structured_rows),
        "learning_object_rows": len(learning_manifest),
        "learned_reconstruction_rows": len(learned_rows),
        "distinguisher_rows": len(distinguisher_rows),
        "matrix_rows": len(matrix_rows),
        "public_exact_methods": sum(public_exact.values()),
        "canonical_digests": {name: canonical_csv_sha256(path) for name, path in paths.items()},
    }
    write_json(run_root / "attack_manifest.json", manifest)
    return manifest


def _time_call(function: Callable[[], Any]) -> tuple[Any, int]:
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    start = time.perf_counter_ns()
    try:
        value = function()
    finally:
        elapsed = time.perf_counter_ns() - start
        if gc_was_enabled:
            gc.enable()
    return value, int(elapsed)


def _profile_exclusive(function: Callable[[], Any]) -> list[tuple[str, str, float]]:
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    profiler.disable()
    records: list[tuple[str, str, float]] = []
    for entry in profiler.getstats():
        code = entry.code
        if isinstance(code, str):
            filename, name = "<builtin>", code
        else:
            filename, name = str(code.co_filename), str(code.co_name)
        records.append((filename, name, float(entry.inlinetime)))
    return records


def _classify_profile(filename: str, name: str, direction: str, applicable: Sequence[str]) -> str:
    fn = filename.replace("\\", "/")
    lower = name.lower()
    app = set(applicable)
    candidates: list[str] = []
    if direction == "encrypt":
        if "serialization.py" in fn and ("image_to_bytes" in lower or "serialize" in lower):
            candidates.append("input_encode" if "image_to_bytes" in lower else "transform_serialize")
        if "envelope.py" in fn and "encode" in lower:
            candidates.append("envelope_serialize")
        if "transforms.py" in fn or "quaternion.py" in fn:
            if "pca" in lower or "descriptor" in lower or "preview" in lower:
                candidates.append("descriptor_compute")
            candidates.append("geometry_forward")
        if any(token in fn for token in ("crypto.py", "components.py", "external.py", "controls.py")):
            candidates.append("primitive_protect")
        if "constructions.py" in fn and "preview" in lower:
            candidates.append("descriptor_compute")
    else:
        if "envelope.py" in fn and "parse" in lower:
            candidates.append("envelope_parse")
        if "serialization.py" in fn and "bytes_to_image" in lower:
            candidates.append("output_decode")
        if "serialization.py" in fn and "deserialize" in lower:
            candidates.append("transform_deserialize")
        if "transforms.py" in fn or "quaternion.py" in fn:
            candidates.append("geometry_inverse")
        if any(token in fn for token in ("crypto.py", "components.py", "external.py", "controls.py")):
            candidates.append("primitive_unprotect")
        if "compare_digest" in lower or "decrypt" in lower:
            candidates.append("release_check")
    for candidate in candidates:
        if candidate in app:
            return candidate
    # Assign unclassified exclusive time to a semantically broad stage.
    preferred = (
        ["primitive_protect", "geometry_forward", "envelope_serialize", "input_encode"]
        if direction == "encrypt"
        else ["primitive_unprotect", "geometry_inverse", "envelope_parse", "output_decode"]
    )
    return next((stage for stage in preferred if stage in app), applicable[0])


def _profile_weights(function: Callable[[], Any], direction: str, applicable: Sequence[str]) -> dict[str, float]:
    totals = {stage: 1e-12 for stage in applicable}
    for filename, name, inline in _profile_exclusive(function):
        stage = _classify_profile(filename, name, direction, applicable)
        totals[stage] += max(inline, 0.0)
    denominator = sum(totals.values())
    return {stage: value / denominator for stage, value in totals.items()}


def _allocate_ns(total_ns: int, stages: Sequence[str], weights: Mapping[str, float]) -> dict[str, int]:
    raw = [total_ns * float(weights[stage]) for stage in stages]
    floors = [int(math.floor(value)) for value in raw]
    remainder = int(total_ns - sum(floors))
    order = sorted(range(len(stages)), key=lambda i: (raw[i] - floors[i], -i), reverse=True)
    for index in order[:remainder]:
        floors[index] += 1
    return dict(zip(stages, floors, strict=True))


def _rss_bytes() -> int:
    # ru_maxrss is KiB on Linux and bytes on macOS; this execution is Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def run_timing_shard(repo_root: Path, run_root: Path, method_id: str) -> dict[str, Any]:
    config = load_protocol_config()
    payload = config.payload
    timing = payload["timing"]
    images = load_timing_images(repo_root, payload)
    method = make_method(method_id, profile="extended")
    total_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    start_all = time.time()
    for size in timing["sizes"]:
        size = int(size)
        for image_id in payload["corpus"]["timing_panel"]:
            image = images[(image_id, size)]
            for direction in ("encrypt", "decrypt"):
                app = applicable_stages(method_id, direction)
                # Warm up the exact path with fresh deterministic contexts.
                for warmup in range(int(timing["warmups"])):
                    ctx = _context_for_object(config, method_id, image_id, size, 10000 + warmup, f"TIMING-WARMUP-{direction}")
                    encrypted = method.encrypt(image, ctx)
                    if direction == "decrypt":
                        method.decrypt(encrypted.object_bytes, ctx)
                # One profiler pass establishes exclusive-time proportions per configuration.
                profile_ctx = _context_for_object(config, method_id, image_id, size, 20000, f"TIMING-PROFILE-{direction}")
                profile_obj = method.encrypt(image, profile_ctx)
                profile_function = (
                    (lambda: method.encrypt(image, profile_ctx))
                    if direction == "encrypt"
                    else (lambda: method.decrypt(profile_obj.object_bytes, profile_ctx))
                )
                weights = _profile_weights(profile_function, direction, app)
                for repetition in range(int(timing["timed_repetitions"])):
                    ctx = _context_for_object(config, method_id, image_id, size, repetition, f"TIMING-{direction}")
                    if direction == "encrypt":
                        value, elapsed = _time_call(lambda: method.encrypt(image, ctx))
                        views = extract_protocol_views(value.object_bytes, method_id)
                        object_bytes = len(value.object_bytes)
                        protected_bytes = len(views.protected_payload)
                    else:
                        encrypted = method.encrypt(image, ctx)
                        views = extract_protocol_views(encrypted.object_bytes, method_id)
                        value, elapsed = _time_call(lambda: method.decrypt(encrypted.object_bytes, ctx))
                        if not np.array_equal(value, image):
                            raise RuntimeError(f"timing decrypt mismatch for {method_id}")
                        object_bytes = len(encrypted.object_bytes)
                        protected_bytes = len(views.protected_payload)
                    run_id = f"{method_id}|{image_id}|{size}|{direction}|r{repetition}"
                    total_rows.append({
                        "run_id": run_id,
                        "method_id": method_id,
                        "image_id": image_id,
                        "image_size": size,
                        "direction": direction,
                        "repetition": repetition,
                        "elapsed_ns": elapsed,
                        "pixel_count": size * size,
                        "raw_image_bytes": image.nbytes,
                        "protected_bytes": protected_bytes,
                        "object_bytes": object_bytes,
                        "gc_disabled_inside_timed_region": True,
                    })
                    allocated = _allocate_ns(elapsed, app, weights)
                    stage_sum = 0
                    for stage in app:
                        stage_elapsed = allocated[stage]
                        stage_sum += stage_elapsed
                        stage_rows.append({
                            "run_id": run_id,
                            "method_id": method_id,
                            "image_id": image_id,
                            "image_size": size,
                            "direction": direction,
                            "repetition": repetition,
                            "stage": stage,
                            "elapsed_ns": stage_elapsed,
                            "total_elapsed_ns": elapsed,
                            "profile_weight": weights[stage],
                            "attribution_basis": "exclusive_cProfile_weights_normalized_to_separate_total",
                        })
                    diagnostic_rows.append({
                        "run_id": run_id,
                        "method_id": method_id,
                        "image_id": image_id,
                        "image_size": size,
                        "direction": direction,
                        "repetition": repetition,
                        "total_elapsed_ns": elapsed,
                        "stage_sum_ns": stage_sum,
                        "stage_sum_relative_error": abs(stage_sum - elapsed) / max(elapsed, 1),
                        "stage_count": len(app),
                    })
                for memory_repetition in range(int(timing["memory_repetitions"])):
                    ctx = _context_for_object(config, method_id, image_id, size, 30000 + memory_repetition, f"MEMORY-{direction}")
                    encrypted = method.encrypt(image, ctx)
                    before = _rss_bytes()
                    if direction == "encrypt":
                        value = method.encrypt(image, ctx)
                        output_bytes = len(value.object_bytes)
                    else:
                        value = method.decrypt(encrypted.object_bytes, ctx)
                        output_bytes = int(value.nbytes)
                    after = _rss_bytes()
                    memory_rows.append({
                        "run_id": f"{method_id}|{image_id}|{size}|{direction}|m{memory_repetition}",
                        "method_id": method_id,
                        "image_id": image_id,
                        "image_size": size,
                        "direction": direction,
                        "memory_repetition": memory_repetition,
                        "rss_before_bytes": before,
                        "rss_after_bytes": after,
                        "rss_nonnegative_delta_bytes": max(0, after - before),
                        "output_bytes": output_bytes,
                        "measurement": "process_ru_maxrss_delta_separate_pass",
                    })
            print(f"[{method_id}] timing {image_id} {size} complete", flush=True)
    shard = run_root / "timing/shards"
    paths = {
        "total": shard / f"{method_id}__total.csv",
        "stage": shard / f"{method_id}__stages.csv",
        "memory": shard / f"{method_id}__memory.csv",
        "diagnostic": shard / f"{method_id}__diagnostics.csv",
    }
    write_rows(paths["total"], total_rows)
    write_rows(paths["stage"], stage_rows)
    write_rows(paths["memory"], memory_rows)
    write_rows(paths["diagnostic"], diagnostic_rows)
    manifest = {
        "method_id": method_id,
        "total_rows": len(total_rows),
        "stage_rows": len(stage_rows),
        "memory_rows": len(memory_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "elapsed_seconds": time.time() - start_all,
        "digests": {name: sha256_file(path) for name, path in paths.items()},
    }
    write_json(shard / f"{method_id}__manifest.json", manifest)
    return manifest


def merge_timing(repo_root: Path, run_root: Path) -> dict[str, Any]:
    config = load_protocol_config()
    shard = run_root / "timing/shards"
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in ("total", "stage", "memory", "diagnostic")}
    for method_id in config.methods:
        for key, suffix in (("total", "total"), ("stage", "stages"), ("memory", "memory"), ("diagnostic", "diagnostics")):
            path = shard / f"{method_id}__{suffix}.csv"
            if not path.exists():
                raise RuntimeError(f"missing timing shard: {path}")
            frames[key].append(pd.read_csv(path, keep_default_na=False))
    merged = {key: pd.concat(value, ignore_index=True) for key, value in frames.items()}
    sort_cols = ["method_id", "image_size", "image_id", "direction", "repetition"]
    merged["total"] = merged["total"].sort_values(sort_cols, kind="mergesort")
    merged["stage"] = merged["stage"].sort_values(sort_cols + ["stage"], kind="mergesort")
    merged["memory"] = merged["memory"].sort_values(["method_id", "image_size", "image_id", "direction", "memory_repetition"], kind="mergesort")
    merged["diagnostic"] = merged["diagnostic"].sort_values(sort_cols, kind="mergesort")
    paths = {
        "total": run_root / "timing_total_records.csv",
        "stage": run_root / "timing_stage_records.csv",
        "memory": run_root / "timing_memory_records.csv",
        "diagnostic": run_root / "timing_component_diagnostics.csv",
    }
    for key, frame in merged.items():
        frame.to_csv(paths[key], index=False, float_format=CSV_FLOAT_FORMAT)
    expected = {"total": 8640, "stage": 32940, "memory": 2880, "diagnostic": 8640}
    for key, count in expected.items():
        if len(merged[key]) != count:
            raise RuntimeError(f"timing {key} count {len(merged[key])} != {count}")
    manifest = {
        **{f"{key}_rows": len(frame) for key, frame in merged.items()},
        "canonical_digests": {key: canonical_csv_sha256(path) for key, path in paths.items()},
    }
    write_json(run_root / "timing_manifest.json", manifest)
    return manifest


def fit_cost_model(run_root: Path, config: ProtocolConfig) -> dict[str, Any]:
    total = pd.read_csv(run_root / "timing_total_records.csv")
    enc = total[total["direction"] == "encrypt"].copy()
    grouping = ["method_id", "image_id", "image_size"]
    samples: list[dict[str, float]] = []
    for _, group in enc.groupby(grouping, sort=True):
        samples.append({
            "elapsed_ns": float(group["elapsed_ns"].median()),
            "pixel_count": float(group["pixel_count"].iloc[0]),
            "protected_bytes": float(group["protected_bytes"].median()),
        })
    fit = fit_serial_cost_model(samples)
    diagnostics = pd.read_csv(run_root / "timing_component_diagnostics.csv")
    fit["sample_count"] = len(samples)
    fit["median_instrumentation_residual_fraction"] = float(
        np.median(np.abs(diagnostics["stage_sum_ns"] - diagnostics["total_elapsed_ns"]) / np.maximum(diagnostics["total_elapsed_ns"], 1))
    )
    fit["median_stage_sum_relative_error"] = float(diagnostics["stage_sum_relative_error"].median())
    holds = config.payload["timing"]["hold_conditions"]
    reasons: list[str] = []
    if any(fit[key] < 0 for key in ("beta0_ns", "beta_geo_ns_per_pixel", "beta_primitive_ns_per_byte")):
        reasons.append("negative_fitted_coefficient")
    if fit["adjusted_r2"] < float(holds["adjusted_r2_below"]):
        reasons.append("adjusted_r2_below_threshold")
    if fit["maximum_relative_residual"] > float(holds["maximum_relative_residual_above"]):
        reasons.append("maximum_relative_residual_above_threshold")
    if fit["median_stage_sum_relative_error"] > float(holds["median_stage_sum_relative_error_above"]):
        reasons.append("median_stage_sum_relative_error_above_threshold")
    fit["accepted_for_interpretation"] = not reasons
    fit["hold_reasons"] = reasons
    fit["model"] = config.payload["timing"]["fit_model"]
    fit["interpretation_policy"] = "reported regardless; substantive stage-additive interpretation only when accepted"
    write_json(run_root / "cost_model_fit.json", fit)
    return fit


def compare_timing(primary: Path, independent: Path, output: Path) -> dict[str, Any]:
    first = pd.read_csv(primary / "timing_total_records.csv")
    second = pd.read_csv(independent / "timing_total_records.csv")
    grouping = ["method_id", "image_id", "image_size", "direction"]
    first_median = first.groupby(grouping, sort=True)["elapsed_ns"].median().rename("primary_median_ns")
    second_median = second.groupby(grouping, sort=True)["elapsed_ns"].median().rename("independent_median_ns")
    merged = pd.concat([first_median, second_median], axis=1).reset_index()
    merged["signed_relative_difference"] = (
        merged["independent_median_ns"] - merged["primary_median_ns"]
    ) / merged["primary_median_ns"].clip(lower=1)
    merged["absolute_relative_difference"] = merged["signed_relative_difference"].abs()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, float_format=CSV_FLOAT_FORMAT)
    summary = {
        "configuration_count": len(merged),
        "median_absolute_relative_difference": float(merged["absolute_relative_difference"].median()),
        "p95_absolute_relative_difference": float(merged["absolute_relative_difference"].quantile(0.95)),
        "median_signed_relative_difference": float(merged["signed_relative_difference"].median()),
        "thresholds": {"median_absolute_max": 0.15, "p95_absolute_max": 0.35, "absolute_signed_median_max": 0.10},
    }
    summary["passed"] = (
        summary["configuration_count"] == 576
        and summary["median_absolute_relative_difference"] <= 0.15
        and summary["p95_absolute_relative_difference"] <= 0.35
        and abs(summary["median_signed_relative_difference"]) <= 0.10
    )
    return summary


def _artifact_inventory() -> list[str]:
    return [
        "differential_pair_records.csv",
        "differential_projection_records.csv",
        "differential_summary.csv",
        "leakage_from_object_records.csv",
        "leakage_from_object_summary.csv",
        "representative_object_manifest.csv",
        "active_tamper_records.csv",
        "forced_reuse_records.csv",
        "structured_recovery_records.csv",
        "learning_object_manifest.csv",
        "learned_reconstruction_records.csv",
        "protected_body_distinguishing_records.csv",
        "attack_leakage_regime_matrix.csv",
    ]


def reconcile_two_runs(repo_root: Path, results_root: Path) -> dict[str, Any]:
    config = load_protocol_config()
    primary = results_root / "primary_execution"
    independent = results_root / "independent_execution"
    primary_cost = fit_cost_model(primary, config)
    independent_cost = fit_cost_model(independent, config)
    timing = compare_timing(primary, independent, results_root / "timing_comparison.csv")

    checks: list[dict[str, Any]] = []
    for filename in _artifact_inventory():
        first_path = primary / filename
        second_path = independent / filename
        first_digest = canonical_csv_sha256(first_path) if first_path.exists() else "MISSING"
        second_digest = canonical_csv_sha256(second_path) if second_path.exists() else "MISSING"
        checks.append({
            "artifact": filename,
            "primary_sha256": first_digest,
            "independent_sha256": second_digest,
            "match": first_digest == second_digest and first_digest != "MISSING",
        })
    for method_id in config.methods:
        first_path = primary / "attack_objects" / f"{method_id}.qsb"
        second_path = independent / "attack_objects" / f"{method_id}.qsb"
        first_digest = sha256_file(first_path) if first_path.exists() else "MISSING"
        second_digest = sha256_file(second_path) if second_path.exists() else "MISSING"
        checks.append({
            "artifact": f"attack_objects/{method_id}.qsb",
            "primary_sha256": first_digest,
            "independent_sha256": second_digest,
            "match": first_digest == second_digest and first_digest != "MISSING",
        })

    validation_rows: list[dict[str, Any]] = []
    def check(identifier: str, description: str, passed: bool, evidence: str) -> None:
        validation_rows.append({
            "check_id": identifier,
            "description": description,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        })

    check("V01", "Both execution trees are populated", primary.exists() and independent.exists() and any(primary.iterdir()) and any(independent.iterdir()), f"{primary}; {independent}")
    check("V02", "All deterministic artifacts match", all(bool(row["match"]) for row in checks), f"{sum(bool(row['match']) for row in checks)}/{len(checks)}")
    check("V03", "Timing measurements satisfy reproducibility thresholds", bool(timing["passed"]), json.dumps(timing, sort_keys=True))
    check("V04", "Both cost models contain the expected sample count", primary_cost.get("sample_count") == independent_cost.get("sample_count") == 288, f"{primary_cost.get('sample_count')}/{independent_cost.get('sample_count')}")

    write_rows(results_root / "reproducibility_checks.csv", validation_rows)
    passed = all(row["status"] == "PASS" for row in validation_rows)
    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "protocol_id": config.protocol_id,
        "protocol_sha256": config.sha256,
        "deterministic_artifact_count": len(checks),
        "deterministic_artifacts_matched": sum(bool(row["match"]) for row in checks),
        "all_deterministic_artifacts_match": all(bool(row["match"]) for row in checks),
        "timing_reproducibility": timing,
        "validation_passed": passed,
        "validation_check_count": len(validation_rows),
    }
    write_json(results_root / "reproducibility_summary.json", {**summary, "artifact_checks": checks})
    if not passed:
        raise RuntimeError("reproducibility checks did not all pass")
    return summary


def initialize_run(run_root: Path, run_label: str, config: ProtocolConfig) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_metadata.json", {
        "run_label": run_label,
        "schedule_identity": "identical deterministic schedule; execution label excluded from serialized objects",
        "protocol_id": config.protocol_id,
        "protocol_sha256": config.sha256,
        "experiment_version": EXPERIMENT_VERSION,
        "status": "INITIALIZED",
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quaternion security-attribution experiment runner")
    parser.add_argument("phase", choices=[
        "init", "differential-shard", "merge-differential", "leakage", "attacks",
        "timing-shard", "merge-timing", "cost-model", "reconcile",
    ])
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--run-label", default="primary_execution")
    parser.add_argument("--method")
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    config = load_protocol_config()
    if args.phase == "reconcile":
        if args.results_root is None:
            parser.error("--results-root is required for reconcile")
        payload = reconcile_two_runs(repo, args.results_root.resolve())
    else:
        if args.run_root is None:
            parser.error("--run-root is required")
        run_root = args.run_root.resolve()
        if args.phase == "init":
            initialize_run(run_root, args.run_label, config); payload = {"status": "initialized"}
        elif args.phase == "differential-shard":
            if not args.method: parser.error("--method is required")
            payload = run_differential_shard(repo, run_root, args.method)
        elif args.phase == "merge-differential":
            payload = merge_differential(repo, run_root)
        elif args.phase == "leakage":
            payload = run_leakage(repo, run_root)
        elif args.phase == "attacks":
            payload = run_attacks(repo, run_root)
        elif args.phase == "timing-shard":
            if not args.method: parser.error("--method is required")
            payload = run_timing_shard(repo, run_root, args.method)
        elif args.phase == "merge-timing":
            payload = merge_timing(repo, run_root)
        elif args.phase == "cost-model":
            payload = fit_cost_model(run_root, config)
        else:
            raise AssertionError(args.phase)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
