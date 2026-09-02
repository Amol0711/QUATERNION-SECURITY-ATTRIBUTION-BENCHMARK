from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import numpy as np
import pandas as pd
from scipy.stats import binom

from qsa_benchmark.attacks.attacks import apply_permutation_attack
from qsa_benchmark.benchmark.components import permutation_indices
from qsa_benchmark.benchmark.envelope import parse_envelope
from qsa_benchmark.benchmark.registry import make_method
from qsa_benchmark.benchmark.utils import canonical_json_bytes

from .config import load_protocol_config, repository_root
from .differential import ideal_uaci_moments, pairwise_flip_probability
from .experiment import (
    CSV_FLOAT_FORMAT,
    _apply_vectorized_inference,
    _context_for_object,
    canonical_csv_sha256,
    load_manifest,
    load_timing_images,
    run_differential_shard,
    write_json,
    write_rows,
)

TARGETED_VERSION = "QSA-TARGETED-VALIDATION-V1"
B13 = "B13_permutation_only"
B23 = "B23_secure_fixed_header"


def load_target_config(repo_root: Path) -> tuple[dict[str, Any], str]:
    config_path = repo_root / "configs/protocol/targeted_validation.json"
    schema_path = repo_root / "configs/protocol/targeted_validation.schema.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    parent = load_protocol_config()
    if payload["parent_protocol_id"] != parent.protocol_id:
        raise RuntimeError("parent protocol identifier mismatch")
    if payload["parent_protocol_sha256"] != parent.sha256:
        raise RuntimeError("parent protocol digest mismatch")
    if payload["b13_query_tightness"]["method_id"] != B13:
        raise RuntimeError("B13 target mismatch")
    if payload["b23_npcr_power"]["method_id"] != B23:
        raise RuntimeError("B23 target mismatch")
    return payload, digest


def minimum_queries(n: int, q: int) -> int:
    if n <= 0 or q < 2:
        raise ValueError("positive domain size and alphabet q>=2 required")
    k = 0
    capacity = 1
    while capacity < n:
        capacity *= q
        k += 1
    return k


def theoretical_class_counts(n: int, capacity: int) -> dict[str, int]:
    """Residue-class counts for indices 0,...,n-1 modulo ``capacity``."""
    if n <= 0 or capacity <= 0:
        raise ValueError("positive n and capacity required")
    if capacity >= n:
        return {
            "unique_codewords": n,
            "ambiguous_codewords": 0,
            "ambiguous_positions": 0,
            "minimum_class_size": 1,
            "maximum_class_size": 1,
        }
    base, remainder = divmod(n, capacity)
    minimum = base
    maximum = base + (1 if remainder else 0)
    if base >= 2:
        ambiguous_codewords = capacity
        ambiguous_positions = n
    else:
        ambiguous_codewords = remainder
        ambiguous_positions = 2 * remainder
    return {
        "unique_codewords": capacity,
        "ambiguous_codewords": ambiguous_codewords,
        "ambiguous_positions": ambiguous_positions,
        "minimum_class_size": minimum,
        "maximum_class_size": maximum,
    }


def observed_class_counts(codes: np.ndarray) -> dict[str, int]:
    _, counts = np.unique(np.asarray(codes, dtype=np.uint64), return_counts=True)
    ambiguous = counts[counts > 1]
    return {
        "unique_codewords": int(len(counts)),
        "ambiguous_codewords": int(len(ambiguous)),
        "ambiguous_positions": int(ambiguous.sum()) if len(ambiguous) else 0,
        "minimum_class_size": int(counts.min()),
        "maximum_class_size": int(counts.max()),
    }


def recover_position_codes(
    shape: tuple[int, int, int],
    context: Any,
    query_count: int,
    q: int = 256,
) -> dict[str, Any]:
    if q != 256:
        raise ValueError("the executable byte-position probes require q=256")
    n = int(np.prod(shape))
    k_star = minimum_queries(n, q)
    if query_count < 1 or query_count > k_star:
        raise ValueError("query_count must lie in [1,k_star]")
    positions = np.arange(n, dtype=np.uint64)
    method = make_method(B13, profile="extended")
    observed = np.zeros(n, dtype=np.uint64)
    digest = hashlib.sha256()
    capacity = 1
    for _digit in range(query_count):
        probe = ((positions // capacity) % q).astype(np.uint8).reshape(shape)
        protected = parse_envelope(method.encrypt(probe, context).object_bytes).protected_payload
        if len(protected) != n:
            raise RuntimeError("B13 probe length changed")
        digest.update(len(protected).to_bytes(8, "big"))
        digest.update(protected)
        observed += np.frombuffer(protected, dtype=np.uint8).astype(np.uint64) * capacity
        capacity *= q
    true_mapping = permutation_indices(n, context, B13).astype(np.uint64)
    expected_codes = true_mapping % capacity
    observed_counts = observed_class_counts(observed)
    theory = theoretical_class_counts(n, capacity)
    return {
        "n_positions": n,
        "alphabet_size": q,
        "k_star": k_star,
        "queries": query_count,
        "code_capacity": capacity,
        "code_consistency_accuracy": float(np.mean(observed == expected_codes)),
        "mapping_accuracy": float(np.mean(observed == true_mapping)),
        "mapping_identifiable": bool(capacity >= n and np.array_equal(observed, true_mapping)),
        "mapping": observed,
        "true_mapping": true_mapping,
        "mapping_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
        "true_mapping_sha256": hashlib.sha256(true_mapping.tobytes()).hexdigest(),
        "probe_ciphertexts_sha256": digest.hexdigest(),
        **observed_counts,
        **{f"theoretical_{key}": value for key, value in theory.items()},
    }


def target_image(repo_root: Path, size: int, target_config: Mapping[str, Any]) -> np.ndarray:
    image_id = str(target_config["b13_query_tightness"]["image_id"])
    if size == 96:
        parent = load_protocol_config()
        return load_timing_images(repo_root, parent.payload)[(image_id, size)]
    _, images, _ = load_manifest(repo_root)
    return images[(image_id, size)]


def _alternative_consistent_mapping(
    observed_codes: np.ndarray,
    true_mapping: np.ndarray,
    target_flat: np.ndarray,
) -> tuple[np.ndarray | None, bool, bool]:
    """Construct one different permutation consistent with an insufficient transcript.

    Two output positions in the same observed-code class are swapped.  The search
    prefers a pair mapping to unequal target bytes, making target ambiguity explicit.
    """
    order = np.argsort(observed_codes, kind="mergesort")
    sorted_codes = observed_codes[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]
    fallback: tuple[int, int] | None = None
    chosen: tuple[int, int] | None = None
    for start, stop in zip(starts, stops, strict=True):
        group = order[start:stop]
        if len(group) < 2:
            continue
        if fallback is None:
            fallback = (int(group[0]), int(group[1]))
        values = target_flat[true_mapping[group].astype(np.int64)]
        for local in range(1, len(group)):
            if values[local] != values[0]:
                chosen = (int(group[0]), int(group[local]))
                break
        if chosen is not None:
            break
    pair = chosen or fallback
    if pair is None:
        return None, False, False
    alternative = true_mapping.copy()
    a, b = pair
    alternative[a], alternative[b] = alternative[b], alternative[a]
    # Transcript consistency is checked by direct reduction modulo the known
    # code capacity in the calling routine.
    target_changes = bool(target_flat[true_mapping[a]] != target_flat[true_mapping[b]])
    return alternative, True, target_changes


def run_b13(repo_root: Path, run_root: Path, target_config: Mapping[str, Any]) -> dict[str, Any]:
    parent = load_protocol_config()
    cfg = target_config["b13_query_tightness"]
    q = int(cfg["alphabet_size"])
    rows: list[dict[str, Any]] = []
    method = make_method(B13, profile="extended")
    for size in [int(v) for v in cfg["sizes"]]:
        shape = (size, size, 3)
        target = target_image(repo_root, size, target_config)
        if tuple(target.shape) != shape:
            raise RuntimeError(f"B13 target shape mismatch at {size}: {target.shape}")
        target_flat = target.reshape(-1)
        n = int(target.size)
        k_star = minimum_queries(n, q)
        for repetition in range(int(cfg["context_repetitions"])):
            context = _context_for_object(
                parent, B13, str(cfg["image_id"]), size, repetition, "QSA-TARGETED-VALIDATION-B13"
            )
            target_cipher = parse_envelope(method.encrypt(target, context).object_bytes).protected_payload
            for query_count in (k_star - 1, k_star):
                result = recover_position_codes(shape, context, query_count, q)
                target_recovery_exact: bool | str = "NA"
                target_recovery_accuracy: float | str = "NA"
                recovered_sha256 = "NA"
                alternative_constructed = False
                alternative_code_consistent: bool | str = "NA"
                alternative_mapping_distinct: bool | str = "NA"
                alternative_target_recovery_exact: bool | str = "NA"
                alternative_target_changes: bool | str = "NA"
                if result["mapping_identifiable"]:
                    recovered = apply_permutation_attack(target_cipher, result["mapping"], shape)
                    target_recovery_exact = bool(np.array_equal(recovered, target))
                    target_recovery_accuracy = float(np.mean(recovered == target))
                    recovered_sha256 = hashlib.sha256(recovered.tobytes()).hexdigest()
                else:
                    alt, alternative_constructed, alternative_target_changes = _alternative_consistent_mapping(
                        result["mapping"], result["true_mapping"], target_flat
                    )
                    if alt is not None:
                        alternative_code_consistent = bool(
                            np.array_equal(alt % int(result["code_capacity"]), result["mapping"])
                        )
                        alternative_mapping_distinct = bool(not np.array_equal(alt, result["true_mapping"]))
                        alt_recovered = apply_permutation_attack(target_cipher, alt, shape)
                        alternative_target_recovery_exact = bool(np.array_equal(alt_recovered, target))
                row = {
                    "method_id": B13,
                    "image_id": str(cfg["image_id"]),
                    "image_size": size,
                    "shape": "x".join(str(v) for v in shape),
                    "context_repetition": repetition,
                    "context_nonce_sha256": hashlib.sha256(context.nonce).hexdigest(),
                    "n_positions": n,
                    "alphabet_size": q,
                    "k_star": k_star,
                    "query_regime": "k_star" if query_count == k_star else "k_star_minus_one",
                    "queries": query_count,
                    "code_capacity": result["code_capacity"],
                    "code_consistency_accuracy": result["code_consistency_accuracy"],
                    "mapping_accuracy": result["mapping_accuracy"],
                    "mapping_identifiable": result["mapping_identifiable"],
                    "mapping_sha256": result["mapping_sha256"],
                    "true_mapping_sha256": result["true_mapping_sha256"],
                    "probe_ciphertexts_sha256": result["probe_ciphertexts_sha256"],
                    "unique_codewords": result["unique_codewords"],
                    "ambiguous_codewords": result["ambiguous_codewords"],
                    "ambiguous_positions": result["ambiguous_positions"],
                    "minimum_class_size": result["minimum_class_size"],
                    "maximum_class_size": result["maximum_class_size"],
                    "theoretical_unique_codewords": result["theoretical_unique_codewords"],
                    "theoretical_ambiguous_codewords": result["theoretical_ambiguous_codewords"],
                    "theoretical_ambiguous_positions": result["theoretical_ambiguous_positions"],
                    "theoretical_minimum_class_size": result["theoretical_minimum_class_size"],
                    "theoretical_maximum_class_size": result["theoretical_maximum_class_size"],
                    "target_plaintext_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                    "target_ciphertext_sha256": hashlib.sha256(target_cipher).hexdigest(),
                    "target_recovery_exact": target_recovery_exact,
                    "target_recovery_accuracy": target_recovery_accuracy,
                    "target_recovered_sha256": recovered_sha256,
                    "alternative_mapping_constructed": alternative_constructed,
                    "alternative_mapping_code_consistent": alternative_code_consistent,
                    "alternative_mapping_distinct": alternative_mapping_distinct,
                    "alternative_target_recovery_exact": alternative_target_recovery_exact,
                    "alternative_target_changes": alternative_target_changes,
                }
                rows.append(row)
    record_path = run_root / "B13_query_tightness_records.csv"
    write_rows(record_path, rows)
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (size, regime), group in frame.groupby(["image_size", "query_regime"], sort=True):
        bool_ident = group["mapping_identifiable"].astype(str).str.lower().isin({"true", "1"})
        exact_target = group["target_recovery_exact"].astype(str).str.lower().isin({"true", "1"})
        alt_consistent = group["alternative_mapping_code_consistent"].astype(str).str.lower().isin({"true", "1"})
        alt_distinct = group["alternative_mapping_distinct"].astype(str).str.lower().isin({"true", "1"})
        alt_inexact = group["alternative_target_recovery_exact"].astype(str).str.lower().isin({"false", "0"})
        summary_rows.append({
            "method_id": B13,
            "image_size": int(size),
            "n_positions": int(group["n_positions"].iloc[0]),
            "alphabet_size": int(group["alphabet_size"].iloc[0]),
            "k_star": int(group["k_star"].iloc[0]),
            "query_regime": str(regime),
            "queries": int(group["queries"].iloc[0]),
            "context_repetitions": len(group),
            "code_consistency_all_exact": bool(np.allclose(group["code_consistency_accuracy"].astype(float), 1.0)),
            "mapping_identifiable_count": int(bool_ident.sum()),
            "mapping_accuracy": float(group["mapping_accuracy"].iloc[0]),
            "unique_codewords": int(group["unique_codewords"].iloc[0]),
            "ambiguous_codewords": int(group["ambiguous_codewords"].iloc[0]),
            "ambiguous_positions": int(group["ambiguous_positions"].iloc[0]),
            "minimum_class_size": int(group["minimum_class_size"].iloc[0]),
            "maximum_class_size": int(group["maximum_class_size"].iloc[0]),
            "target_recovery_exact_count": int(exact_target.sum()),
            "target_recovery_applicable_count": int((group["target_recovery_exact"].astype(str) != "NA").sum()),
            "alternative_consistent_distinct_count": int((alt_consistent & alt_distinct).sum()),
            "alternative_target_inexact_count": int(alt_inexact.sum()),
        })
    summary_path = run_root / "B13_query_tightness_summary.csv"
    write_rows(summary_path, summary_rows)
    manifest = {
        "record_count": len(rows),
        "summary_count": len(summary_rows),
        "record_canonical_sha256": canonical_csv_sha256(record_path),
        "summary_canonical_sha256": canonical_csv_sha256(summary_path),
    }
    write_json(run_root / "B13_query_tightness_manifest.json", manifest)
    return manifest


def summarize_b23(inferred: pd.DataFrame) -> list[dict[str, Any]]:
    eligible = inferred[inferred["inferential_eligible"].astype(str).str.lower().isin({"true", "1"})].copy()
    group_cols = [
        "tier_id", "method_id", "protocol", "protocol_code", "operation_regime",
        "image_size", "projection_id", "semantic_domain",
    ]
    mean0, var0 = ideal_uaci_moments(256)
    rows: list[dict[str, Any]] = []
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
        rows.append(row)
    return rows


def run_b23(repo_root: Path, run_root: Path, target_config: Mapping[str, Any]) -> dict[str, Any]:
    shard_manifest = run_differential_shard(repo_root, run_root, B23)
    shard_dir = run_root / "differential/shards"
    pair_raw = shard_dir / f"{B23}__pairs.csv"
    projection_raw = shard_dir / f"{B23}__projections_raw.csv"
    pairs = pd.read_csv(pair_raw, keep_default_na=False, low_memory=False)
    projections = pd.read_csv(projection_raw, keep_default_na=False, low_memory=False)
    pair_sort = ["tier_id", "method_id", "protocol", "image_size", "image_id", "perturbation_id", "key_index", "state_index"]
    proj_sort = pair_sort + ["projection_id"]
    pairs = pairs.sort_values(pair_sort, kind="mergesort")
    projections = projections.sort_values(proj_sort, kind="mergesort")
    alpha = float(target_config["b23_npcr_power"]["primary_familywise_alpha"])
    q = int(target_config["b23_npcr_power"]["alphabet_size"])
    inferred = _apply_vectorized_inference(projections, alpha=alpha, q=q)
    pair_path = run_root / "B23_differential_pair_records.csv"
    projection_path = run_root / "B23_differential_projection_records.csv"
    summary_path = run_root / "B23_differential_summary.csv"
    pairs.to_csv(pair_path, index=False, float_format=CSV_FLOAT_FORMAT)
    inferred.to_csv(projection_path, index=False, float_format=CSV_FLOAT_FORMAT, na_rep="NA")
    write_rows(summary_path, summarize_b23(inferred))
    if len(pairs) != 2976 or len(inferred) != 14880:
        raise RuntimeError(f"unexpected B23 targeted counts: {len(pairs)} pairs, {len(inferred)} projections")
    manifest = {
        "shard_manifest": shard_manifest,
        "pair_rows": len(pairs),
        "projection_rows": len(inferred),
        "summary_rows": len(pd.read_csv(summary_path)),
        "pair_canonical_sha256": canonical_csv_sha256(pair_path),
        "projection_canonical_sha256": canonical_csv_sha256(projection_path),
        "summary_canonical_sha256": canonical_csv_sha256(summary_path),
    }
    write_json(run_root / "B23_targeted_rerun_manifest.json", manifest)
    return manifest


def lower_critical_count(n: int, p: float, level: float) -> int:
    if not (0.0 < p < 1.0 and 0.0 < level < 1.0 and n >= 0):
        raise ValueError("invalid binomial critical-count arguments")
    candidate = int(binom.ppf(level, n, p))
    while candidate >= 0 and float(binom.cdf(candidate, n, p)) > level:
        candidate -= 1
    while candidate + 1 <= n and float(binom.cdf(candidate + 1, n, p)) <= level:
        candidate += 1
    return candidate


def exact_probability_checks(n: int, p: float, c: int) -> tuple[float, float]:
    cdf_changed = float(binom.cdf(c, n, p))
    equal_threshold = n - c
    sf_equal = float(binom.sf(equal_threshold - 1, n, 1.0 - p))
    return cdf_changed, sf_equal


def calibrate_b23(run_root: Path, target_config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = target_config["b23_npcr_power"]
    q = int(cfg["alphabet_size"])
    p = (q - 1.0) / q
    r = int(cfg["fixed_equal_coordinates"])
    alpha = float(cfg["primary_familywise_alpha"])
    frame = pd.read_csv(run_root / "B23_differential_projection_records.csv", keep_default_na=False, low_memory=False)
    full = frame[
        (frame["method_id"] == B23)
        & (frame["protocol"] == cfg["protocol"])
        & (frame["projection_id"] == cfg["projection_id"])
    ].copy()
    suffix = frame[
        (frame["method_id"] == B23)
        & (frame["protocol"] == cfg["protocol"])
        & (frame["projection_id"] == cfg["suffix_projection_id"])
    ].copy()
    rows: list[dict[str, Any]] = []
    for size in [int(v) for v in cfg["sizes"]]:
        group = full[full["image_size"].astype(int) == size].copy()
        suffix_group = suffix[suffix["image_size"].astype(int) == size].copy()
        if group.empty or suffix_group.empty:
            raise RuntimeError(f"missing B23 P2 projection rows for size {size}")
        n_values = set(group["sample_count"].astype(int))
        suffix_values = set(suffix_group["sample_count"].astype(int))
        if len(n_values) != 1 or len(suffix_values) != 1:
            raise RuntimeError("B23 projection lengths are not constant within a size")
        n = n_values.pop()
        n_suffix = suffix_values.pop()
        if n - n_suffix != r or n_suffix != 3 * size * size:
            raise RuntimeError(f"B23 fixed-prefix length mismatch at {size}: n={n}, suffix={n_suffix}")
        m = len(group)
        expected_m = int(cfg["expected_holm_family_sizes"][str(size)])
        if m != expected_m:
            raise RuntimeError(f"B23 family-size mismatch at {size}: {m} != {expected_m}")
        raw_c = lower_critical_count(n, p, alpha)
        raw_null, raw_null_equal = exact_probability_checks(n, p, raw_c)
        raw_power = float(binom.cdf(raw_c, n - r, p))
        holm_level = alpha / m
        holm_c = lower_critical_count(n, p, holm_level)
        holm_null, holm_null_equal = exact_probability_checks(n, p, holm_c)
        per_test_power = float(binom.cdf(holm_c, n - r, p))
        expected_first = m * per_test_power
        familywise_any = float(-math.expm1(m * math.log1p(-per_test_power)))
        bool_col = lambda name: group[name].astype(str).str.lower().isin({"true", "1"})
        suffix_bool = lambda name: suffix_group[name].astype(str).str.lower().isin({"true", "1"})
        observed_raw = int(bool_col("npcr_raw_reject").sum())
        observed_holm = int(bool_col("npcr_holm_reject").sum())
        observed_uaci = int(bool_col("uaci_holm_reject").sum())
        observed_joint_fail = int((~bool_col("joint_holm_pass")).sum())
        observed_first = int((group["npcr_p_value"].astype(float) <= holm_level).sum())
        if observed_first != observed_holm:
            raise RuntimeError(
                f"B23 Holm count at {size} is not exhausted by first-step exceedances: "
                f"first={observed_first}, Holm={observed_holm}"
            )
        pmf_observed = float(binom.pmf(observed_first, m, per_test_power))
        tail_observed_or_more = float(binom.sf(observed_first - 1, m, per_test_power)) if observed_first else 1.0
        pred_low = int(binom.ppf(0.025, m, per_test_power))
        pred_high = int(binom.ppf(0.975, m, per_test_power))
        t1 = group[group["tier_id"] == "T1_primary_all_methods"]
        t1_rejects = int(t1["npcr_holm_reject"].astype(str).str.lower().isin({"true", "1"}).sum())
        rows.append({
            "method_id": B23,
            "protocol": cfg["protocol"],
            "projection_id": cfg["projection_id"],
            "image_size": size,
            "alphabet_size": q,
            "fixed_equal_coordinates": r,
            "suffix_coordinates": n_suffix,
            "full_body_coordinates": n,
            "standardized_mean_shift": r * math.sqrt((q - 1.0) / n),
            "familywise_alpha": alpha,
            "holm_family_size": m,
            "raw_critical_count": raw_c,
            "raw_actual_null_size": raw_null,
            "raw_actual_null_size_equal_count_check": raw_null_equal,
            "raw_exact_per_test_power": raw_power,
            "holm_first_step_level": holm_level,
            "holm_first_step_critical_count": holm_c,
            "holm_first_step_actual_null_size": holm_null,
            "holm_first_step_null_size_equal_count_check": holm_null_equal,
            "holm_first_step_exact_per_test_power": per_test_power,
            "expected_first_step_rejections": expected_first,
            "exact_familywise_probability_any_holm_rejection": familywise_any,
            "observed_raw_npcr_rejections": observed_raw,
            "observed_first_step_exceedances": observed_first,
            "observed_holm_npcr_rejections": observed_holm,
            "observed_holm_uaci_rejections": observed_uaci,
            "observed_joint_failures": observed_joint_fail,
            "observed_t1_rows": len(t1),
            "observed_t1_holm_npcr_rejections": t1_rejects,
            "observed_count_exact_probability": pmf_observed,
            "observed_or_more_tail_probability": tail_observed_or_more,
            "equal_tailed_95_prediction_low": pred_low,
            "equal_tailed_95_prediction_high": pred_high,
            "observed_within_equal_tailed_95_prediction_interval": pred_low <= observed_first <= pred_high,
            "suffix_holm_npcr_rejections": int(suffix_bool("npcr_holm_reject").sum()),
            "suffix_holm_uaci_rejections": int(suffix_bool("uaci_holm_reject").sum()),
            "direct_fixed_prefix_advantage": f"1-256^-{r}=1-2^-{8*r}",
            "model_scope": "exact under independent ideal suffix-coordinate pairs and independent family rows; B23 comparison is calibration, not a primitive-security proof",
        })
    path = run_root / "B23_exact_multiplicity_power_calibration.csv"
    write_rows(path, rows)
    manifest = {
        "row_count": len(rows),
        "canonical_sha256": canonical_csv_sha256(path),
        "all_null_dual_representations_match": all(
            math.isclose(float(row["raw_actual_null_size"]), float(row["raw_actual_null_size_equal_count_check"]), rel_tol=1e-12, abs_tol=1e-15)
            and math.isclose(float(row["holm_first_step_actual_null_size"]), float(row["holm_first_step_null_size_equal_count_check"]), rel_tol=1e-12, abs_tol=1e-15)
            for row in rows
        ),
        "all_observed_counts_within_95_prediction_intervals": all(
            bool(row["observed_within_equal_tailed_95_prediction_interval"]) for row in rows
        ),
    }
    write_json(run_root / "B23_exact_multiplicity_power_manifest.json", manifest)
    return manifest


def initialize_run(repo_root: Path, run_root: Path, run_label: str) -> dict[str, Any]:
    target_config, target_digest = load_target_config(repo_root)
    run_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol_id": target_config["protocol_id"],
        "protocol_sha256": target_digest,
        "parent_protocol_id": target_config["parent_protocol_id"],
        "parent_protocol_sha256": target_config["parent_protocol_sha256"],
        "run_label": run_label,
        "schedule_identity": "identical targeted schedule; execution label excluded from construction contexts and serialized objects",
        "status": "INITIALIZED",
    }
    write_json(run_root / "run_metadata.json", metadata)
    return metadata


def execute_run(repo_root: Path, run_root: Path, run_label: str) -> dict[str, Any]:
    target_config, target_digest = load_target_config(repo_root)
    initialize_run(repo_root, run_root, run_label)
    b13_manifest = run_b13(repo_root, run_root, target_config)
    b23_manifest = run_b23(repo_root, run_root, target_config)
    power_manifest = calibrate_b23(run_root, target_config)
    manifest = {
        "protocol_sha256": target_digest,
        "run_label": run_label,
        "b13": b13_manifest,
        "b23_experiment": b23_manifest,
        "b23_power": power_manifest,
        "status": "COMPLETE",
    }
    write_json(run_root / "targeted_run_manifest.json", manifest)
    return manifest


def reconcile_targeted_runs(repo_root: Path, results_root: Path) -> dict[str, Any]:
    target_config, target_digest = load_target_config(repo_root)
    primary = results_root / "primary_execution"
    independent = results_root / "independent_execution"
    deterministic_names = [
        "B13_query_tightness_records.csv",
        "B13_query_tightness_summary.csv",
        "B23_differential_pair_records.csv",
        "B23_differential_projection_records.csv",
        "B23_differential_summary.csv",
        "B23_exact_multiplicity_power_calibration.csv",
    ]
    artifact_checks: list[dict[str, Any]] = []
    for name in deterministic_names:
        first_path = primary / name
        second_path = independent / name
        first_digest = canonical_csv_sha256(first_path) if first_path.exists() else "MISSING"
        second_digest = canonical_csv_sha256(second_path) if second_path.exists() else "MISSING"
        artifact_checks.append({
            "artifact": name,
            "primary_sha256": first_digest,
            "independent_sha256": second_digest,
            "match": first_digest == second_digest and first_digest != "MISSING",
        })

    b13 = pd.read_csv(primary / "B13_query_tightness_records.csv", keep_default_na=False, low_memory=False)
    b23 = pd.read_csv(primary / "B23_exact_multiplicity_power_calibration.csv", keep_default_na=False, low_memory=False)
    kstar = b13[b13["query_regime"] == "k_star"]
    previous = b13[b13["query_regime"] == "k_star_minus_one"]
    checks: list[dict[str, Any]] = []
    def check(identifier: str, description: str, passed: bool, evidence: str) -> None:
        checks.append({"check_id": identifier, "description": description, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    check("T01", "Both targeted execution trees are complete", (primary / "targeted_run_manifest.json").exists() and (independent / "targeted_run_manifest.json").exists(), f"{primary}; {independent}")
    check("T02", "Deterministic targeted artifacts match", all(bool(row["match"]) for row in artifact_checks), f"{sum(bool(row['match']) for row in artifact_checks)}/{len(artifact_checks)}")
    check("T03", "B13 covers three sizes, two query regimes, and four contexts", len(b13) == 24 and set(b13["image_size"].astype(int)) == {96, 256, 512} and set(b13["query_regime"]) == {"k_star", "k_star_minus_one"} and b13["context_repetition"].nunique() == 4, f"rows={len(b13)}")
    check("T04", "B13 recovers every permutation at k-star", len(kstar) == 12 and kstar["mapping_identifiable"].astype(str).str.lower().isin({"true", "1"}).all() and np.allclose(kstar["mapping_accuracy"].astype(float), 1.0), f"rows={len(kstar)}")
    check("T05", "B13 remains ambiguous at k-star-minus-one", len(previous) == 12 and not previous["mapping_identifiable"].astype(str).str.lower().isin({"true", "1"}).any() and (previous["ambiguous_positions"].astype(int) == previous["n_positions"].astype(int)).all(), f"rows={len(previous)}")
    check("T06", "B23 fixed-prefix and Holm family sizes match the configured model", set(zip(b23["image_size"].astype(int), b23["fixed_equal_coordinates"].astype(int), b23["holm_family_size"].astype(int))) == {(256, 32, 960), (512, 32, 528)}, "r=32; m=960/528")
    check("T07", "B23 exact null probabilities agree in both binomial representations", np.allclose(b23["raw_actual_null_size"].astype(float), b23["raw_actual_null_size_equal_count_check"].astype(float), rtol=1e-12, atol=1e-15) and np.allclose(b23["holm_first_step_actual_null_size"].astype(float), b23["holm_first_step_null_size_equal_count_check"].astype(float), rtol=1e-12, atol=1e-15), "dual representations")
    check("T08", "B23 observed counts lie in the exact 95-percent predictive intervals", b23["observed_within_equal_tailed_95_prediction_interval"].astype(str).str.lower().isin({"true", "1"}).all(), "both sizes")
    check("T09", "B23 suffix has no registered Holm rejection", (b23["suffix_holm_npcr_rejections"].astype(int) == 0).all() and (b23["suffix_holm_uaci_rejections"].astype(int) == 0).all(), "zero suffix rejections")

    write_rows(results_root / "targeted_reproducibility_checks.csv", checks)
    passed = all(row["status"] == "PASS" for row in checks)
    summary = {
        "targeted_version": TARGETED_VERSION,
        "protocol_id": target_config["protocol_id"],
        "protocol_sha256": target_digest,
        "artifact_count": len(artifact_checks),
        "matched_artifact_count": sum(bool(row["match"]) for row in artifact_checks),
        "all_deterministic_artifacts_match": all(bool(row["match"]) for row in artifact_checks),
        "validation_passed": passed,
        "validation_check_count": len(checks),
        "artifact_checks": artifact_checks,
    }
    write_json(results_root / "targeted_reproducibility_summary.json", summary)
    if not passed:
        raise RuntimeError("targeted reproducibility checks did not all pass")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Targeted security-attribution validation")
    parser.add_argument("phase", choices=["run", "reconcile"])
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-label", default="primary_execution")
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    if args.phase == "run":
        if args.run_root is None:
            parser.error("--run-root is required")
        result = execute_run(repo, args.run_root.resolve(), args.run_label)
    else:
        if args.results_root is None:
            parser.error("--results-root is required")
        result = reconcile_targeted_runs(repo, args.results_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
