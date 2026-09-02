from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import jsonschema

from qsa_benchmark.protocol.config import load_protocol_config
from qsa_benchmark.protocol.primitive_scaling import load_primitive_config
from qsa_benchmark.protocol.registry_export import (
    check_registry_files,
    registry_summary,
    verify_sha256sums,
)
from qsa_benchmark.validation.deterministic import (
    compare_execution_trees,
    deterministic_reference_summary,
    verify_execution_tree,
    verify_static_artifact_contract,
)
from qsa_benchmark.validation.environment import compare_environment
from qsa_benchmark.validation.known_answers import known_answer_summary
from qsa_benchmark.validation.malformed import malformed_object_summary
from qsa_benchmark.validation.neutrality import prospective_archive_safety, scan_repository

OUTPUT_MARKER_NAME = ".qsa_verification_output"
CONFIG_PATH = "configs/verification/repository_verification.json"
SCHEMA_PATH = "configs/verification/repository_verification.schema.json"


def load_verification_config(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    payload = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    schema = json.loads((root / SCHEMA_PATH).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(payload)
    return payload


def _prepare_output(path: Path) -> None:
    if path.exists():
        marker = path / OUTPUT_MARKER_NAME
        if any(path.iterdir()) and not marker.is_file():
            raise RuntimeError("verification output directory exists without the repository-owned marker")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / OUTPUT_MARKER_NAME).write_text("QSA repository verification output\n", encoding="utf-8", newline="\n")


def _safe_message(text: str, root: Path, output: Path) -> str:
    return str(text).replace(str(root), "<repository>").replace(str(output), "<verification-output>")


def _run_command(
    root: Path,
    command: list[str],
    log_path: Path,
    *,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run a repository command with file-backed output capture.

    Direct file capture avoids pipe-inheritance deadlocks from numerical or
    cryptographic runtimes while preserving a complete validation log.
    """
    env = os.environ.copy()
    source = str(root / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        try:
            completed = subprocess.run(
                command,
                env=env,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            handle.write("\nCommand exceeded the registered timeout.\n")
            handle.flush()
            returncode = 124
            timed_out = True
    return {"returncode": returncode, "log": log_path.name, "timed_out": timed_out}


def _write_summary(output: Path, summary: dict[str, Any]) -> None:
    (output / "verification_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = [
        "Quaternion security-attribution repository verification",
        f"Tier: {summary['executed_tier']}",
        f"Status: {summary['status']}",
        f"Checks passed: {summary['passed_check_count']}/{summary['check_count']}",
    ]
    if summary.get("failures"):
        lines.append("Failures:")
        lines.extend(f"- {item}" for item in summary["failures"])
    (output / "verification_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_repository_verification(
    repo_root: str | Path,
    *,
    output_root: str | Path,
    static: bool = False,
    full: bool = False,
    timing: bool = False,
    environment_policy: str = "exact",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    if timing and not full:
        raise ValueError("timing verification requires full verification")
    mode = "static" if static else "timing" if timing else "full" if full else "default"
    _prepare_output(output)
    announce = progress or (lambda _message: None)
    config = load_verification_config(root)
    expected = config["expected_counts"]
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    environment_failed = False
    timing_failed = False

    def check(identifier: str, description: str, passed: bool, evidence: Any = None, *, category: str = "computational") -> None:
        nonlocal environment_failed, timing_failed
        row = {"check_id": identifier, "description": description, "status": "PASS" if passed else "FAIL"}
        if evidence is not None:
            row["evidence"] = evidence
        checks.append(row)
        if not passed:
            failures.append(description)
            if category == "environment":
                environment_failed = True
            if category == "timing":
                timing_failed = True

    if not static:
        announce("V1: executing isolated test, benchmark, and reference-validation commands")
        bundle_command = [
            sys.executable,
            str(root / "scripts/run_verification_commands.py"),
            "--repo-root",
            str(root),
            "--output-root",
            str(output),
        ]
        if full:
            bundle_command.append("--full")
        if timing:
            bundle_command.append("--timing")
        bundle_result = _run_command(
            root,
            bundle_command,
            output / "isolated_command_bundle.log",
            timeout_seconds=28800 if full else 3600,
        )
        bundle_path = output / "isolated_command_summary.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8")) if bundle_path.is_file() else {}
        commands = bundle.get("commands", {})

        pytest_result = commands.get("pytest", {})
        check("V1-04", "Automated test suite passes", int(pytest_result.get("returncode", 1)) == 0, pytest_result or bundle_result)

        smoke_result = commands.get("smoke", {})
        fingerprint = bundle.get("smoke_fingerprint", {})
        frozen_fingerprint = json.loads((root / config["paths"]["compact_fingerprint"]).read_text(encoding="utf-8"))
        smoke_ok = int(smoke_result.get("returncode", 1)) == 0 and fingerprint.get("sha256") == frozen_fingerprint["deterministic_fingerprint_sha256"] and int(fingerprint.get("run_count", 0)) == int(frozen_fingerprint["run_count"])
        check("V1-05", "Compact 20-construction benchmark matches the frozen fingerprint", smoke_ok, {"command": smoke_result, "fingerprint": fingerprint})

        reference_summary = bundle.get("reference_layers", {})
        known_record = reference_summary.get("known_answers", {})
        malformed_record = reference_summary.get("malformed_objects", {})
        active_record = reference_summary.get("active_modifications", {})
        known_case_count = int(known_record.get("summary", {}).get("case_count", 0))
        malformed_case_count = int(malformed_record.get("summary", {}).get("case_count", 0))
        active_counts = active_record.get("counts", {})
        reference_command = commands.get("reference_layers", {})
        reference_ok = int(reference_command.get("returncode", 1)) == 0
        check("V1-01", "All 24 fixed known-answer cases match", reference_ok and known_record.get("status") == "PASS" and known_case_count == int(expected["known_answer_cases"]), known_record or reference_command)
        check("V1-02", "All 33 malformed-object vectors produce their registered parser outcomes", reference_ok and malformed_record.get("status") == "PASS" and malformed_case_count == int(expected["malformed_object_vectors"]), malformed_record or reference_command)
        active_ok = reference_ok and active_record.get("status") == "PASS" and int(active_record.get("row_count", 0)) == int(expected["active_modification_cases"]) and int(active_counts.get("authenticated_cases", 0)) == int(expected["authenticated_active_modifications"]) and int(active_counts.get("authenticated_rejections", 0)) == int(expected["authenticated_active_modifications"])
        check("V1-03", "The complete 180-case active-modification schedule matches its expected outcomes", active_ok, active_record or reference_command)

        if full:
            announce("V2: reconciling isolated full-execution results")
            experiment_root = output / "experiment"
            result = commands.get("full_experiment", {})
            check("V2-01", "Two complete registered execution trees finish successfully", int(result.get("returncode", 1)) == 0, result or bundle_result)
            if int(result.get("returncode", 1)) == 0:
                primary = verify_execution_tree(root, experiment_root / "primary_execution")
                independent = verify_execution_tree(root, experiment_root / "independent_execution")
                cross = compare_execution_trees(root, experiment_root / "primary_execution", experiment_root / "independent_execution")
                check("V2-02", "Primary execution matches all 37 frozen deterministic artifacts", primary["status"] == "PASS", {key: value for key, value in primary.items() if key != "checks"})
                check("V2-03", "Independent execution matches all 37 frozen deterministic artifacts", independent["status"] == "PASS", {key: value for key, value in independent.items() if key != "checks"})
                check("V2-04", "The two execution trees agree on all 37 deterministic artifacts", cross["status"] == "PASS", {key: value for key, value in cross.items() if key != "checks"})

            targeted = commands.get("targeted_validation", {})
            targeted_record = bundle.get("targeted_validation", {})
            check("V2-05", "Targeted validation passes in two execution trees", int(targeted.get("returncode", 1)) == 0 and bool(targeted_record.get("validation_passed")), {"command": targeted, "summary": targeted_record})

            primitive_manifests = commands.get("primitive_scaling", [])
            total_exact = sum(int(item.get("round_trip_cases_exact", 0)) for item in primitive_manifests)
            primitive_ok = len(primitive_manifests) == 2 and all(int(item.get("returncode", 1)) == 0 for item in primitive_manifests) and total_exact == int(expected["primitive_round_trips_across_two_executions"])
            check("V2-06", "Both primitive-scaling executions pass all 96 exact round trips", primitive_ok, primitive_manifests)

            if timing:
                timing_record = bundle.get("timing_reconciliation", {})
                timing_ok = bool(timing_record.get("passed")) and int(timing_record.get("configuration_count", 0)) == int(expected["timing_configurations"])
                check("V3-01", "All 576 complete-path timing configurations satisfy the registered limits", timing_ok, timing_record, category="timing")
                primitive_timing_ok = len(primitive_manifests) == 2 and all(int(item.get("timing_records", 0)) == int(expected["primitive_timing_records_per_execution"]) for item in primitive_manifests)
                check("V3-02", "Primitive-scaling timing produces 1,440 records per execution", primitive_timing_ok, primitive_manifests, category="timing")

    # Execute child processes before in-process integrity inspection. Several
    # numerical and cryptographic runtimes initialize worker state during V0;
    # launching child commands first prevents fork-after-runtime deadlocks.
    announce("V0: validating public identity, environment, integrity, neutrality, schemas, and registries")
    project = root / "pyproject.toml"
    license_path = root / "LICENSE.txt"
    check("V0-01", "Project metadata and license are present", project.is_file() and license_path.is_file())

    environment = compare_environment(root, environment_policy)
    check("V0-02", "Software environment satisfies the selected policy", environment["status"] == "PASS", environment, category="environment")

    neutrality = scan_repository(root)
    check("V0-03", "Public repository contains no publication-development or personal metadata", neutrality["status"] == "PASS", neutrality)
    safety = prospective_archive_safety(root)
    check("V0-04", "Prospective archive paths and file modes are safe", safety["status"] == "PASS", safety)

    checksum_errors = verify_sha256sums(root)
    check("V0-05", "Release checksum inventory matches every public file", not checksum_errors, checksum_errors)
    registry_errors = check_registry_files(root)
    check("V0-06", "Committed registries match their executable authorities", not registry_errors, registry_errors)

    registry = registry_summary(root)
    invariants = registry["invariants"]
    count_pairs = {
        "registered_constructions": invariants["registered_constructions"],
        "authenticated_constructions": invariants["authenticated_constructions"],
        "active_modification_cases": invariants["active_modification_cases"],
        "authenticated_active_modifications": invariants["authenticated_active_modifications"],
        "timing_configurations": invariants["timing_configurations"],
    }
    check("V0-07", "Registered protocol counts match the verification contract", all(int(count_pairs[key]) == int(expected[key]) for key in count_pairs), count_pairs)

    deterministic_errors = verify_static_artifact_contract(root)
    check("V0-08", "Frozen deterministic-artifact contract is internally consistent", not deterministic_errors, deterministic_errors)
    primitive = load_primitive_config(root)
    check("V0-09", "Primitive-scaling factorization matches 48 cases per execution", primitive["expected_counts"] == {
        "round_trip_cases_per_execution": 48,
        "round_trip_cases_across_two_executions": 96,
        "timing_records_per_execution": 1440,
        "timing_aggregate_rows_per_execution": 48,
    }, primitive["expected_counts"])
    protocol = load_protocol_config(root / "configs/protocol/experiment.json", root / "configs/protocol/experiment.schema.json")
    check("V0-10", "Authoritative protocol contains 24 constructions", len(protocol.methods) == int(expected["registered_constructions"]), {"method_count": len(protocol.methods)})

    documented_paths = [
        "configs/benchmark/core_full.yaml",
        "configs/verification/repository_verification.json",
        "reference/deterministic_outputs/artifact_hashes.json",
        "reference/verification/compact_fingerprint.json",
        "scripts/lock_environment.py",
        "scripts/run_primitive_scaling.py",
        "scripts/verify_reference_layers.py",
        "src/qsa_benchmark/validation/repository.py",
    ]
    missing_documented = [path for path in documented_paths if not (root / path).is_file()]
    check("V0-11", "Every documented verification path is present", not missing_documented, missing_documented)


    passed = sum(row["status"] == "PASS" for row in checks)
    if environment_failed:
        exit_code = 3
    elif timing_failed:
        exit_code = 5
    elif failures:
        exit_code = 2 if static else 4
    else:
        exit_code = 0
    summary = {
        "verification_id": config["verification_id"],
        "executed_tier": mode,
        "environment_policy": environment_policy,
        "status": "PASS" if not failures else "FAIL",
        "exit_code": exit_code,
        "check_count": len(checks),
        "passed_check_count": passed,
        "failures": [_safe_message(item, root, output) for item in failures],
        "checks": checks,
        "reference_layers": {
            "registries": registry,
            "known_answers": known_answer_summary(root) if (root / "reference/known_answers/manifest.json").is_file() else {},
            "malformed_objects": malformed_object_summary(root) if (root / "reference/malformed_objects/manifest.json").is_file() else {},
            "deterministic_outputs": deterministic_reference_summary(root),
        },
        "output_paths": ["verification_summary.json", "verification_summary.txt"],
    }
    _write_summary(output, summary)
    return summary
