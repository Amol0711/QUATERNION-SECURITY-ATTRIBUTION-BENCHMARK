#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MARKER = ".qsa_verification_output"


def _bootstrap(root: Path) -> None:
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _prepare_output(path: Path) -> None:
    if path.exists():
        marker = path / MARKER
        if any(path.iterdir()) and not marker.is_file():
            raise RuntimeError(
                "verification output directory exists without the repository-owned marker"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / MARKER).write_text(
        "QSA repository verification output\n", encoding="utf-8", newline="\n"
    )


def _run_stage(
    root: Path,
    output: Path,
    name: str,
    command: list[str],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    log = output / f"{name}.log"
    timed_out = False
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            handle.write("\nCommand exceeded the registered timeout.\n")
            handle.flush()
            returncode = 124
            timed_out = True
    return {"returncode": returncode, "log": log.name, "timed_out": timed_out}



def _run_parallel(
    root: Path,
    output: Path,
    stages: list[tuple[str, list[str], int]],
) -> dict[str, dict[str, Any]]:
    """Launch independent verification stages together and collect their status."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    running: dict[str, dict[str, Any]] = {}
    for name, command, timeout_seconds in stages:
        log = output / f"{name}.log"
        handle = log.open("w", encoding="utf-8", newline="\n")
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running[name] = {
            "process": process,
            "handle": handle,
            "deadline": time.monotonic() + timeout_seconds,
            "log": log,
            "timed_out": False,
        }
    results: dict[str, dict[str, Any]] = {}
    while running:
        now = time.monotonic()
        completed_names: list[str] = []
        for name, record in running.items():
            process = record["process"]
            returncode = process.poll()
            if returncode is None and now >= record["deadline"]:
                process.kill()
                returncode = process.wait()
                record["handle"].write("\nCommand exceeded the registered timeout.\n")
                record["handle"].flush()
                record["timed_out"] = True
            if returncode is not None:
                record["handle"].close()
                results[name] = {
                    "returncode": int(returncode),
                    "log": record["log"].name,
                    "timed_out": bool(record["timed_out"]),
                }
                completed_names.append(name)
        for name in completed_names:
            del running[name]
        if running:
            time.sleep(0.1)
    return results

def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _run_bundle(
    root: Path,
    output: Path,
    *,
    full: bool,
    timing: bool,
    environment_policy: str,
) -> dict[str, Any]:
    """Run every child command before importing computational modules."""
    commands: dict[str, Any] = {}
    integrity_path = output / "integrity_layers_summary.json"
    commands.update(
        _run_parallel(
            root,
            output,
            [
                (
                    "pytest",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-c",
                        str(root / "pyproject.toml"),
                        str(root / "tests"),
                    ],
                    600,
                ),
                (
                    "smoke",
                    [sys.executable, str(root / "scripts/run_smoke.py")],
                    1800,
                ),
                (
                    "integrity_layers",
                    [
                        sys.executable,
                        str(root / "scripts/verify_integrity_layers.py"),
                        "--repo-root",
                        str(root),
                        "--output-root",
                        str(output),
                        "--environment-policy",
                        environment_policy,
                    ],
                    1800,
                ),
            ],
        )
    )

    default_passed = all(
        int(commands[name]["returncode"]) == 0
        for name in ("pytest", "smoke", "integrity_layers")
    )
    if full and default_passed:
        experiment_root = output / "experiment"
        experiment_command = [
            sys.executable,
            str(root / "scripts/run_experiment.py"),
            "--repo-root",
            str(root),
            "--results-root",
            str(experiment_root),
        ]
        if not timing:
            experiment_command.append("--skip-timing")
        commands["full_experiment"] = _run_stage(
            root,
            output,
            "full_experiment",
            experiment_command,
            timeout_seconds=21600,
        )
        if int(commands["full_experiment"].get("returncode", 1)) == 0:
            deterministic_result = _run_stage(
                root,
                output,
                "deterministic_reference",
                [
                    sys.executable,
                    str(root / "scripts/generate_reference_assets.py"),
                    "--repo-root",
                    str(root),
                    "deterministic-artifacts",
                    "--check",
                    "--results-root",
                    str(experiment_root),
                ],
                timeout_seconds=1800,
            )
            deterministic_result["summary"] = _read_json(
                output / "deterministic_reference.log"
            )
            commands["deterministic_reference"] = deterministic_result
        targeted_root = output / "targeted_validation"
        commands["targeted_validation"] = _run_stage(
            root,
            output,
            "targeted_validation",
            [
                sys.executable,
                str(root / "scripts/run_targeted_validation.py"),
                "--repo-root",
                str(root),
                "--results-root",
                str(targeted_root),
            ],
            timeout_seconds=3600,
        )
        primitive_commands: list[dict[str, Any]] = []
        for label in ("primary_execution", "independent_execution"):
            primitive_root = output / "primitive_scaling" / label
            command = [
                sys.executable,
                str(root / "scripts/run_primitive_scaling.py"),
                "--repo-root",
                str(root),
                "--results-root",
                str(primitive_root),
                "--run-label",
                label,
            ]
            if timing:
                command.append("--timing")
            result = _run_stage(
                root,
                output,
                f"primitive_{label}",
                command,
                timeout_seconds=3600,
            )
            manifest = _read_json(
                primitive_root / "primitive_scaling_manifest.json"
            )
            primitive_commands.append(
                {
                    "label": label,
                    "returncode": result["returncode"],
                    "log": result["log"],
                    "timed_out": result["timed_out"],
                    "round_trip_cases_exact": int(
                        manifest.get("round_trip_cases_exact", 0)
                    ),
                    "timing_records": int(manifest.get("timing_records", 0)),
                }
            )
        commands["primitive_scaling"] = primitive_commands

    summary = {
        "verification_id": "QSA-ISOLATED-COMMAND-BUNDLE-V1",
        "mode": "timing" if timing else "full" if full else "default",
        "commands": commands,
        "reference_layers": _read_json(integrity_path).get("reference_layers", {}),
        "static_summary": _read_json(integrity_path).get("static_summary", {}),
        "smoke_fingerprint": _read_json(
            root / "results/benchmark/smoke/deterministic_fingerprint.json"
        ),
        "targeted_validation": {
            "validation_passed": bool(
                _read_json(
                    output
                    / "targeted_validation/targeted_reproducibility_summary.json"
                ).get("validation_passed")
            )
        },
        "timing_reconciliation": _read_json(
            output / "experiment/reproducibility_summary.json"
        ).get("timing_reproducibility", {}),
    }
    command_results: list[int] = []
    for value in commands.values():
        if isinstance(value, list):
            command_results.extend(int(item.get("returncode", 1)) for item in value)
        else:
            command_results.append(int(value.get("returncode", 1)))
    summary["status"] = (
        "PASS"
        if command_results and all(code == 0 for code in command_results)
        else "FAIL"
    )
    summary["exit_code"] = 0 if summary["status"] == "PASS" else 2
    (output / "isolated_command_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary

def _write_summary(output: Path, summary: dict[str, Any]) -> None:
    (output / "verification_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "Quaternion security-attribution repository verification",
        f"Tier: {summary['executed_tier']}",
        f"Status: {summary['status']}",
        f"Checks passed: {summary['passed_check_count']}/{summary['check_count']}",
    ]
    if summary.get("failures"):
        lines.append("Failures:")
        lines.extend(f"- {item}" for item in summary["failures"])
    (output / "verification_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _merge_verification(
    root: Path,
    output: Path,
    config: dict[str, Any],
    bundle: dict[str, Any],
    static_summary: dict[str, Any],
    *,
    full: bool,
    timing: bool,
    environment_policy: str,
) -> dict[str, Any]:
    expected = config["expected_counts"]
    checks = list(static_summary.get("checks", []))
    failures = list(static_summary.get("failures", []))
    timing_failed = False

    def check(
        identifier: str,
        description: str,
        passed: bool,
        evidence: Any = None,
        *,
        category: str = "computational",
    ) -> None:
        nonlocal timing_failed
        row: dict[str, Any] = {
            "check_id": identifier,
            "description": description,
            "status": "PASS" if passed else "FAIL",
        }
        if evidence is not None:
            row["evidence"] = evidence
        checks.append(row)
        if not passed:
            failures.append(description)
            if category == "timing":
                timing_failed = True

    commands = bundle.get("commands", {})
    pytest_result = commands.get("pytest", {})
    check(
        "V1-04",
        "Automated test suite passes",
        int(pytest_result.get("returncode", 1)) == 0,
        pytest_result,
    )

    smoke_result = commands.get("smoke", {})
    fingerprint = bundle.get("smoke_fingerprint", {})
    frozen_fingerprint = json.loads(
        (root / config["paths"]["compact_fingerprint"]).read_text(encoding="utf-8")
    )
    smoke_ok = (
        int(smoke_result.get("returncode", 1)) == 0
        and fingerprint.get("sha256")
        == frozen_fingerprint["deterministic_fingerprint_sha256"]
        and int(fingerprint.get("run_count", 0))
        == int(frozen_fingerprint["run_count"])
    )
    check(
        "V1-05",
        "Compact 20-construction benchmark matches the frozen fingerprint",
        smoke_ok,
        {"command": smoke_result, "fingerprint": fingerprint},
    )

    reference = bundle.get("reference_layers", {})
    known = reference.get("known_answers", {})
    malformed = reference.get("malformed_objects", {})
    active = reference.get("active_modifications", {})
    reference_command = commands.get("integrity_layers", {})
    reference_ok = int(reference_command.get("returncode", 1)) == 0
    check(
        "V1-01",
        "All 24 fixed known-answer cases match",
        reference_ok
        and known.get("status") == "PASS"
        and int(known.get("summary", {}).get("case_count", 0))
        == int(expected["known_answer_cases"]),
        known,
    )
    check(
        "V1-02",
        "All 33 malformed-object vectors produce their registered parser outcomes",
        reference_ok
        and malformed.get("status") == "PASS"
        and int(malformed.get("summary", {}).get("case_count", 0))
        == int(expected["malformed_object_vectors"]),
        malformed,
    )
    active_counts = active.get("counts", {})
    active_ok = (
        reference_ok
        and active.get("status") == "PASS"
        and int(active.get("row_count", 0))
        == int(expected["active_modification_cases"])
        and int(active_counts.get("authenticated_cases", 0))
        == int(expected["authenticated_active_modifications"])
        and int(active_counts.get("authenticated_rejections", 0))
        == int(expected["authenticated_active_modifications"])
    )
    check(
        "V1-03",
        "The complete 180-case active-modification schedule matches its expected outcomes",
        active_ok,
        active,
    )

    if full:
        experiment = commands.get("full_experiment", {})
        experiment_ok = int(experiment.get("returncode", 1)) == 0
        check(
            "V2-01",
            "Two complete registered execution trees finish successfully",
            experiment_ok,
            experiment,
        )
        deterministic_command = commands.get("deterministic_reference", {})
        deterministic_summary = deterministic_command.get("summary", {})
        two_tree = deterministic_summary.get("two_tree_verification", {})
        primary = two_tree.get("primary", {})
        independent = two_tree.get("independent", {})
        cross = two_tree.get("cross_execution", {})
        deterministic_command_ok = int(deterministic_command.get("returncode", 1)) == 0
        check(
            "V2-02",
            "Primary execution matches all 37 frozen deterministic artifacts",
            experiment_ok and deterministic_command_ok and primary.get("status") == "PASS" and int(primary.get("matched_artifact_count", 0)) == 37,
            primary or deterministic_command,
        )
        check(
            "V2-03",
            "Independent execution matches all 37 frozen deterministic artifacts",
            experiment_ok and deterministic_command_ok and independent.get("status") == "PASS" and int(independent.get("matched_artifact_count", 0)) == 37,
            independent or deterministic_command,
        )
        check(
            "V2-04",
            "The two execution trees agree on all 37 deterministic artifacts",
            experiment_ok and deterministic_command_ok and cross.get("status") == "PASS" and int(cross.get("matched_count", 0)) == 37,
            cross or deterministic_command,
        )

        targeted = commands.get("targeted_validation", {})
        targeted_summary = bundle.get("targeted_validation", {})
        check(
            "V2-05",
            "Targeted validation passes in two execution trees",
            int(targeted.get("returncode", 1)) == 0
            and bool(targeted_summary.get("validation_passed")),
            {"command": targeted, "summary": targeted_summary},
        )
        primitive = commands.get("primitive_scaling", [])
        total_exact = sum(
            int(item.get("round_trip_cases_exact", 0)) for item in primitive
        )
        primitive_ok = (
            len(primitive) == 2
            and all(int(item.get("returncode", 1)) == 0 for item in primitive)
            and total_exact
            == int(expected["primitive_round_trips_across_two_executions"])
        )
        check(
            "V2-06",
            "Both primitive-scaling executions pass all 96 exact round trips",
            primitive_ok,
            primitive,
        )
        if timing:
            timing_record = bundle.get("timing_reconciliation", {})
            timing_ok = (
                bool(timing_record.get("passed"))
                and int(timing_record.get("configuration_count", 0))
                == int(expected["timing_configurations"])
            )
            check(
                "V3-01",
                "All 576 complete-path timing configurations satisfy the registered limits",
                timing_ok,
                timing_record,
                category="timing",
            )
            primitive_timing_ok = len(primitive) == 2 and all(
                int(item.get("timing_records", 0))
                == int(expected["primitive_timing_records_per_execution"])
                for item in primitive
            )
            check(
                "V3-02",
                "Primitive-scaling timing produces 1,440 records per execution",
                primitive_timing_ok,
                primitive,
                category="timing",
            )

    passed = sum(row.get("status") == "PASS" for row in checks)
    environment_failed = int(static_summary.get("exit_code", 0)) == 3
    if environment_failed:
        exit_code = 3
    elif timing_failed:
        exit_code = 5
    elif failures:
        exit_code = 4
    else:
        exit_code = 0
    mode = "timing" if timing else "full" if full else "default"
    summary = {
        "verification_id": config["verification_id"],
        "executed_tier": mode,
        "environment_policy": environment_policy,
        "status": "PASS" if not failures else "FAIL",
        "exit_code": exit_code,
        "check_count": len(checks),
        "passed_check_count": passed,
        "failures": failures,
        "checks": checks,
        "reference_layers": static_summary.get("reference_layers", {}),
        "output_paths": ["verification_summary.json", "verification_summary.txt"],
    }
    _write_summary(output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the complete public computational repository"
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-root", type=Path)
    tier = parser.add_mutually_exclusive_group()
    tier.add_argument("--static", action="store_true")
    tier.add_argument("--full", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument(
        "--environment-policy",
        choices=("exact", "packages", "report"),
        default=None,
    )
    args = parser.parse_args()
    if args.timing and not args.full:
        parser.error("--timing requires --full")

    root = args.repo_root.resolve()
    raw_config = json.loads(
        (root / "configs/verification/repository_verification.json").read_text(
            encoding="utf-8"
        )
    )
    mode = (
        "static"
        if args.static
        else "timing"
        if args.timing
        else "full"
        if args.full
        else "default"
    )
    output = (
        args.output_root.resolve()
        if args.output_root is not None
        else (root / raw_config["default_output_root"] / mode).resolve()
    )
    environment_policy = (
        args.environment_policy or raw_config["default_environment_policy"]
    )

    try:
        if args.static:
            _bootstrap(root)
            from qsa_benchmark.validation.repository import (  # pylint: disable=import-outside-toplevel
                run_repository_verification,
            )

            summary = run_repository_verification(
                root,
                output_root=output,
                static=True,
                environment_policy=environment_policy,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
        else:
            _prepare_output(output)
            print(
                "Executing isolated repository commands",
                file=sys.stderr,
                flush=True,
            )
            bundle = _run_bundle(
                root,
                output,
                full=bool(args.full),
                timing=bool(args.timing),
                environment_policy=environment_policy,
            )
            print(
                "Reconciling isolated verification results",
                file=sys.stderr,
                flush=True,
            )
            static_summary = bundle.get("static_summary", {})
            integrity_result = bundle.get("commands", {}).get("integrity_layers", {})
            if int(integrity_result.get("returncode", 1)) != 0 and not static_summary:
                static_summary = {
                    "status": "FAIL",
                    "exit_code": 4,
                    "checks": [],
                    "failures": ["static integrity worker did not produce a verification summary"],
                    "reference_layers": {},
                }
            summary = _merge_verification(
                root,
                output,
                raw_config,
                bundle,
                static_summary,
                full=bool(args.full),
                timing=bool(args.timing),
                environment_policy=environment_policy,
            )
    except Exception as exc:
        safe = f"{type(exc).__name__}: {exc}".replace(str(root), "<repository>").replace(
            str(output), "<verification-output>"
        )
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            "executed_tier": mode,
            "status": "FAIL",
            "exit_code": 4,
            "environment_policy": environment_policy,
            "failures": [safe],
            "output_paths": ["verification_summary.json", "verification_summary.txt"],
        }
        (output / "verification_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (output / "verification_summary.txt").write_text(
            "Repository verification failed.\n" + safe + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return int(summary.get("exit_code", 4))


if __name__ == "__main__":
    raise SystemExit(main())
