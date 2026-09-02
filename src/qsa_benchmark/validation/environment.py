from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import ssl
import struct
import sys
import sysconfig
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from qsa_benchmark.benchmark.utils import canonical_json_bytes, sha256_file

ENVIRONMENT_PATH = "reference/environment/validated_environment.json"
LOCK_PATH = "requirements-lock.txt"


def _metadata_sha256(distribution: metadata.Distribution) -> str:
    text = distribution.read_text("METADATA") or ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_record(name: str, *, template: dict[str, Any] | None = None) -> dict[str, Any]:
    distribution = metadata.distribution(name)
    template = template or {}
    return {
        "active_requirements": list(template.get("active_requirements", [])),
        "dependencies": list(template.get("dependencies", [])),
        "metadata_sha256": _metadata_sha256(distribution),
        "name": name.lower().replace("_", "-"),
        "requires_python": distribution.metadata.get("Requires-Python", ""),
        "roles": list(template.get("roles", ["transitive"])),
        "version": distribution.version,
    }


def _numpy_runtime() -> dict[str, Any]:
    config = getattr(np.__config__, "CONFIG", {})
    blas = config.get("Build Dependencies", {}).get("blas", {})
    simd = config.get("SIMD Extensions", {})
    return {
        "blas_configuration": str(blas.get("openblas configuration", "")),
        "blas_name": str(blas.get("name", "")),
        "blas_version": str(blas.get("version", "")),
        "simd_baseline": list(simd.get("baseline", [])),
        "simd_found": list(simd.get("found", [])),
    }


def load_environment_manifest(repo_root: str | Path) -> dict[str, Any]:
    path = Path(repo_root).resolve() / ENVIRONMENT_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("environment manifest must contain a JSON object")
    return value


def collect_environment(repo_root: str | Path, template: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    template = template or load_environment_manifest(root)
    package_templates = {entry["name"]: entry for entry in template.get("packages", [])}
    package_names = [entry["name"] for entry in template.get("packages", [])]
    packages = [_package_record(name, template=package_templates.get(name)) for name in package_names]
    packages.sort(key=lambda item: item["name"])
    pip_dist = metadata.distribution("pip")
    libc_family, libc_version = platform.libc_ver()
    payload: dict[str, Any] = {
        "direct_dependency_groups": template.get("direct_dependency_groups", {}),
        "installer": {
            "metadata_sha256": _metadata_sha256(pip_dist),
            "name": "pip",
            "version": pip_dist.version,
        },
        "lock_id": template.get("lock_id", "QSA-REFERENCE-ENVIRONMENT-LOCK-V1"),
        "optional_native_accelerator": template.get("optional_native_accelerator", {}),
        "packages": packages,
        "project_requires_python": project["requires-python"],
        "project_version": project["version"],
        "requirements_lock": {
            "hash_policy": "exact versions plus installed METADATA SHA-256 values",
            "package_count": len(packages),
            "path": LOCK_PATH,
            "sha256": sha256_file(root / LOCK_PATH),
        },
        "runtime_libraries": {
            "numpy": _numpy_runtime(),
            "openssl": ssl.OPENSSL_VERSION,
        },
        "schema_version": int(template.get("schema_version", 1)),
        "validated_interpreter": {
            "byteorder": sys.byteorder,
            "cache_tag": sys.implementation.cache_tag,
            "implementation": platform.python_implementation(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "pointer_bits": struct.calcsize("P") * 8,
            "python_compiler": platform.python_compiler(),
            "version": platform.python_version(),
        },
        "validated_platform": {
            "libc_family": libc_family,
            "libc_version": libc_version,
            "machine": platform.machine(),
            "sysconfig_platform": sysconfig.get_platform(),
            "system": platform.system(),
        },
    }
    core = dict(payload)
    payload["environment_canonical_sha256"] = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    return payload


def compare_environment(repo_root: str | Path, policy: str = "exact") -> dict[str, Any]:
    if policy not in {"exact", "packages", "report"}:
        raise ValueError("environment policy must be exact, packages, or report")
    root = Path(repo_root).resolve()
    expected = load_environment_manifest(root)
    actual = collect_environment(root, expected)
    differences: list[str] = []

    expected_packages = {entry["name"]: entry for entry in expected["packages"]}
    actual_packages = {entry["name"]: entry for entry in actual["packages"]}
    if set(expected_packages) != set(actual_packages):
        differences.append("installed package inventory differs from the reference lock")
    for name in sorted(set(expected_packages) | set(actual_packages)):
        if name not in expected_packages or name not in actual_packages:
            continue
        if expected_packages[name]["version"] != actual_packages[name]["version"]:
            differences.append(f"package version differs: {name}")
        if policy == "exact" and expected_packages[name]["metadata_sha256"] != actual_packages[name]["metadata_sha256"]:
            differences.append(f"package metadata differs: {name}")

    if expected["requirements_lock"]["sha256"] != sha256_file(root / LOCK_PATH):
        differences.append("requirements-lock.txt digest differs")
    if expected["installer"]["version"] != actual["installer"]["version"]:
        differences.append("pip version differs")
    if policy == "exact":
        for section in ("validated_interpreter", "validated_platform", "runtime_libraries"):
            if expected[section] != actual[section]:
                differences.append(f"{section} differs")
        if expected["installer"]["metadata_sha256"] != actual["installer"]["metadata_sha256"]:
            differences.append("pip metadata differs")
        if expected.get("environment_canonical_sha256") != actual.get("environment_canonical_sha256"):
            differences.append("canonical environment identity differs")

    rejected = bool(differences) and policy != "report"
    return {
        "environment_policy": policy,
        "status": "FAIL" if rejected else "PASS",
        "difference_count": len(differences),
        "differences": differences,
        "reference_environment_sha256": expected.get("environment_canonical_sha256", ""),
        "active_environment_sha256": actual.get("environment_canonical_sha256", ""),
        "package_count": len(actual_packages),
    }


def write_environment_lock(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    template = load_environment_manifest(root)
    package_names = [entry["name"] for entry in template["packages"]]
    lock_lines = [
        "# Exact reference environment for deterministic repository verification.",
        f"# Validated interpreter: CPython {platform.python_version()}.",
        "# Ordinary installations may use the flexible constraints in requirements.txt.",
        "# Optional native-accelerator system requirements are recorded separately.",
        "",
    ]
    for name in sorted(package_names):
        lock_lines.append(f"{name}=={metadata.version(name)}")
    (root / LOCK_PATH).write_text("\n".join(lock_lines) + "\n", encoding="utf-8", newline="\n")
    payload = collect_environment(root, template)
    (root / ENVIRONMENT_PATH).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return payload
