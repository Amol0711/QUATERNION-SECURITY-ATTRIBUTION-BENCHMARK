from pathlib import Path

from qsa_benchmark.validation.neutrality import prospective_archive_safety, scan_repository
from qsa_benchmark.validation.repository import load_verification_config


def test_documented_repository_paths_exist(repository_root):
    paths = [
        "configs/benchmark/core_full.yaml",
        "configs/verification/repository_verification.json",
        "reference/deterministic_outputs/artifact_hashes.json",
        "reference/verification/compact_fingerprint.json",
        "scripts/lock_environment.py",
        "scripts/run_primitive_scaling.py",
        "scripts/verify_reference_layers.py",
        "src/qsa_benchmark/validation/repository.py",
    ]
    assert all((repository_root / path).is_file() for path in paths)


def test_verification_config_counts(repository_root):
    config = load_verification_config(repository_root)
    assert config["expected_counts"]["registered_constructions"] == 24
    assert config["expected_counts"]["deterministic_artifacts"] == 37
    assert config["expected_counts"]["timing_configurations"] == 576


def test_repository_neutrality_and_safety(repository_root):
    assert scan_repository(repository_root)["status"] == "PASS"
    assert prospective_archive_safety(repository_root)["status"] == "PASS"
