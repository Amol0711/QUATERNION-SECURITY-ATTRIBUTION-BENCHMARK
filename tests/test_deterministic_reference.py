from qsa_benchmark.validation.deterministic import (
    deterministic_reference_summary,
    load_artifact_manifest,
    verify_static_artifact_contract,
)


def test_deterministic_reference_contract(repository_root):
    assert verify_static_artifact_contract(repository_root) == []
    summary = deterministic_reference_summary(repository_root)
    assert summary["artifact_count"] == 37
    assert summary["table_count"] == 13
    assert summary["representative_object_count"] == 24


def test_deterministic_reference_paths_are_unique(repository_root):
    manifest = load_artifact_manifest(repository_root)
    paths = [entry["path"] for entry in manifest["artifacts"]]
    assert len(paths) == len(set(paths)) == 37
