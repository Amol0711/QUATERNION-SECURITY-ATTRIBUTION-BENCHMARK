from qsa_benchmark.validation.environment import compare_environment, load_environment_manifest


def test_exact_environment_matches_reference(repository_root):
    result = compare_environment(repository_root, "exact")
    assert result["status"] == "PASS", result
    assert result["difference_count"] == 0


def test_environment_manifest_has_32_packages(repository_root):
    manifest = load_environment_manifest(repository_root)
    assert manifest["requirements_lock"]["package_count"] == 32
    assert len(manifest["packages"]) == 32
