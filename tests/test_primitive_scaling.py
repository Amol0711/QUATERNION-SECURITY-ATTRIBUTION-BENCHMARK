from qsa_benchmark.protocol.primitive_scaling import execute_primitive_scaling, load_primitive_config


def test_primitive_scaling_config_factorization(repository_root):
    config = load_primitive_config(repository_root)
    assert config["expected_counts"]["round_trip_cases_per_execution"] == 48
    assert config["expected_counts"]["round_trip_cases_across_two_executions"] == 96
    assert config["expected_counts"]["timing_records_per_execution"] == 1440


def test_primitive_scaling_all_round_trips(repository_root, tmp_path):
    result = execute_primitive_scaling(repository_root, tmp_path / "primitive", run_label="test_execution", timing=False)
    assert result["round_trip_cases"] == 48
    assert result["round_trip_cases_exact"] == 48
    assert result["timing_records"] == 0
