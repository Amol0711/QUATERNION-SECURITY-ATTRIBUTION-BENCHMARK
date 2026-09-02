from __future__ import annotations

import numpy as np

from qsa_benchmark.protocol.targeted_validation import (
    minimum_queries,
    observed_class_counts,
    theoretical_class_counts,
)


def test_minimum_base_256_queries() -> None:
    assert minimum_queries(96 * 96 * 3, 256) == 2
    assert minimum_queries(256 * 256 * 3, 256) == 3
    assert minimum_queries(512 * 512 * 3, 256) == 3


def test_class_count_formula_matches_enumeration() -> None:
    n, capacity = 1000, 256
    codes = np.arange(n, dtype=np.uint64) % capacity
    assert observed_class_counts(codes) == theoretical_class_counts(n, capacity)
