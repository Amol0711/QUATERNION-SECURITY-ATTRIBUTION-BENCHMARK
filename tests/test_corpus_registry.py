from __future__ import annotations

import csv
import json
from pathlib import Path

from qsa_benchmark.protocol.config import load_protocol_config

ROOT = Path(__file__).resolve().parents[1]
_HEX = set("0123456789abcdef")


def _rows() -> list[dict[str, str]]:
    with (ROOT / "registries/corpus_registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _perturbation_rows() -> list[dict[str, str]]:
    with (ROOT / "registries/perturbation_schedule.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def test_corpus_registry_membership_order_and_shape() -> None:
    config = load_protocol_config()
    rows = _rows()
    image_order = list(config.payload["corpus"]["image_ids"])
    sizes = [96, 256, 512]

    assert len(rows) == 72
    assert [row["registry_row_id"] for row in rows] == [
        f"{image_id}@{size}x{size}" for image_id in image_order for size in sizes
    ]
    assert {int(row["image_size"]) for row in rows} == set(sizes)
    assert {int(row["width"]) for row in rows} == set(sizes)
    assert {int(row["height"]) for row in rows} == set(sizes)
    assert {int(row["channels"]) for row in rows} == {3}
    assert {row["dtype"] for row in rows} == {"uint8"}
    assert {row["color_space"] for row in rows} == {"nonlinear sRGB"}
    assert sum(row["standard_experiment_size"] == "true" for row in rows) == 48
    assert sum(row["timing_size"] == "true" for row in rows) == 72
    assert sum(row["corpus"] == "natural" for row in rows) == 36
    assert sum(row["corpus"] == "synthetic" for row in rows) == 36


def test_corpus_panel_flags_match_protocol_at_every_size() -> None:
    config = load_protocol_config()
    panels = {
        name: set(config.payload["corpus"][name])
        for name in ("primary_panel", "ensemble_panel", "secondary_panel", "timing_panel")
    }
    rows = _rows()
    for row in rows:
        image_id = row["image_id"]
        for panel_name, members in panels.items():
            assert (row[panel_name] == "true") == (image_id in members)

    for size in (96, 256, 512):
        at_size = [row for row in rows if int(row["image_size"]) == size]
        assert sum(row["primary_panel"] == "true" for row in at_size) == 12
        assert sum(row["ensemble_panel"] == "true" for row in at_size) == 4
        assert sum(row["secondary_panel"] == "true" for row in at_size) == 12
        assert sum(row["timing_panel"] == "true" for row in at_size) == 4


def test_corpus_registry_paths_hashes_and_provenance_are_public_safe() -> None:
    for row in _rows():
        generated_path = Path(row["generated_relative_path"])
        assert not generated_path.is_absolute()
        assert ".." not in generated_path.parts
        assert generated_path.as_posix().startswith("data/generated/")
        assert row["source"]
        assert row["source_provider"] in {"qsa_benchmark", "scikit-image"}
        assert row["preprocessing"]
        assert row["license_note"]
        for field in ("pixel_sha256", "file_sha256"):
            assert len(row[field]) == 64
            assert set(row[field]) <= _HEX
        assert int(row["file_size_bytes"]) > 0


def test_corpus_hashes_regenerate_exactly(
    regenerated_registry_files: dict[str, bytes],
) -> None:
    assert regenerated_registry_files["registries/corpus_registry.csv"] == (
        ROOT / "registries/corpus_registry.csv"
    ).read_bytes()


def test_perturbation_probe_validation_matches_declared_counts() -> None:
    rows = _perturbation_rows()
    assert len(rows) == 24
    assert sum(row["panel"] == "primary" for row in rows) == 12
    assert sum(row["panel"] == "secondary" for row in rows) == 12

    for row in rows:
        expected_pixels = json.loads(row["expected_pixels_by_size"])
        expected_coordinates = json.loads(row["expected_coordinates_by_size"])
        validated_pixels = json.loads(row["validation_pixel_counts"])
        validated_coordinates = json.loads(row["validation_coordinate_counts"])
        channel_indices = json.loads(row["channel_indices"])
        applicable_tiers = json.loads(row["applicable_tiers"])
        tier_schedule = json.loads(row["tier_schedule"])

        assert set(expected_pixels) == {"96", "256", "512"}
        assert validated_pixels == expected_pixels
        assert validated_coordinates == expected_coordinates
        assert all(
            expected_coordinates[size] == expected_pixels[size] * len(channel_indices)
            for size in expected_pixels
        )
        assert applicable_tiers
        assert applicable_tiers == list(tier_schedule)
        assert row["validation_probe_id"] == "syn_lowrank"
