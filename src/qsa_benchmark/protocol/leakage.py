from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from qsa_benchmark.benchmark.constructions import FullAEADExplicitPreview

from .registry import method_policy


def log2_multinomial(counts: Iterable[int]) -> float:
    values = [int(value) for value in counts if int(value) > 0]
    if not values:
        return 0.0
    total = sum(values)
    return (math.lgamma(total + 1) - sum(math.lgamma(value + 1) for value in values)) / math.log(2.0)


def _pixel_codes(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    return (values[:, 0] << 16) | (values[:, 1] << 8) | values[:, 2]


def pca_pixel_multiset_orbit_bits(image: np.ndarray) -> float:
    _, counts = np.unique(_pixel_codes(image), return_counts=True)
    return log2_multinomial(int(value) for value in counts)


def curvature_residue_orbit_bits(image: np.ndarray, states: int = 8) -> float:
    if states <= 0:
        raise ValueError("state count must be positive")
    codes = _pixel_codes(image)
    total = 0.0
    for state in range(states):
        _, counts = np.unique(codes[state::states], return_counts=True)
        total += log2_multinomial(int(value) for value in counts)
    return total


def preview_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(FullAEADExplicitPreview._preview(np.asarray(image, dtype=np.uint8))).hexdigest()


def load_manifest_images(manifest_rows: Iterable[Mapping[str, Any]], repo_root: Path) -> dict[tuple[str, int], np.ndarray]:
    images: dict[tuple[str, int], np.ndarray] = {}
    for row in manifest_rows:
        path = repo_root / str(row["path"])
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        size = int(row["height"])
        images[(str(row["image_id"]), size)] = image
    return images


def leakage_entropy_rows(
    images: Mapping[tuple[str, int], np.ndarray],
    corpus_by_image_size: Mapping[tuple[str, int], str],
    methods: Iterable[str],
) -> list[dict[str, Any]]:
    preview_groups: dict[tuple[int, str], list[str]] = {}
    for (image_id, size), image in images.items():
        preview_groups.setdefault((size, preview_sha256(image)), []).append(image_id)

    rows: list[dict[str, Any]] = []
    for (image_id, size), image in sorted(images.items(), key=lambda item: (item[0][1], item[0][0])):
        height, width, channels = image.shape
        if height != size or width != size or channels != 3:
            raise ValueError("standard corpus image shape mismatch")
        message_bits = float(24 * height * width)
        pca_bits = pca_pixel_multiset_orbit_bits(image)
        curvature_bits = curvature_residue_orbit_bits(image, 8)
        p_hash = preview_sha256(image)
        preview_class_size = len(preview_groups[(size, p_hash)])
        preview_empirical_bits = math.log2(preview_class_size)

        for method_id in methods:
            policy = method_policy(method_id)
            if policy.prechallenge_entropy_rule == "shape_only_message_space":
                pre_bits: float | str = message_bits
                pre_status = "exact_for_declared_shape_length_leakage"
            elif policy.prechallenge_entropy_rule == "singleton_complete_leakage_class":
                pre_bits = 0.0
                pre_status = "exact_due_to_complete_public_invertible_payload"
            elif policy.prechallenge_entropy_rule == "curvature_residue_orbit":
                pre_bits = curvature_bits
                pre_status = "constructive_lower_bound_from_within_residue_permutations"
            elif policy.prechallenge_entropy_rule == "preview_preimage_not_formally_lower_bounded":
                pre_bits = preview_empirical_bits
                pre_status = "empirical_frozen_corpus_collision_class_only_not_formal"
            else:
                raise ValueError(f"unknown prechallenge entropy rule: {policy.prechallenge_entropy_rule}")

            if policy.descriptor_entropy_rule == "pca_pixel_multiset_orbit":
                descriptor_bits: float | str = pca_bits
                descriptor_status = "constructive_PCA_descriptor_orbit"
            elif policy.descriptor_entropy_rule == "curvature_residue_orbit":
                descriptor_bits = curvature_bits
                descriptor_status = "constructive_curvature_descriptor_orbit"
            elif policy.descriptor_entropy_rule == "preview_collision_class_empirical_only":
                descriptor_bits = preview_empirical_bits
                descriptor_status = "empirical_preview_collision_class_only"
            elif policy.descriptor_entropy_rule == "shape_only_message_space":
                descriptor_bits = message_bits
                descriptor_status = "exact_for_constant_or_shape_only_descriptor"
            else:
                raise ValueError(f"unknown descriptor entropy rule: {policy.descriptor_entropy_rule}")

            post_bits: float | str = 0.0 if policy.post_object_recovery == "zero_via_public_inverse" else "NA"
            rows.append({
                "method_id": method_id,
                "image_id": image_id,
                "corpus": corpus_by_image_size[(image_id, size)],
                "image_size": size,
                "height": height,
                "width": width,
                "message_space_bits": round(message_bits, 6),
                "prechallenge_L_class_bits": round(pre_bits, 6) if isinstance(pre_bits, float) else pre_bits,
                "prechallenge_status": pre_status,
                "descriptor_only_orbit_bits": round(descriptor_bits, 6) if isinstance(descriptor_bits, float) else descriptor_bits,
                "descriptor_status": descriptor_status,
                "post_complete_object_public_recovery_bits": post_bits,
                "publicly_invertible": policy.publicly_invertible,
                "preview_sha256": p_hash if method_id == "B20_full_aead_explicit_preview" else "",
                "preview_collision_class_size": preview_class_size if method_id == "B20_full_aead_explicit_preview" else "",
                "formal_scope": "prechallenge leakage class and descriptor orbit are distinct from post-object recoverability",
            })
    return rows
