from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from skimage import data, transform

from .models import DatasetRecord
from .utils import sha256_file


def _rgb(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    if values.ndim != 3:
        raise ValueError("unsupported source image rank")
    if values.shape[2] >= 4:
        values = values[..., :3]
    if values.dtype == bool:
        values = values.astype(np.uint8) * 255
    elif np.issubdtype(values.dtype, np.floating):
        maximum = float(np.nanmax(values)) if values.size else 1.0
        if maximum <= 1.0:
            values = values * 255.0
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _resize(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rgb = _rgb(array)
    resized = transform.resize(
        rgb,
        size + (3,),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
        mode="reflect",
    )
    return np.clip(np.rint(resized), 0, 255).astype(np.uint8)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path, format="PNG", optimize=False, compress_level=9)


def _synthetic(size: tuple[int, int]) -> list[tuple[str, np.ndarray, str, str]]:
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    zeros = np.zeros((h, w, 3), dtype=np.uint8)
    white = np.full_like(zeros, 255)
    gray = np.full_like(zeros, 127)
    red = zeros.copy(); red[..., 0] = 220
    horizontal = np.repeat(np.rint(np.linspace(0, 255, w))[None, :, None], h, axis=0)
    horizontal = np.repeat(horizontal, 3, axis=2).astype(np.uint8)
    vertical = np.repeat(np.rint(np.linspace(0, 255, h))[:, None, None], w, axis=1)
    vertical = np.repeat(vertical, 3, axis=2).astype(np.uint8)
    impulse = zeros.copy(); impulse[h // 2, w // 2] = [255, 127, 63]
    checker = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)
    checker = np.stack([checker, np.roll(checker, 4, axis=1), np.roll(checker, 4, axis=0)], axis=-1)
    basis = zeros.copy()
    basis[:, :w // 3, 0] = 255; basis[:, w // 3:2 * w // 3, 1] = 255; basis[:, 2 * w // 3:, 2] = 255
    lowrank = np.stack([
        (3 * x + 2 * y) % 256,
        (6 * x + 4 * y) % 256,
        (9 * x + 6 * y) % 256,
    ], axis=-1).astype(np.uint8)
    bars = np.zeros_like(zeros)
    colors = np.array([[255,255,255],[255,255,0],[0,255,255],[0,255,0],[255,0,255],[255,0,0],[0,0,255],[0,0,0]], dtype=np.uint8)
    for index, color in enumerate(colors):
        bars[:, index * w // 8:(index + 1) * w // 8] = color
    periodic = np.stack([
        127.5 + 127.5 * np.sin(2 * np.pi * x / 9),
        127.5 + 127.5 * np.sin(2 * np.pi * y / 11),
        127.5 + 127.5 * np.sin(2 * np.pi * (x + y) / 13),
    ], axis=-1)
    periodic = np.clip(np.rint(periodic), 0, 255).astype(np.uint8)
    items = [
        ("syn_black", zeros, "development", "constant-black"),
        ("syn_white", white, "development", "constant-white"),
        ("syn_gray", gray, "development", "constant-gray"),
        ("syn_red", red, "development", "single-channel-constant"),
        ("syn_ramp_h", horizontal, "development", "horizontal-ramp"),
        ("syn_ramp_v", vertical, "development", "vertical-ramp"),
        ("syn_impulse", impulse, "validation", "impulse"),
        ("syn_checker", checker, "validation", "repeated-blocks"),
        ("syn_basis", basis, "test", "rgb-bases"),
        ("syn_lowrank", lowrank, "test", "low-rank-color"),
        ("syn_bars", bars, "test", "color-bars"),
        ("syn_periodic", periodic, "test", "periodic-pattern"),
    ]
    return [(name, image, split, label) for name, image, split, label in items]


_NATURAL: list[tuple[str, Callable[[], np.ndarray], str, str, str]] = [
    ("astronaut", data.astronaut, "development", "person", "public domain sample distributed by scikit-image"),
    ("coffee", data.coffee, "development", "food-object", "CC0 sample distributed by scikit-image"),
    ("chelsea", data.chelsea, "development", "animal", "CC0 sample distributed by scikit-image"),
    ("rocket", data.rocket, "development", "vehicle", "public-domain SpaceX/NASA-origin sample distributed by scikit-image"),
    ("brick", data.brick, "development", "texture", "CC0 sample distributed by scikit-image"),
    ("grass", data.grass, "development", "texture", "CC0 sample distributed by scikit-image"),
    ("hubble", data.hubble_deep_field, "validation", "astronomy", "public-domain Hubble sample distributed by scikit-image"),
    ("ihc", data.immunohistochemistry, "validation", "histology", "source sample distributed by scikit-image; upstream terms retained"),
    ("retina", data.retina, "test", "retina", "CC0 sample distributed by scikit-image"),
    ("logo", data.logo, "test", "graphic", "scikit-image project logo; project terms retained"),
    ("colorwheel", data.colorwheel, "test", "color-graphic", "sample distributed by scikit-image; upstream terms retained"),
    ("horse", data.horse, "test", "silhouette", "sample distributed by scikit-image; upstream terms retained"),
]


def build_corpora(root: Path, size: tuple[int, int] = (96, 96)) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    synthetic_root = root / "synthetic"
    natural_root = root / "natural"
    for image_id, image, split, label in _synthetic(size):
        path = synthetic_root / f"{image_id}_{size[0]}x{size[1]}.png"
        _save_png(path, image)
        records.append(DatasetRecord(
            image_id, "synthetic", split, "deterministic repository generator",
            str(path), size[1], size[0], label, "CC0-1.0 generated benchmark probe", sha256_file(path),
        ))
    for image_id, function, split, label, license_note in _NATURAL:
        image = _resize(function(), size)
        path = natural_root / f"{image_id}_{size[0]}x{size[1]}.png"
        _save_png(path, image)
        records.append(DatasetRecord(
            image_id, "natural", split, f"skimage.data.{function.__name__}",
            str(path), size[1], size[0], label, license_note, sha256_file(path),
        ))
    return records


def write_manifests(records: list[DatasetRecord], manifest_dir: Path, size: tuple[int, int]) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    # Convert to repository-relative paths.
    repo_root = manifest_dir.parents[2]
    fields = ["image_id", "corpus", "split", "source", "path", "width", "height", "semantic_label", "license", "sha256"]
    with (manifest_dir / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for record in records:
            row = record.__dict__.copy(); row["path"] = str(Path(record.path).resolve().relative_to(repo_root.resolve()))
            writer.writerow(row)
    with (manifest_dir / "semantic_subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split", "semantic_label", "path"]); writer.writeheader()
        for record in records:
            if record.corpus == "natural":
                writer.writerow({
                    "image_id": record.image_id,
                    "split": record.split,
                    "semantic_label": record.semantic_label,
                    "path": str(Path(record.path).resolve().relative_to(repo_root.resolve())),
                })
    provenance = {
        "profile": "QSA-DATASET-V1",
        "image_size": list(size),
        "color_space": "nonlinear sRGB",
        "dtype": "uint8",
        "resizing": "bilinear, antialiasing, preserve range, round-to-nearest, clip [0,255]",
        "synthetic_generator": "qsa_benchmark.benchmark.datasets",
        "natural_provider": "scikit-image data module",
        "record_count": len(records),
        "manifest_sha256": sha256_file(manifest_dir / "dataset_manifest.csv"),
    }
    (manifest_dir / "data_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> list[DatasetRecord]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [DatasetRecord(
            row["image_id"], row["corpus"], row["split"], row["source"], row["path"],
            int(row["width"]), int(row["height"]), row["semantic_label"], row["license"], row["sha256"],
        ) for row in csv.DictReader(handle)]
