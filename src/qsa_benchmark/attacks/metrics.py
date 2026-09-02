from __future__ import annotations

import math
import zlib

import numpy as np
from scipy.spatial.distance import jensenshannon
from skimage.color import deltaE_ciede2000, rgb2lab, rgb2ycbcr
from skimage.metrics import structural_similarity


def byte_entropy(payload: bytes) -> float:
    if not payload:
        return 0.0
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256).astype(float)
    p = counts[counts > 0] / len(payload)
    return float(-np.sum(p * np.log2(p)))


def chi_square_per_df(payload: bytes) -> float:
    if not payload:
        return 0.0
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256).astype(float)
    expected = len(payload) / 256.0
    return float(np.sum((counts - expected) ** 2 / max(expected, 1e-12)) / 255.0)


def js_uniform(payload: bytes) -> float:
    if not payload:
        return 0.0
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256).astype(float)
    p = counts / counts.sum()
    q = np.full(256, 1.0 / 256.0)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def adjacent_byte_correlation(payload: bytes) -> float:
    arr = np.frombuffer(payload, dtype=np.uint8)
    return _corr(arr[:-1], arr[1:]) if len(arr) > 2 else 0.0


def fold_bytes_to_rgb(payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    count = int(np.prod(shape))
    if not payload:
        return np.zeros(shape, dtype=np.uint8)
    arr = np.frombuffer(payload, dtype=np.uint8)
    if len(arr) < count:
        arr = np.resize(arr, count)
    elif len(arr) > count:
        blocks = int(math.ceil(len(arr) / count))
        padded = np.pad(arr, (0, blocks * count - len(arr)), mode="wrap")
        arr = np.rint(padded.reshape(blocks, count).mean(axis=0)).astype(np.uint8)
    return arr[:count].reshape(shape).copy()


def spatial_correlations(image: np.ndarray) -> dict[str, float]:
    values = np.asarray(image, dtype=float)
    return {
        "horizontal_correlation": _corr(values[:, :-1], values[:, 1:]),
        "vertical_correlation": _corr(values[:-1, :], values[1:, :]),
        "diagonal_correlation": _corr(values[:-1, :-1], values[1:, 1:]),
    }


def zlib_ratio(payload: bytes) -> float:
    return float(len(zlib.compress(payload, level=9)) / max(1, len(payload)))


def bit_balance(payload: bytes) -> float:
    if not payload:
        return 0.0
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    return float(abs(bits.mean() - 0.5))


def psl_byte_triplets(payload: bytes) -> float:
    usable = (len(payload) // 3) * 3
    if usable < 9:
        return 1.0
    x = np.frombuffer(payload[:usable], dtype=np.uint8).reshape(-1, 3).astype(float)
    cov = np.cov(x, rowvar=False)
    trace = float(np.trace(cov))
    if trace <= 1e-12:
        return 1.0
    return float(np.linalg.eigvalsh(cov).max() / trace)


def npcr_uaci(left: bytes, right: bytes) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("NPCR/UACI require equal nonempty byte strings")
    a = np.frombuffer(left, dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(right, dtype=np.uint8).astype(np.int16)
    return float(np.mean(a != b)), float(np.mean(np.abs(a - b)) / 255.0)


def mutual_information_aligned(plain: bytes, cipher: bytes, bins: int = 16) -> float:
    n = min(len(plain), len(cipher))
    if n < 2:
        return 0.0
    a = np.frombuffer(plain[:n], dtype=np.uint8) // (256 // bins)
    b = np.frombuffer(cipher[:n], dtype=np.uint8) // (256 // bins)
    joint = np.zeros((bins, bins), dtype=float)
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    expected = pa @ pb
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / expected[mask])))


def reconstruction_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float | bool]:
    ref = np.asarray(reference, dtype=np.uint8)
    rec = np.clip(np.rint(reconstruction), 0, 255).astype(np.uint8)
    exact = bool(np.array_equal(ref, rec))
    mse = float(np.mean((ref.astype(float) - rec.astype(float)) ** 2))
    psnr = math.inf if mse == 0 else float(10 * np.log10(255.0**2 / mse))
    ssim = float(structural_similarity(ref, rec, channel_axis=2, data_range=255))
    lab_ref = rgb2lab(ref / 255.0)
    lab_rec = rgb2lab(rec / 255.0)
    de = float(np.mean(deltaE_ciede2000(lab_ref, lab_rec)))
    y_ref = rgb2ycbcr(ref / 255.0)
    y_rec = rgb2ycbcr(rec / 255.0)
    ymse = float(np.mean((y_ref - y_rec) ** 2))
    ypsnr = math.inf if ymse == 0 else float(10 * np.log10(255.0**2 / ymse))
    return {"exact": exact, "psnr_db": psnr, "ssim": ssim, "mean_delta_e": de, "ycbcr_psnr_db": ypsnr}
