from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from qsa_benchmark.benchmark.components import permutation_indices
from qsa_benchmark.benchmark.envelope import encode_envelope, parse_envelope
from qsa_benchmark.benchmark.models import EnvelopeParts, RunContext, TransformOutput
from qsa_benchmark.benchmark.quaternion import fixed_axis_rotation, random_so3
from qsa_benchmark.benchmark.registry import make_method
from qsa_benchmark.benchmark.serialization import bytes_to_image
from qsa_benchmark.benchmark.transforms import transform_from_descriptor

from .metrics import reconstruction_metrics

NONCE16 = {"B03_shake_hmac", "B15_geometry_shake_hmac"}


def key_variant(master: bytes, index: int) -> bytes:
    h = hashlib.shake_256(b"QSA-ATTACK-KEY-V1" + index.to_bytes(4, "big") + master)
    return h.digest(32)


def deterministic_nonce(method_id: str, benchmark: str, image_id: str, repetition: int, seed: int) -> bytes:
    length = 16 if method_id in NONCE16 else 12
    h = hashlib.shake_256(b"QSA-ATTACK-NONCE-V1")
    for value in (method_id, benchmark, image_id, str(repetition), str(seed)):
        blob = value.encode(); h.update(len(blob).to_bytes(4, "big")); h.update(blob)
    return h.digest(length)


def make_context(benchmark: str, master: bytes, seed: int, method_id: str, image_id: str, repetition: int, *, key_index: int = 0, forced_nonce: bytes | None = None, suffix: str = "") -> RunContext:
    local_seed = int.from_bytes(hashlib.sha256(f"{benchmark}|{method_id}|{image_id}|{repetition}|{seed}|{suffix}".encode()).digest()[:8], "big")
    nonce = forced_nonce if forced_nonce is not None else deterministic_nonce(method_id, benchmark, image_id, repetition, local_seed)
    return RunContext(key_variant(master, key_index), nonce, local_seed, image_id, method_id, f"{benchmark}|{method_id}|{image_id}|{repetition}|k{key_index}|{suffix}")


def public_exact_recovery(method_id: str, object_bytes: bytes) -> np.ndarray:
    parsed = parse_envelope(object_bytes)
    header = parsed.header
    context = RunContext(bytes(32), parsed.nonce, 0, str(header["metadata"]["image_id"]), method_id, "public-attacker")
    return make_method(method_id).decrypt(object_bytes, context)


def recover_permutation(shape: tuple[int, int, int], context: RunContext) -> tuple[np.ndarray, float, int]:
    n = int(np.prod(shape))
    if n > 65536:
        raise ValueError("two-query recovery supports at most 65536 bytes")
    method = make_method("B13_permutation_only")
    low = (np.arange(n, dtype=np.uint32) & 0xFF).astype(np.uint8).reshape(shape)
    high = ((np.arange(n, dtype=np.uint32) >> 8) & 0xFF).astype(np.uint8).reshape(shape)
    c0 = parse_envelope(method.encrypt(low, context).object_bytes).protected_payload
    c1 = parse_envelope(method.encrypt(high, context).object_bytes).protected_payload
    mapping = np.frombuffer(c0, dtype=np.uint8).astype(np.int64) + (np.frombuffer(c1, dtype=np.uint8).astype(np.int64) << 8)
    true = permutation_indices(n, context, "B13_permutation_only")
    return mapping, float(np.mean(mapping == true)), 2


def apply_permutation_attack(ciphertext: bytes, mapping: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    c = np.frombuffer(ciphertext, dtype=np.uint8)
    out = np.empty_like(c); out[mapping] = c
    return out.reshape(shape).copy()


def _inverse_diffusion_difference(delta: np.ndarray) -> np.ndarray:
    db = delta.astype(np.int64) % 256
    df = np.empty_like(db)
    if len(db):
        df[-1] = db[-1]
        if len(db) > 1:
            df[:-1] = (db[:-1] - db[1:]) % 256
    dp = np.empty_like(df)
    if len(df):
        dp[0] = df[0]
        if len(df) > 1:
            dp[1:] = (df[1:] - df[:-1]) % 256
    return dp.astype(np.uint8)


def affine_diffusion_recovery(method_id: str, target: np.ndarray, context: RunContext) -> np.ndarray:
    method = make_method(method_id)
    zero = np.zeros_like(target)
    c0 = np.frombuffer(parse_envelope(method.encrypt(zero, context).object_bytes).protected_payload, dtype=np.uint8)
    ct = np.frombuffer(parse_envelope(method.encrypt(target, context).object_bytes).protected_payload, dtype=np.uint8)
    recovered = _inverse_diffusion_difference((ct.astype(np.int64) - c0.astype(np.int64)) % 256)
    if method_id == "B19_external_chaos_pd":
        h, w, _ = target.shape
        y, x = np.mgrid[0:h, 0:w]; xx, yy = x.copy(), y.copy()
        for _ in range(3): xx, yy = (xx + yy) % h, (xx + 2 * yy) % h
        indices = (yy * h + xx).reshape(-1)
        values = recovered.reshape(h * w, 3)
        inverse = np.empty_like(indices); inverse[indices] = np.arange(len(indices))
        recovered = values[inverse].reshape(target.shape).reshape(-1)
    return recovered.reshape(target.shape)


def recover_rotation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 2:
        raise ValueError("need at least two 3-D correspondences")
    if len(x) == 2:
        x = np.vstack([x, np.cross(x[0], x[1])])
        y = np.vstack([y, np.cross(y[0], y[1])])
    if np.linalg.matrix_rank(x) < 3 or np.linalg.matrix_rank(y) < 3:
        raise ValueError("correspondences are rank deficient")
    u, _, vt = np.linalg.svd(y.T @ x)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1; r = u @ vt
    return r


def rotation_recovery_study(repetitions: int, seed: int, counts: tuple[int, ...], noises: tuple[float, ...], quantization: tuple[float, ...]) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(seed)
    for method_id in ("B08_random_so3", "B09_fixed_axis_quaternion"):
        for rep in range(repetitions):
            r = random_so3(int(rng.integers(0, 2**63 - 1))) if method_id.startswith("B08") else fixed_axis_rotation()
            for count in counts:
                base = rng.normal(size=(max(count, 3), 3))
                for sigma in noises:
                    for q in quantization:
                        x = base[:count].copy(); y = x @ r.T
                        if sigma: y += rng.normal(scale=sigma, size=y.shape)
                        if q: y = np.rint(y / q) * q
                        succeeded = True
                        try: estimate = recover_rotation(x, y)
                        except ValueError:
                            succeeded = False; estimate = np.eye(3)
                        fro = float(np.linalg.norm(estimate - r)) if succeeded else float(2 * np.sqrt(2))
                        relative = estimate @ r.T
                        geo = float(Rotation.from_matrix(relative).magnitude()) if succeeded else float(np.pi)
                        rows.append({"method_id": method_id, "repetition": rep, "correspondences": count, "noise_sigma": sigma, "quantization_step": q, "recovery_succeeded": succeeded, "frobenius_error": fro, "geodesic_error_rad": geo})
    return rows


def nonce_reuse_attack(method_id: str, first: np.ndarray, second: np.ndarray, first_context: RunContext, second_context: RunContext) -> dict[str, Any]:
    method = make_method(method_id)
    a = parse_envelope(method.encrypt(first, first_context).object_bytes)
    b = parse_envelope(method.encrypt(second, second_context).object_bytes)
    c1 = np.frombuffer(a.protected_payload, dtype=np.uint8)
    c2 = np.frombuffer(b.protected_payload, dtype=np.uint8)
    if len(c1) != len(c2):
        return {"relation_accuracy": 0.0, "second_message_exact": False}
    if method_id in {"B15_geometry_shake_hmac", "B16_geometry_aes_gcm"}:
        descriptor = a.header["descriptor"]["transform"]
        transformed = transform_from_descriptor(descriptor).forward(first, first_context).payload
        known = np.frombuffer(transformed, dtype=np.uint8)
        recovered_payload = bytes(c1 ^ c2 ^ known)
        out_desc = b.header["descriptor"]["transform"]
        recovered = transform_from_descriptor(out_desc).inverse(TransformOutput(recovered_payload, out_desc, tuple(b.header["image_shape"])), second_context)
        expected_payload = transform_from_descriptor(out_desc).forward(second, second_context).payload
        relation = float(np.mean((c1 ^ c2) == (known ^ np.frombuffer(expected_payload, dtype=np.uint8))))
    else:
        known = np.frombuffer(first.tobytes(), dtype=np.uint8)
        recovered_payload = bytes(c1 ^ c2 ^ known)
        recovered = bytes_to_image(recovered_payload, tuple(second.shape))
        relation = float(np.mean((c1 ^ c2) == (known ^ np.frombuffer(second.tobytes(), dtype=np.uint8))))
    return {"relation_accuracy": relation, "second_message_exact": bool(np.array_equal(recovered, second)), **reconstruction_metrics(second, recovered)}


def _parts(parsed, *, header=None, nonce=None, public=None, protected=None, tag=None) -> bytes:
    h = parsed.header if header is None else header
    return encode_envelope(EnvelopeParts(h["method_id"], tuple(h["image_shape"]), h["descriptor"], h["metadata"], parsed.nonce if nonce is None else nonce, parsed.public_payload if public is None else public, parsed.protected_payload if protected is None else protected, parsed.tag if tag is None else tag))


def run_active_attacks(method_id: str, first: np.ndarray, second: np.ndarray, ctx1: RunContext, ctx2: RunContext) -> list[dict[str, Any]]:
    method = make_method(method_id)
    obj1 = method.encrypt(first, ctx1).object_bytes; obj2 = method.encrypt(second, ctx2).object_bytes
    p1 = parse_envelope(obj1); p2 = parse_envelope(obj2)
    mutations: dict[str, bytes] = {}
    def flip(blob: bytes) -> bytes:
        if not blob: return blob
        out = bytearray(blob); out[len(out)//2] ^= 1; return bytes(out)
    if p1.public_payload: mutations["public-bit-flip"] = _parts(p1, public=flip(p1.public_payload))
    if p1.protected_payload: mutations["protected-bit-flip"] = _parts(p1, protected=flip(p1.protected_payload))
    if p1.protected_payload and len(p1.protected_payload) == len(p2.protected_payload):
        half = len(p1.protected_payload)//2; mutations["protected-splice"] = _parts(p1, protected=p1.protected_payload[:half]+p2.protected_payload[half:])
    if p1.public_payload and len(p1.public_payload) == len(p2.public_payload):
        half = len(p1.public_payload)//2; mutations["public-splice"] = _parts(p1, public=p1.public_payload[:half]+p2.public_payload[half:])
    mutations["tag-removal"] = _parts(p1, tag=b"")
    mutations["nonce-modification"] = _parts(p1, nonce=flip(p1.nonce))
    header = json.loads(json.dumps(p1.header)); header["metadata"]["image_id"] = "substituted"
    mutations["metadata-substitution"] = _parts(p1, header=header)
    mutations["truncation"] = obj1[:-1]
    mutations["trailing-byte"] = obj1 + b"\x00"
    mutations["replay"] = obj1
    rows = []
    for name, blob in mutations.items():
        accepted = changed = False; error = ""
        try:
            recovered = method.decrypt(blob, ctx1)
            accepted = True; changed = not np.array_equal(recovered, first)
        except Exception as exc:
            error = type(exc).__name__
        rows.append({"method_id": method_id, "mutation": name, "authenticated": method.authenticated, "accepted": accepted, "accepted_plaintext_changed": changed, "error_type": error, "replay_excluded_from_forgery": name == "replay"})
    return rows
