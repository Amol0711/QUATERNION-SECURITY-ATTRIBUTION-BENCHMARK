from __future__ import annotations

import hashlib

import numpy as np
import pytest

from qsa_benchmark.benchmark.models import RunContext
from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES, make_method
from qsa_benchmark.benchmark.tamper import tamper_object
from qsa_benchmark.protocol.registry import method_policy


def context_for(method_id: str, ordinal: int) -> RunContext:
    policy = method_policy(method_id)
    nonce = hashlib.shake_256(method_id.encode("utf-8")).digest(policy.nonce_length)
    return RunContext(
        master_key=bytes(range(32)),
        nonce=nonce,
        seed=10_000 + ordinal,
        image_id="unit_probe",
        method_id=method_id,
        run_id=f"unit-{ordinal}",
        public_metadata={},
        public_material=b"",
        protocol_id="QSA-PROTOCOL-V1",
    )


@pytest.mark.parametrize("method_id", list(EXTENDED_METHOD_FACTORIES))
def test_exact_round_trip(method_id: str) -> None:
    ordinal = list(EXTENDED_METHOD_FACTORIES).index(method_id)
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    method = make_method(method_id, profile="extended")
    context = context_for(method_id, ordinal)
    protected = method.encrypt(image, context).object_bytes
    recovered = method.decrypt(protected, context)
    assert np.array_equal(image, recovered)


@pytest.mark.parametrize(
    "method_id",
    [
        "B01_aes_gcm",
        "B02_chacha20_poly1305",
        "B03_shake_hmac",
        "B15_geometry_shake_hmac",
        "B16_geometry_aes_gcm",
        "B17_tip_r0_emulation",
        "B20_full_aead_explicit_preview",
        "B23_secure_fixed_header",
        "B24_aes_gcm_siv",
    ],
)
def test_authenticated_methods_reject_modification(method_id: str) -> None:
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    method = make_method(method_id, profile="extended")
    context = context_for(method_id, 100)
    protected = method.encrypt(image, context).object_bytes
    with pytest.raises(Exception):
        method.decrypt(tamper_object(protected), context)
