from __future__ import annotations

from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES, make_method, method_registry

from .models import MethodPolicy


_AUTH = ("canonical_header", "nonce", "public_payload", "protected_payload")
_SHAPE = ("image_shape", "field_lengths", "constant_descriptor")


def _policy(
    method_id: str,
    *,
    source: str = "protected_payload",
    domain: str = "rgb_u8",
    leakage: tuple[str, ...] = _SHAPE,
    randomness: tuple[str, ...] = (),
    recovery: tuple[str, ...] = (),
    auth: tuple[str, ...] = (),
    functionality: tuple[str, ...] = (),
    equivalence: str = "same_shape_lengths_and_declared_deterministic_fields",
    preentropy: str = "shape_only_message_space",
    descriptor_entropy: str = "shape_only_message_space",
    post_recovery: str = "not_asserted",
    p1: str,
    p2: str,
    p1_correct: bool,
    p2_applicable: bool = True,
    public_inverse: bool = False,
    nonce_length: int = 12,
    map_class: str = "keyed_or_randomized_byte_map",
    timing_path: str = "primitive_only",
    notes: str = "",
) -> MethodPolicy:
    return MethodPolicy(
        method_id=method_id,
        metric_body_source=source,
        body_metric_domain=domain,
        deterministic_plaintext_leakage=leakage,
        public_randomness=randomness,
        public_recovery_material=recovery,
        authenticated_coverage=auth,
        permitted_functionality=functionality,
        leakage_equivalence_rule=equivalence,
        prechallenge_entropy_rule=preentropy,
        descriptor_entropy_rule=descriptor_entropy,
        post_object_recovery=post_recovery,
        p1_semantics=p1,
        p2_semantics=p2,
        p1_correct_use=p1_correct,
        p2_applicable=p2_applicable,
        publicly_invertible=public_inverse,
        nonce_length=nonce_length,
        common_map_class=map_class,
        timing_path=timing_path,
        notes=notes,
    )


POLICIES: dict[str, MethodPolicy] = {
    "B01_aes_gcm": _policy(
        "B01_aes_gcm", randomness=("nonce",), auth=_AUTH,
        p1="forced AES-GCM nonce reuse", p2="fresh nonce correct use", p1_correct=False,
    ),
    "B02_chacha20_poly1305": _policy(
        "B02_chacha20_poly1305", randomness=("nonce",), auth=_AUTH,
        p1="forced ChaCha20-Poly1305 nonce reuse", p2="fresh nonce correct use", p1_correct=False,
    ),
    "B03_shake_hmac": _policy(
        "B03_shake_hmac", randomness=("nonce",), auth=_AUTH, nonce_length=16,
        p1="forced effective-mask reuse", p2="fresh nonce correct use", p1_correct=False,
    ),
    "B04_public_high_entropy": _policy(
        "B04_public_high_entropy", randomness=("public nonce",),
        recovery=("serialized nonce", "public SHAKE algorithm", "method identifier"),
        post_recovery="zero_via_public_inverse", public_inverse=True,
        p1="same public mask", p2="independent public masks", p1_correct=True,
        map_class="public_additive_mask",
        notes="corrected control removes the historical dependence on nonserialized image_id.",
    ),
    "B05_identity": _policy(
        "B05_identity", source="public_payload", domain="int32_serialized",
        leakage=("image_shape", "field_lengths", "complete public invertible payload", "constant descriptor"),
        recovery=("public payload", "public inverse"), equivalence="identical complete public payload",
        preentropy="singleton_complete_leakage_class", descriptor_entropy="shape_only_message_space",
        post_recovery="zero_via_public_inverse", public_inverse=True,
        p1="same deterministic public identity map", p2="not applicable: no fresh state",
        p1_correct=True, p2_applicable=False, map_class="public_bijection", timing_path="geometry_only",
    ),
    "B06_reversible_color": _policy(
        "B06_reversible_color", source="public_payload", domain="int32_serialized",
        leakage=("image_shape", "field_lengths", "complete public invertible payload", "constant RCT descriptor"),
        recovery=("public payload", "public inverse"), equivalence="identical complete public payload",
        preentropy="singleton_complete_leakage_class", descriptor_entropy="shape_only_message_space",
        post_recovery="zero_via_public_inverse", public_inverse=True,
        p1="same deterministic public color map", p2="not applicable: no fresh state",
        p1_correct=True, p2_applicable=False, map_class="public_bijection", timing_path="geometry_only",
    ),
    "B07_pca_klt": _policy(
        "B07_pca_klt", source="public_payload", domain="int32_serialized",
        leakage=("image_shape", "field_lengths", "PCA descriptor", "complete public invertible payload"),
        recovery=("public payload", "PCA descriptor", "public inverse"),
        equivalence="identical PCA descriptor and complete public payload",
        preentropy="singleton_complete_leakage_class", descriptor_entropy="pca_pixel_multiset_orbit",
        post_recovery="zero_via_public_inverse", public_inverse=True,
        p1="common external context with descriptor recomputed from each plaintext",
        p2="not applicable: no external fresh state", p1_correct=True, p2_applicable=False,
        map_class="plaintext_adaptive_public_bijection", timing_path="adaptive_geometry_only",
    ),
    "B08_random_so3": _policy(
        "B08_random_so3", source="public_payload", domain="int32_serialized",
        leakage=("image_shape", "field_lengths", "public rotation descriptor", "complete public invertible payload"),
        randomness=("public transform seed",), recovery=("public payload", "rotation descriptor", "public inverse"),
        equivalence="identical public transform context and complete public payload",
        preentropy="singleton_complete_leakage_class", descriptor_entropy="shape_only_message_space",
        post_recovery="zero_via_public_inverse", public_inverse=True,
        p1="same public rotation", p2="independent public rotations", p1_correct=True,
        map_class="public_bijection", timing_path="geometry_only",
    ),
    "B09_fixed_axis_quaternion": _policy(
        "B09_fixed_axis_quaternion", source="public_payload", domain="int32_serialized",
        leakage=("image_shape", "field_lengths", "constant rotation descriptor", "complete public invertible payload"),
        recovery=("public payload", "rotation descriptor", "public inverse"),
        equivalence="identical complete public payload", preentropy="singleton_complete_leakage_class",
        descriptor_entropy="shape_only_message_space", post_recovery="zero_via_public_inverse", public_inverse=True,
        p1="same deterministic public rotation", p2="not applicable: no fresh state",
        p1_correct=True, p2_applicable=False, map_class="public_bijection", timing_path="geometry_only",
    ),
    "B10_caseii": _policy(
        "B10_caseii", source="public_payload", domain="int32_serialized",
        leakage=("image_shape", "field_lengths", "Case-II schedule descriptor", "complete public invertible payload"),
        recovery=("public payload", "schedule descriptor", "public inverse"),
        equivalence="identical complete public payload", preentropy="singleton_complete_leakage_class",
        descriptor_entropy="shape_only_message_space", post_recovery="zero_via_public_inverse", public_inverse=True,
        p1="same deterministic public schedule", p2="not applicable: no fresh state",
        p1_correct=True, p2_applicable=False, map_class="public_bijection", timing_path="geometry_only",
    ),
    "B11_curvature_caseii": _policy(
        "B11_curvature_caseii", source="public_payload", domain="int32_serialized",
        leakage=("image_shape", "field_lengths", "curvature descriptor", "complete public invertible payload"),
        recovery=("public payload", "curvature descriptor", "public inverse"),
        equivalence="identical curvature descriptor and complete public payload",
        preentropy="singleton_complete_leakage_class", descriptor_entropy="curvature_residue_orbit",
        post_recovery="zero_via_public_inverse", public_inverse=True,
        p1="common external context with descriptor recomputed from each plaintext",
        p2="not applicable: no external fresh state", p1_correct=True, p2_applicable=False,
        map_class="plaintext_adaptive_public_bijection", timing_path="adaptive_geometry_only",
    ),
    "B12_chebyshev_only": _policy(
        "B12_chebyshev_only", randomness=("public nonce",),
        recovery=("serialized nonce", "public Chebyshev algorithm"), post_recovery="zero_via_public_inverse",
        public_inverse=True, p1="same public legacy mask", p2="independent public legacy masks",
        p1_correct=True, map_class="public_additive_mask", timing_path="custom_only",
    ),
    "B13_permutation_only": _policy(
        "B13_permutation_only", domain="opaque_u8", randomness=("nonce-conditioned secret permutation",),
        p1="reused keyed permutation state", p2="independent nonce-conditioned permutation",
        p1_correct=False, map_class="keyed_position_permutation", timing_path="custom_only",
    ),
    "B14_diffusion_only": _policy(
        "B14_diffusion_only", randomness=("nonce-conditioned secret diffusion state",),
        p1="reused keyed diffusion state", p2="independent nonce-conditioned diffusion state",
        p1_correct=False, map_class="keyed_serial_diffusion", timing_path="custom_only",
    ),
    "B15_geometry_shake_hmac": _policy(
        "B15_geometry_shake_hmac", domain="opaque_u8",
        leakage=("image_shape", "field_lengths", "constant Case-II schedule descriptor"),
        randomness=("nonce",), auth=_AUTH, nonce_length=16,
        p1="forced effective-mask reuse", p2="fresh nonce correct use", p1_correct=False,
        map_class="geometry_then_stream_mask", timing_path="geometry_plus_primitive",
    ),
    "B16_geometry_aes_gcm": _policy(
        "B16_geometry_aes_gcm", domain="opaque_u8",
        leakage=("image_shape", "field_lengths", "constant Case-II schedule descriptor"),
        randomness=("nonce",), auth=_AUTH,
        p1="forced AES-GCM nonce reuse", p2="fresh nonce correct use", p1_correct=False,
        map_class="geometry_then_aead", timing_path="geometry_plus_primitive",
    ),
    "B17_tip_r0_emulation": _policy(
        "B17_tip_r0_emulation", domain="opaque_u8",
        leakage=("image_shape", "field_lengths", "curvature descriptor", "Case-II schedule descriptor"),
        randomness=("nonce-conditioned mask, permutation, and diffusion state",), auth=_AUTH,
        equivalence="same shape and identical curvature/Case-II descriptors",
        preentropy="curvature_residue_orbit", descriptor_entropy="curvature_residue_orbit",
        p1="forced nonce/state reuse with each plaintext descriptor recomputed",
        p2="fresh nonce correct-use emulation", p1_correct=False,
        map_class="plaintext_adaptive_geometry_then_keyed_cascade", timing_path="geometry_plus_custom_primitive",
    ),
    "B18_external_quaternion_feistel": _policy(
        "B18_external_quaternion_feistel", domain="opaque_u8",
        randomness=("nonce-conditioned secret Feistel key schedule",),
        p1="reused keyed Feistel state", p2="independent nonce-conditioned state",
        p1_correct=False, map_class="keyed_block_feistel", timing_path="custom_only",
    ),
    "B19_external_chaos_pd": _policy(
        "B19_external_chaos_pd", domain="opaque_u8",
        randomness=("nonce-conditioned secret diffusion state",),
        p1="reused keyed permutation-diffusion state", p2="independent nonce-conditioned state",
        p1_correct=False, map_class="fixed_public_permutation_then_keyed_diffusion", timing_path="custom_only",
    ),
    "B20_full_aead_explicit_preview": _policy(
        "B20_full_aead_explicit_preview",
        leakage=("image_shape", "field_lengths", "explicit public PNG preview"), randomness=("nonce",), auth=_AUTH,
        functionality=("public 24x24 thumbnail preview",),
        equivalence="same shape, lengths, and byte-identical public preview",
        preentropy="preview_preimage_not_formally_lower_bounded", descriptor_entropy="preview_collision_class_empirical_only",
        p1="forced AES-GCM nonce reuse while retaining permitted preview leakage",
        p2="fresh nonce correct use with permitted preview leakage", p1_correct=False,
        map_class="explicit_preview_plus_aead", timing_path="preview_plus_primitive",
    ),
    "B21_public_fresh_pad": _policy(
        "B21_public_fresh_pad", randomness=("published full-length pad",),
        recovery=("published pad", "modular subtraction"), post_recovery="zero_via_public_inverse",
        public_inverse=True, p1="same published pad", p2="independent published pads", p1_correct=True,
        map_class="public_additive_pad", timing_path="public_control",
    ),
    "B22_public_wideblock_prp": _policy(
        "B22_public_wideblock_prp", domain="opaque_u8", randomness=("published 256-bit Feistel key",),
        recovery=("published key", "public Feistel inverse"), post_recovery="zero_via_public_inverse",
        public_inverse=True, p1="same published Feistel key", p2="independent published Feistel keys",
        p1_correct=True, map_class="public_wideblock_bijection", timing_path="public_control",
        notes="Engineering instantiation only; the random-permutation theorem remains a separate idealized result.",
    ),
    "B23_secure_fixed_header": _policy(
        "B23_secure_fixed_header", domain="fixed_prefix_rgb_suffix", randomness=("nonce",), auth=_AUTH,
        p1="forced AES-GCM nonce reuse", p2="fresh nonce correct use", p1_correct=False,
        map_class="fixed_nonuniform_prefix_plus_aead", timing_path="primitive_plus_release_check",
    ),
    "B24_aes_gcm_siv": _policy(
        "B24_aes_gcm_siv", randomness=("nonce",), auth=_AUTH,
        p1="forced nonce reuse under misuse-resistant AEAD", p2="fresh nonce correct use",
        p1_correct=False, map_class="misuse_resistant_aead", timing_path="primitive_only",
    ),
}


def validate_registry() -> None:
    expected = list(EXTENDED_METHOD_FACTORIES)
    if list(POLICIES) != expected:
        raise RuntimeError("protocol policy order does not match the construction registry")
    valid_sources = {"protected_payload", "public_payload"}
    valid_domains = {"rgb_u8", "opaque_u8", "int32_serialized", "fixed_prefix_rgb_suffix"}
    for item in POLICIES.values():
        if item.metric_body_source not in valid_sources:
            raise RuntimeError(f"invalid body source for {item.method_id}")
        if item.body_metric_domain not in valid_domains:
            raise RuntimeError(f"invalid body domain for {item.method_id}")
        if item.nonce_length not in {12, 16}:
            raise RuntimeError(f"invalid nonce length for {item.method_id}")
        method = make_method(item.method_id, profile="extended")
        if bool(item.authenticated_coverage) != bool(method.authenticated):
            raise RuntimeError(f"authentication declaration mismatch for {item.method_id}")


def method_policy(method_id: str) -> MethodPolicy:
    try:
        return POLICIES[method_id]
    except KeyError as exc:
        raise KeyError(f"unknown protocol method: {method_id}") from exc


def protocol_method_registry() -> list[dict[str, object]]:
    metadata = {row["method_id"]: row for row in method_registry("extended")}
    rows: list[dict[str, object]] = []
    for method_id, policy in POLICIES.items():
        row = dict(metadata[method_id])
        row.update(policy.to_dict())
        rows.append(row)
    return rows


def operation_regime(method_id: str, protocol_name: str) -> str:
    policy = method_policy(method_id)
    if protocol_name == "P1_common_context":
        if not policy.p1_correct_use:
            return "forced_reuse_misuse"
        if policy.publicly_invertible:
            return "public_invertible_control"
        return "deterministic_no_freshness"
    if protocol_name == "P2_fresh_randomness":
        if not policy.p2_applicable:
            raise ValueError(f"P2 is not applicable to {method_id}")
        if method_id == "B20_full_aead_explicit_preview":
            return "permitted_leakage_control"
        if policy.publicly_invertible:
            return "public_invertible_control"
        return "correct_use"
    raise ValueError(f"unknown protocol: {protocol_name}")


validate_registry()
