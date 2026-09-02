from __future__ import annotations

from collections.abc import Callable

from .constructions import (
    ExternalChaosPD,
    ExternalQuaternionFeistel,
    FullAEADExplicitPreview,
    LegacyAblationConstruction,
    PublicHighEntropyConstruction,
    PublicTransformConstruction,
    RawAEADConstruction,
    ShakeHMACConstruction,
    TipR0Emulation,
    TransformCryptoConstruction,
)
from .controls import (
    AESGCMSIVConstruction,
    PublicHighEntropyConstruction,
    PublicFreshPadConstruction,
    PublicWideBlockPRPConstruction,
    SecureFixedHeaderConstruction,
)
from .plugins import ConstructionPlugin
from .transforms import (
    CaseIITransform,
    FixedAxisQuaternionTransform,
    IdentityTransform,
    PCATransform,
    RandomSO3Transform,
    ReversibleColorTransform,
)


def _factories() -> dict[str, Callable[[], ConstructionPlugin]]:
    return {
        "B01_aes_gcm": lambda: RawAEADConstruction("B01_aes_gcm", "Full-image AES-256-GCM", "aes-gcm-256"),
        "B02_chacha20_poly1305": lambda: RawAEADConstruction("B02_chacha20_poly1305", "Full-image ChaCha20-Poly1305", "chacha20-poly1305"),
        "B03_shake_hmac": ShakeHMACConstruction,
        "B04_public_high_entropy": PublicHighEntropyConstruction,
        "B05_identity": lambda: PublicTransformConstruction("B05_identity", "Identity / centered RGB public transform", "transform-control", "identity/RGB control", IdentityTransform()),
        "B06_reversible_color": lambda: PublicTransformConstruction("B06_reversible_color", "Reversible color-transform public control", "transform-control", "reversible color-transform control", ReversibleColorTransform()),
        "B07_pca_klt": lambda: PublicTransformConstruction("B07_pca_klt", "PCA/KLT exact lifting public transform", "transform-control", "adaptive PCA/KLT comparator", PCATransform()),
        "B08_random_so3": lambda: PublicTransformConstruction("B08_random_so3", "Seeded random SO(3) exact lifting transform", "transform-control", "random orthogonal comparator", RandomSO3Transform()),
        "B09_fixed_axis_quaternion": lambda: PublicTransformConstruction("B09_fixed_axis_quaternion", "Fixed-axis quaternion exact lifting transform", "quaternion-control", "fixed-axis quaternion comparator", FixedAxisQuaternionTransform()),
        "B10_caseii": lambda: PublicTransformConstruction("B10_caseii", "Canonical Case-II exact scheduled lifting transform", "quaternion-control", "canonical time-varying Case-II comparator", CaseIITransform(False)),
        "B11_curvature_caseii": lambda: PublicTransformConstruction("B11_curvature_caseii", "Curvature-oriented Case-II exact lifting transform", "quaternion-control", "curvature-oriented Case-II comparator", CaseIITransform(True)),
        "B12_chebyshev_only": lambda: LegacyAblationConstruction("B12_chebyshev_only", "Chebyshev-mask-only legacy ablation", "chebyshev-mask"),
        "B13_permutation_only": lambda: LegacyAblationConstruction("B13_permutation_only", "Permutation-only ablation", "permutation-only"),
        "B14_diffusion_only": lambda: LegacyAblationConstruction("B14_diffusion_only", "Two-pass diffusion-only ablation", "diffusion-only"),
        "B15_geometry_shake_hmac": lambda: TransformCryptoConstruction("B15_geometry_shake_hmac", "Case-II transform plus keyed SHAKE/HMAC", CaseIITransform(False), "shake-hmac"),
        "B16_geometry_aes_gcm": lambda: TransformCryptoConstruction("B16_geometry_aes_gcm", "Case-II transform plus AES-256-GCM", CaseIITransform(False), "aes-gcm"),
        "B17_tip_r0_emulation": lambda: TipR0Emulation(CaseIITransform(True)),
        "B18_external_quaternion_feistel": ExternalQuaternionFeistel,
        "B19_external_chaos_pd": ExternalChaosPD,
        "B20_full_aead_explicit_preview": FullAEADExplicitPreview,
    }


METHOD_FACTORIES = _factories()

EXTENDED_METHOD_FACTORIES: dict[str, Callable[[], ConstructionPlugin]] = {
    **METHOD_FACTORIES,
    "B04_public_high_entropy": PublicHighEntropyConstruction,
    "B21_public_fresh_pad": PublicFreshPadConstruction,
    "B22_public_wideblock_prp": PublicWideBlockPRPConstruction,
    "B23_secure_fixed_header": SecureFixedHeaderConstruction,
    "B24_aes_gcm_siv": AESGCMSIVConstruction,
}


def factories_for_profile(profile: str = "core") -> dict[str, Callable[[], ConstructionPlugin]]:
    if profile == "core":
        return METHOD_FACTORIES
    if profile in {"extended", "full"}:
        return EXTENDED_METHOD_FACTORIES
    raise ValueError(f"unknown benchmark profile: {profile}")


def make_method(method_id: str, profile: str = "core") -> ConstructionPlugin:
    factories = factories_for_profile(profile)
    try:
        return factories[method_id]()
    except KeyError as exc:
        raise KeyError(f"unknown {profile} method: {method_id}") from exc


def method_registry(profile: str = "core") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_id in factories_for_profile(profile):
        method = make_method(method_id, profile=profile)
        rows.append({
            "method_id": method.method_id,
            "display_name": method.display_name,
            "family": method.family,
            "benchmark_role": method.benchmark_role,
            "authenticated": method.authenticated,
            "secure_control": method.secure_control,
            "exact": method.exact,
            "component_ids": list(method.component_ids),
            "profile": profile,
        })
    return rows
