# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from structsplat.observation_field import (
    AlphaSemantics,
    CameraMetadata,
    CanvasCropTransform,
    FieldAdaptation,
    FieldSemantics,
    FilterSemantics,
    ObservationField2D,
    SupportSemantics,
    adapt_direct_additive,
    adapt_factorized_additive_gaussian_field,
    adapt_normalized_gaussian_field,
    clip_for_display,
    pack_alpha,
    unpack_alpha,
)


def _canvas(width: int = 5, height: int = 4, *, x: int = 0, y: int = 0):
    return CanvasCropTransform(
        canvas_width=width + x,
        canvas_height=height + y,
        crop_x=x,
        crop_y=y,
        crop_width=width,
        crop_height=height,
    )


def _semantics(
    *,
    domain: str = "signed",
    support: SupportSemantics | None = None,
    filtering: FilterSemantics | None = None,
    alpha: AlphaSemantics | None = None,
    background: bool = False,
):
    return FieldSemantics(
        coefficient_domain=domain,
        background_mode="explicit_counted_dc" if background else "zero_dc",
        support=support or SupportSemantics(mode="infinite"),
        filtering=filtering or FilterSemantics(),
        alpha=alpha or AlphaSemantics(),
    )


def _field(
    *,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
    coeff: np.ndarray | None = None,
    semantics: FieldSemantics | None = None,
    structural_mass: np.ndarray | None = None,
    filter_variance: np.ndarray | None = None,
    background: np.ndarray | None = None,
    packed_alpha: np.ndarray | None = None,
    canvas: CanvasCropTransform | None = None,
    camera: CameraMetadata | None = None,
):
    means = np.asarray([[1.0, 1.0]], dtype=np.float32) if means is None else means
    n = means.shape[0]
    scales = np.ones((n, 2), dtype=np.float32) if scales is None else scales
    rotations = np.zeros(n, dtype=np.float32) if rotations is None else rotations
    coeff = np.ones((n, 3), dtype=np.float32) if coeff is None else coeff
    return ObservationField2D(
        means_xy=means,
        log_scales_xy=np.log(scales),
        rotations_rad=rotations,
        rgb_coeff=coeff,
        canvas_crop=canvas or _canvas(),
        semantics=semantics or _semantics(),
        structural_mass=structural_mass,
        filter_variance_px2=filter_variance,
        background_rgb=background,
        packed_alpha=packed_alpha,
        camera=camera,
    )


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return json.loads(arrays["__metadata__"].tobytes().decode("utf-8"))


def _replace_metadata(arrays: dict[str, np.ndarray], value: dict[str, object]) -> None:
    arrays["__metadata__"] = np.frombuffer(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    ).copy()


def test_canvas_crop_transform_is_explicit_and_bounded():
    transform = _canvas(4, 3, x=7, y=11)
    local = np.array([[0.0, 0.0], [3.0, 2.0]])
    assert np.array_equal(transform.crop_to_canvas(local), [[7.0, 11.0], [10.0, 13.0]])
    assert np.array_equal(transform.canvas_to_crop([[7.0, 11.0]]), [[0.0, 0.0]])
    with pytest.raises(ValueError, match="beyond canvas width"):
        CanvasCropTransform(10, 10, 9, 0, 2, 1)
    with pytest.raises(TypeError, match="must be an integer"):
        CanvasCropTransform(True, 10, 0, 0, 1, 1)


def test_constructor_owns_arrays_and_validates_shape_domain_and_mass():
    means = np.array([[1.0, 1.0]], dtype=np.float32)
    value = _field(means=means, semantics=_semantics(domain="nonnegative"))
    means[0, 0] = 99.0
    assert value.means_xy[0, 0] == 1.0
    assert not value.means_xy.flags.writeable
    with pytest.raises(ValueError, match="negative rgb_coeff"):
        _field(
            coeff=np.array([[-0.1, 0.0, 0.0]], dtype=np.float32),
            semantics=_semantics(domain="nonnegative"),
        )
    with pytest.raises(ValueError, match="structural_mass must be nonnegative"):
        _field(structural_mass=np.array([-1.0], dtype=np.float32))
    with pytest.raises(TypeError, match="dtype float32 or float64"):
        _field(means=np.array([[1, 1]], dtype=np.int32))
    with pytest.raises(ValueError, match="unsupported ObservationField2D schema"):
        ObservationField2D(
            means_xy=np.empty((0, 2), dtype=np.float32),
            log_scales_xy=np.empty((0, 2), dtype=np.float32),
            rotations_rad=np.empty(0, dtype=np.float32),
            rgb_coeff=np.empty((0, 3), dtype=np.float32),
            canvas_crop=_canvas(),
            semantics=_semantics(),
            schema_version="3.0.0",
        )


def test_filter_and_background_semantics_fail_closed():
    filtered = FilterSemantics(mode="isotropic_covariance_add", aa_dilation_px2=0.25)
    value = _field(semantics=_semantics(filtering=filtered))
    assert np.allclose(value._effective_variance(), (np.array([1.25]), np.array([1.25])))
    with pytest.raises(ValueError, match="requires filter.mode"):
        _field(filter_variance=np.array([0.2], dtype=np.float32))
    with pytest.raises(ValueError, match="requires nonzero filter data"):
        _field(
            semantics=_semantics(
                filtering=FilterSemantics(mode="isotropic_covariance_add")
            )
        )
    with pytest.raises(ValueError, match="background_mode does not match"):
        _field(background=np.zeros(3, dtype=np.float32))


def test_camera_metadata_is_json_finite_and_immutable():
    source = {"K": [1.0, 0.0, 2.0], "nested": {"model": "pinhole"}}
    camera = CameraMetadata("realtime-gs.camera.v1", source)
    source["nested"]["model"] = "mutated"
    assert camera.payload["nested"]["model"] == "pinhole"
    with pytest.raises(TypeError):
        camera.payload["new"] = 1
    with pytest.raises(ValueError, match="nonfinite"):
        CameraMetadata("camera.v1", {"fx": float("nan")})


def test_no_rows_zero_and_counted_background_closed_form():
    no_rows = np.empty((0, 2), dtype=np.float64)
    value = _field(
        means=no_rows,
        scales=np.empty((0, 2), dtype=np.float64),
        rotations=np.empty(0, dtype=np.float64),
        coeff=np.empty((0, 3), dtype=np.float64),
        structural_mass=np.empty(0, dtype=np.float64),
        background=np.array([0.25, -0.5, 1.5], dtype=np.float64),
        semantics=_semantics(background=True),
    )
    points = np.array([[0.0, 0.0], [4.0, 3.0]])
    assert np.array_equal(value.kernel_weights(points), np.empty((2, 0)))
    assert np.array_equal(
        value.appearance_raw(points),
        np.array([[0.25, -0.5, 1.5], [0.25, -0.5, 1.5]]),
    )
    assert np.array_equal(value.structural_density(points), np.zeros(2))
    assert value.responsibilities(points).shape == (2, 0)


def test_one_row_peak_one_gaussian_closed_form():
    value = _field(
        means=np.array([[0.0, 0.0]], dtype=np.float64),
        scales=np.array([[1.0, 2.0]], dtype=np.float64),
        coeff=np.array([[2.0, -1.0, 0.5]], dtype=np.float64),
        structural_mass=np.array([3.0], dtype=np.float64),
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    expected_weight = np.array([1.0, np.exp(-0.5), np.exp(-0.5)])
    assert np.allclose(value.kernel_weights(points)[:, 0], expected_weight)
    assert np.allclose(value.appearance_raw(points), expected_weight[:, None] * [2, -1, 0.5])
    assert np.allclose(value.structural_density(points), expected_weight * 3.0)


def test_overlap_density_responsibility_and_near_zero_are_stable():
    value = _field(
        means=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        scales=np.ones((2, 2), dtype=np.float32),
        coeff=np.array([[0.25, 0.0, 0.0], [0.75, 1.0, 0.0]], dtype=np.float32),
        structural_mass=np.array([1.0, 3.0], dtype=np.float32),
    )
    point = np.array([[1.0, 1.0]])
    assert np.allclose(value.appearance_raw(point), [[1.0, 1.0, 0.0]])
    assert np.allclose(value.structural_density(point), [4.0])
    assert np.allclose(value.responsibilities(point, epsilon=1e-12), [[0.25, 0.75]])

    far = np.array([[1e6, 1e6]])
    assert value.structural_density(far)[0] == 0.0
    responsibility = value.responsibilities(far)
    assert np.array_equal(responsibility, np.zeros((1, 2)))
    assert np.isfinite(responsibility).all()
    with pytest.raises(ValueError, match="epsilon must be > 0"):
        value.responsibilities(point, epsilon=0.0)


def test_structural_mass_is_optional_and_never_derived_from_rgb():
    value = _field()
    with pytest.raises(ValueError, match="independently defined structural_mass"):
        value.structural_density([[1.0, 1.0]])
    with pytest.raises(ValueError, match="independently defined structural_mass"):
        value.responsibilities([[1.0, 1.0]])


def test_bbox_support_keeps_visible_tail_of_off_canvas_row():
    support = SupportSemantics(mode="axis_aligned_bbox", sigma_cutoff=3.0)
    value = _field(
        means=np.array([[-1.0, 1.0], [-10.0, 1.0]], dtype=np.float64),
        scales=np.ones((2, 2), dtype=np.float64),
        coeff=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        semantics=_semantics(support=support),
    )
    pixel = value.appearance_raw([[0.0, 1.0]])[0]
    assert np.allclose(pixel[0], np.exp(-0.5))
    assert pixel[1] == 0.0


def test_support_modes_and_fade_are_semantically_distinct():
    means = np.array([[0.0, 0.0]], dtype=np.float64)
    scales = np.ones((1, 2), dtype=np.float64)
    coeff = np.ones((1, 3), dtype=np.float64)
    corner = np.array([[3.0, 3.0]])
    bbox = _field(
        means=means,
        scales=scales,
        coeff=coeff,
        semantics=_semantics(support=SupportSemantics(mode="axis_aligned_bbox")),
    )
    ellipse = _field(
        means=means,
        scales=scales,
        coeff=coeff,
        semantics=_semantics(support=SupportSemantics(mode="ellipse")),
    )
    faded = _field(
        means=means,
        scales=scales,
        coeff=coeff,
        semantics=_semantics(
            support=SupportSemantics(mode="axis_aligned_bbox", fade_alpha=1.0)
        ),
    )
    assert bbox.kernel_weights(corner)[0, 0] > 0.0
    assert ellipse.kernel_weights(corner)[0, 0] == 0.0
    assert faded.kernel_weights([[3.0, 0.0]])[0, 0] == 0.0
    with pytest.raises(ValueError, match="canonical unused sigma_cutoff"):
        SupportSemantics(mode="infinite", sigma_cutoff=4.0)
    with pytest.raises(ValueError, match="canonical unused minimum_radius"):
        SupportSemantics(mode="ellipse", minimum_radius_px=2)


def test_exact_and_explicitly_thresholded_alpha_pack_round_trip():
    binary = np.array([[True, False, True], [False, True, False]])
    packed = pack_alpha(binary)
    assert np.array_equal(unpack_alpha(packed, 2, 3), binary)
    with pytest.raises(ValueError, match="explicit lossy threshold"):
        pack_alpha(np.array([[0.2, 0.8]], dtype=np.float32))
    thresholded = pack_alpha(np.array([[0.2, 0.8]], dtype=np.float32), threshold=0.5)
    assert np.array_equal(unpack_alpha(thresholded, 1, 2), [[False, True]])
    AlphaSemantics(
        payload_encoding="binary_thresholded_packbits_little", source_threshold=0.5
    )
    with pytest.raises(ValueError, match="requires source_threshold"):
        AlphaSemantics(payload_encoding="binary_thresholded_packbits_little")


def test_alpha_matte_raw_oracle_and_canvas_query_are_separate():
    mask = np.array([[True, False], [False, True]])
    alpha = AlphaSemantics(
        payload_encoding="binary_exact_packbits_little", matting_mode="multiply_alpha"
    )
    value = _field(
        means=np.empty((0, 2), dtype=np.float32),
        scales=np.empty((0, 2), dtype=np.float32),
        rotations=np.empty(0, dtype=np.float32),
        coeff=np.empty((0, 3), dtype=np.float32),
        background=np.array([0.5, 0.25, 1.5], dtype=np.float32),
        semantics=_semantics(alpha=alpha, background=True),
        packed_alpha=pack_alpha(mask),
        canvas=_canvas(2, 2, x=7, y=9),
    )
    raw = value.render_raw()
    matted = value.render_matted()
    assert np.allclose(raw, [0.5, 0.25, 1.5])
    assert np.array_equal(matted[0, 1], np.zeros(3))
    assert np.allclose(matted[1, 1], [0.5, 0.25, 1.5])
    assert np.array_equal(value.alpha_at([[7.0, 9.0]], coordinate_space="canvas"), [1.0])
    assert np.array_equal(value.alpha_at([[6.0, 9.0]], coordinate_space="canvas"), [0.0])


def test_alpha_binding_rejects_missing_payload_and_nonzero_tail_bits():
    alpha = AlphaSemantics(payload_encoding="binary_exact_packbits_little")
    with pytest.raises(ValueError, match="requires packed_alpha"):
        _field(semantics=_semantics(alpha=alpha))
    with pytest.raises(ValueError, match="nonzero unused tail bits"):
        _field(
            semantics=_semantics(alpha=alpha),
            packed_alpha=np.full(3, 255, dtype=np.uint8),
        )


def test_signed_coefficients_dc_and_display_clip_never_change_raw_oracle():
    value = _field(
        coeff=np.array([[2.0, -1.0, 0.5]], dtype=np.float64),
        background=np.array([-0.25, 0.5, 1.0], dtype=np.float64),
        semantics=_semantics(domain="signed", background=True),
    )
    raw = value.appearance_raw([[1.0, 1.0]])
    assert np.allclose(raw, [[1.75, -0.5, 1.5]])
    shown = clip_for_display(raw)
    assert np.array_equal(shown, [[1.0, 0.0, 1.0]])
    assert np.allclose(value.appearance_raw([[1.0, 1.0]]), raw)


def test_lossless_round_trip_preserves_every_array_semantic_and_hash(tmp_path):
    mask = np.array([[True, False, True], [False, True, False]])
    filtering = FilterSemantics(mode="isotropic_covariance_add", aa_dilation_px2=0.125)
    alpha = AlphaSemantics(
        payload_encoding="binary_exact_packbits_little",
        matting_mode="multiply_alpha",
        boundary_policy="hard_contained",
    )
    value = _field(
        means=np.array([[-0.0, 0.5], [2.0, 1.0]], dtype=np.float64),
        scales=np.array([[0.75, 1.25], [2.0, 0.5]], dtype=np.float64),
        rotations=np.array([0.25, -0.5], dtype=np.float64),
        coeff=np.array([[1.2, -0.2, 0.3], [-0.4, 0.5, 1.6]], dtype=np.float64),
        structural_mass=np.array([0.0, 2.5], dtype=np.float64),
        filter_variance=np.array([0.0, 0.75], dtype=np.float64),
        background=np.array([0.1, -0.1, 0.2], dtype=np.float64),
        semantics=_semantics(
            support=SupportSemantics(mode="ellipse", sigma_cutoff=4.0, fade_alpha=0.2),
            filtering=filtering,
            alpha=alpha,
            background=True,
        ),
        packed_alpha=pack_alpha(mask),
        canvas=_canvas(3, 2, x=4, y=5),
        camera=CameraMetadata(
            "realtime-gs.camera.v1", {"K": [10.0, 0.0, 1.0], "camera_id": "C0014"}
        ),
    )
    first = tmp_path / "field.of2d.npz"
    second = tmp_path / "field-copy.of2d.npz"
    value.save_lossless(first)
    value.save_lossless(second)
    loaded = ObservationField2D.load_lossless(first)
    assert first.read_bytes() == second.read_bytes()
    assert loaded.canonical_hash() == value.canonical_hash()
    assert loaded.semantic_record() == value.semantic_record()
    for name, array in value._array_items().items():
        restored = loaded._array_items()[name]
        assert restored.dtype == array.dtype
        assert restored.shape == array.shape
        assert restored.tobytes() == array.tobytes()
    assert np.array_equal(loaded.render_matted(), value.render_matted())


@pytest.mark.parametrize("mutation", ["version", "extra_metadata", "content_hash"])
def test_loader_rejects_unknown_or_mismatched_metadata(tmp_path, mutation):
    path = tmp_path / "valid.npz"
    _field().save_lossless(path)
    arrays = _npz_arrays(path)
    metadata = _metadata(arrays)
    if mutation == "version":
        metadata["field"]["schema_version"] = "99.0.0"
    elif mutation == "extra_metadata":
        metadata["surprise"] = True
    else:
        metadata["content_sha256"] = "0" * 64
    _replace_metadata(arrays, metadata)
    damaged = tmp_path / f"damaged-{mutation}.npz"
    np.savez(damaged, **arrays)
    with pytest.raises(ValueError):
        ObservationField2D.load_lossless(damaged)


def test_loader_rejects_unknown_array_and_payload_tamper(tmp_path):
    path = tmp_path / "valid.npz"
    _field().save_lossless(path)
    arrays = _npz_arrays(path)
    with_extra = dict(arrays)
    with_extra["surprise"] = np.zeros(1, dtype=np.float32)
    extra_path = tmp_path / "extra.npz"
    np.savez(extra_path, **with_extra)
    with pytest.raises(ValueError, match="members do not match"):
        ObservationField2D.load_lossless(extra_path)

    arrays["rgb_coeff"] = arrays["rgb_coeff"].copy()
    arrays["rgb_coeff"][0, 0] += 1.0
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **arrays)
    with pytest.raises(ValueError, match="payload mismatch"):
        ObservationField2D.load_lossless(tampered)


def test_loader_rejects_duplicate_zip_member(tmp_path):
    path = tmp_path / "duplicate.npz"
    payload = io_bytes = b"not important"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("__metadata__.npy", payload)
            archive.writestr("__metadata__.npy", io_bytes)
    with pytest.raises(ValueError, match="duplicate members"):
        ObservationField2D.load_lossless(path)


def test_direct_adapter_declares_both_exactness_axes():
    result = adapt_direct_additive(
        means_xy=np.zeros((0, 2), dtype=np.float32),
        log_scales_xy=np.zeros((0, 2), dtype=np.float32),
        rotations_rad=np.zeros(0, dtype=np.float32),
        rgb_coeff=np.zeros((0, 3), dtype=np.float32),
        canvas_crop=_canvas(),
        semantics=_semantics(domain="nonnegative"),
    )
    assert result.pixel_exact
    assert result.component_semantics_exact
    assert result.require_pixel_exact() is result.field


def test_factorized_additive_adapter_matches_current_reference_pixels():
    torch = pytest.importorskip("torch")
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    means = np.array([[-0.5, 1.0], [2.25, 2.0]], dtype=np.float32)
    scales = np.array([[1.1, 0.7], [0.8, 1.5]], dtype=np.float32)
    rotations = np.array([0.3, -0.6], dtype=np.float32)
    colors = np.array([[0.8, 0.2, 0.1], [0.1, 0.4, 0.9]], dtype=np.float32)
    logits = np.array([0.2, -0.7], dtype=np.float32)
    filter_variance = np.array([0.0, 0.3], dtype=np.float32)
    legacy = GaussianField.from_numpy(
        means,
        scales,
        rotations,
        colors,
        opacities=logits,
        filter_variance=filter_variance,
    )
    converted = adapt_factorized_additive_gaussian_field(
        legacy,
        canvas_crop=_canvas(5, 4),
        coefficient_domain="nonnegative",
        sigma_cutoff=3.0,
        support_fade_alpha=1.0,
        aa_dilation_px2=0.2,
    )
    assert converted.pixel_exact
    assert not converted.component_semantics_exact
    field_value = converted.require_pixel_exact()
    current = render_field(
        legacy.means,
        legacy.conics(dilation=0.2),
        legacy.colors,
        legacy.radii(3.0, dilation=0.2),
        4,
        5,
        mode="additive",
        opacities=legacy.opacity_values(),
        support_fade=True,
        sigma_cutoff=3.0,
    )
    assert np.allclose(field_value.render_raw(), current.detach().numpy(), atol=2e-7, rtol=2e-7)
    assert field_value.structural_mass is None
    assert torch.isfinite(current).all()


def test_factorized_adapter_rejects_unrepresented_affine_color():
    pytest.importorskip("torch")
    from structsplat.gaussians import GaussianField

    legacy = GaussianField.from_numpy(
        np.zeros((1, 2), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        color_grads=np.zeros((1, 2, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="cannot preserve affine"):
        adapt_factorized_additive_gaussian_field(
            legacy, canvas_crop=_canvas(), coefficient_domain="nonnegative"
        )


def test_normalized_adapter_refuses_equivalence_and_marks_approximation():
    pytest.importorskip("torch")
    from structsplat.gaussians import GaussianField

    legacy = GaussianField.from_numpy(
        np.zeros((1, 2), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )
    refused = adapt_normalized_gaussian_field(
        legacy, canvas_crop=_canvas(), coefficient_domain="nonnegative"
    )
    assert refused.field is None
    assert not refused.pixel_exact and not refused.component_semantics_exact
    with pytest.raises(ValueError, match="not an exact additive"):
        refused.require_pixel_exact()
    approximate = adapt_normalized_gaussian_field(
        legacy,
        canvas_crop=_canvas(),
        coefficient_domain="nonnegative",
        permit_inexact=True,
    )
    assert approximate.field is not None
    assert not approximate.pixel_exact and not approximate.component_semantics_exact
    with pytest.raises(ValueError, match="normalized adapter"):
        FieldAdaptation(
            adapter="bad",
            source_semantics="normalized_weighted_sum_v1",
            field=approximate.field,
            pixel_exact=True,
            component_semantics_exact=False,
            assumptions=("bad claim",),
        )


def test_existing_gaussian_default_npz_and_sspl1_paths_remain_legacy(tmp_path):
    pytest.importorskip("torch")
    from structsplat.codec import decode, encode
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    rng = np.random.default_rng(13)
    means = rng.uniform(0.0, 3.0, (3, 2)).astype(np.float32)
    scales = rng.uniform(0.7, 1.3, (3, 2)).astype(np.float32)
    rotations = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
    colors = rng.uniform(0.0, 1.0, (3, 3)).astype(np.float32)
    legacy = GaussianField.from_numpy(means, scales, rotations, colors)
    default = render_field(
        legacy.means, legacy.conics(), legacy.colors, legacy.radii(3.0), 4, 5
    )
    explicit = render_field(
        legacy.means,
        legacy.conics(),
        legacy.colors,
        legacy.radii(3.0),
        4,
        5,
        mode="normalized",
    )
    assert np.array_equal(default.detach().numpy(), explicit.detach().numpy())

    npz = tmp_path / "legacy.npz"
    legacy.save(str(npz))
    with np.load(npz, allow_pickle=False) as archive:
        assert set(archive.files) == {"means", "log_scales", "rotations", "colors"}
    blob = encode(legacy, 4, 5)
    assert blob.startswith(b"SSPL1")
    decoded = decode(blob)
    assert decoded.n == legacy.n


def test_observation_contract_and_numpy_layers_import_without_torch():
    command = (
        "import sys; sys.modules['torch'] = None; "
        "import structsplat.observation_field; "
        "import structsplat.structure_tensor; import structsplat.density; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
