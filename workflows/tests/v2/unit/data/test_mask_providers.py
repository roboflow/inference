"""Mask providers: grid, dense and RLE masks behind one binary request."""

import threading
import time

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.data import rle as rle_module
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import (
    APPROXIMATE_BINARY,
    BINARY,
    SCORE_GRID,
    DenseMaskProvider,
    GridMaskProvider,
    MaskGridGeometry,
    RasterGeometry,
    RLEMaskProvider,
    SegmentationView,
    UnsupportedMaskRequest,
    mask_provider_of,
    providers,
)
from roboflow_workflows.v2.data.masks.operations import nearest_full_res
from roboflow_workflows.v2.engine.errors import ContractError
from roboflow_workflows.v2.engine.recording import (
    CodecRegistry,
    RecordingCodecError,
)

from inference_models.entities import ImageDimensions
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
)
from tests.v2.support.recording.codecs import MemoryBlobs

HEIGHT, WIDTH = 9, 11


def _frame_metadata(height=HEIGHT, width=WIDTH) -> dict:
    image = ImageData.from_tensor(
        torch.zeros((3, height, width), dtype=torch.uint8), image_id="frame"
    )
    metadata = FrameRef.from_image(image).prediction_metadata()

    return metadata


def _analytic_masks() -> torch.Tensor:
    """Row 0: square with a hole; row 1: two separate blobs; row 2: one pixel."""
    masks = torch.zeros((3, HEIGHT, WIDTH), dtype=torch.bool)
    masks[0, 1:7, 1:7] = True
    masks[0, 3:5, 3:5] = False
    masks[1, 0:2, 0:3] = True
    masks[1, 6:9, 8:11] = True
    masks[2, 8, 10] = True

    return masks


def _rows(metadata=None) -> dict:
    fields = {
        "xyxy": torch.tensor(
            [[1, 1, 7, 7], [0, 0, 11, 9], [10, 8, 11, 9]], dtype=torch.float32
        ),
        "class_id": torch.tensor([4, 5, 6]),
        "confidence": torch.tensor([0.9, 0.8, 0.7]),
        "image_metadata": metadata,
        "bboxes_metadata": [{"row": 0}, {"row": 1}, {"row": 2}],
    }

    return fields


def _dense(metadata=None) -> InstanceDetections:
    predictions = InstanceDetections(**_rows(metadata), mask=_analytic_masks())

    return predictions


def _rle(metadata=None) -> InstanceDetections:
    masks = _analytic_masks().numpy().transpose(1, 2, 0).astype(np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(masks))
    rle = InstancesRLEMasks.from_coco_rle_masks((HEIGHT, WIDTH), encoded)
    predictions = InstanceDetections(**_rows(metadata), mask=rle)

    return predictions


def _pre_processing(
    *, original_hw, resized_hw, inference_hw, padding_ltrb, crop_xywh=None
) -> PreProcessingMetadata:
    crop_xywh = crop_xywh or (0, 0, original_hw[1], original_hw[0])
    left, top, right, bottom = padding_ltrb
    metadata = PreProcessingMetadata(
        pad_left=left,
        pad_top=top,
        pad_right=right,
        pad_bottom=bottom,
        original_size=ImageDimensions(*original_hw),
        size_after_pre_processing=ImageDimensions(*resized_hw),
        inference_size=ImageDimensions(*inference_hw),
        scale_width=(inference_hw[1] - left - right) / resized_hw[1],
        scale_height=(inference_hw[0] - top - bottom) / resized_hw[0],
        static_crop_offset=StaticCropOffset(*crop_xywh),
    )

    return metadata


# Odd 9 x 11 image; static crop (2, 1, 7, 6) letterboxed into a 16 x 16 input.
ODD_CROP = _pre_processing(
    original_hw=(HEIGHT, WIDTH),
    resized_hw=(6, 7),
    inference_hw=(16, 16),
    padding_ltrb=(0, 1, 2, 1),
    crop_xywh=(2, 1, 7, 6),
)
# Centre crop: negative padding.
CENTRE_CROP = _pre_processing(
    original_hw=(HEIGHT, WIDTH),
    resized_hw=(HEIGHT, WIDTH),
    inference_hw=(8, 8),
    padding_ltrb=(-2, -1, -1, -2),
)


def _grid(
    pre_processing=ODD_CROP,
    *,
    grid_hw=(4, 4),
    dtype=torch.float32,
    seed=0,
    scores=None,
    metadata=None,
    threshold=0.0,
) -> GridMaskProvider:
    if scores is None:
        generator = torch.Generator().manual_seed(seed)
        scores = torch.randn((3, *grid_hw), generator=generator).to(dtype)
    view = SegmentationView(
        detections=Detections(**_rows(metadata)),
        scores=scores,
        score_type="logits",
        threshold=threshold,
        geometry=MaskGridGeometry.from_pre_processing(
            pre_processing, grid_size_hw=tuple(scores.shape[1:])
        ),
    )
    provider = GridMaskProvider(view)

    return provider


def _native_grid_mask(provider: GridMaskProvider, pre_processing) -> torch.Tensor:
    scores, recipe = provider.score_grid()
    _, mask = align_instance_segmentation_results(
        image_bboxes=torch.zeros((len(provider), 4)),
        masks=scores.clone(),
        padding=(
            pre_processing.pad_left,
            pre_processing.pad_top,
            pre_processing.pad_right,
            pre_processing.pad_bottom,
        ),
        scale_width=pre_processing.scale_width,
        scale_height=pre_processing.scale_height,
        original_size=pre_processing.original_size,
        size_after_pre_processing=pre_processing.size_after_pre_processing,
        inference_size=pre_processing.inference_size,
        static_crop_offset=pre_processing.static_crop_offset,
        binarization_threshold=recipe.threshold,
    )

    return mask


# ---------------------------------------------------------------------------
# Dense and RLE: analytic holes and disconnected regions
# ---------------------------------------------------------------------------


def test_dense_reference_is_the_supplied_raster_with_its_hole() -> None:
    predictions = _dense()
    provider = mask_provider_of(predictions)

    masks = provider.binary()

    assert isinstance(provider, DenseMaskProvider)
    assert masks.predictions is predictions
    assert masks.fidelity == "reference"
    assert masks.recipe.reference == providers.DENSE_REFERENCE
    assert masks.recipe.geometry == RasterGeometry(size_hw=(HEIGHT, WIDTH))
    assert not masks.predictions.mask[0, 3:5, 3:5].any()
    assert int(masks.predictions.mask[0].sum()) == 36 - 4


def test_rle_decodes_holes_and_disconnected_regions_like_the_native_helper() -> None:
    predictions = _rle()
    provider = mask_provider_of(predictions)

    masks = provider.binary()

    assert isinstance(provider, RLEMaskProvider)
    assert masks.fidelity == "reference"
    assert masks.predictions.mask.dtype == torch.bool
    assert torch.equal(masks.predictions.mask, _analytic_masks())
    assert np.array_equal(
        masks.predictions.mask.numpy(), coco_rle_masks_to_numpy_mask(predictions.mask)
    )
    assert masks.predictions.xyxy is predictions.xyxy
    assert masks.recipe.score_domain == "coco_rle"
    assert masks.recipe.score_dtype == "bytes"


def test_rle_provider_keeps_native_bytes_and_image_size() -> None:
    predictions = _rle()

    selected = mask_provider_of(predictions).select(indices=[2, 0, 2])

    rle = selected.predictions.mask
    assert rle.image_size == (HEIGHT, WIDTH)
    assert [id(counts) for counts in rle.masks] == [
        id(predictions.mask.masks[row]) for row in (2, 0, 2)
    ]
    assert not selected.is_decoded


# ---------------------------------------------------------------------------
# Selection: one row authority for every aligned field
# ---------------------------------------------------------------------------


def _all_providers(metadata=None):
    return {
        "grid": _grid(metadata=metadata),
        "dense": DenseMaskProvider(_dense(metadata)),
        "rle": RLEMaskProvider(_rle(metadata)),
    }


@pytest.mark.parametrize("kind", ["grid", "dense", "rle"])
def test_repeated_selection_moves_every_aligned_field_and_mask(kind) -> None:
    metadata = _frame_metadata()
    provider = _all_providers(metadata)[kind]
    full = provider.binary().predictions

    selected = provider.select(indices=[2, 0, 2])
    masks = selected.binary().predictions

    assert len(selected) == 3
    assert torch.equal(masks.mask, full.mask[[2, 0, 2]])
    assert torch.equal(masks.xyxy, full.xyxy[[2, 0, 2]])
    assert masks.class_id.tolist() == [6, 4, 6]
    assert torch.equal(masks.confidence, full.confidence[[2, 0, 2]])
    assert masks.bboxes_metadata == [{"row": 2}, {"row": 0}, {"row": 2}]
    assert masks.image_metadata is metadata
    assert selected.frame == provider.frame
    assert selected.recipe.output_size_hw == provider.recipe.output_size_hw


@pytest.mark.parametrize("kind", ["grid", "dense", "rle"])
def test_empty_selection_keeps_shapes_and_decodes_nothing(kind, monkeypatch) -> None:
    provider = _all_providers()[kind]

    def no_decode(*args, **kwargs):
        raise AssertionError("an empty selection must not decode")

    monkeypatch.setattr(mask_utils, "decode", no_decode)
    selected = provider.select(mask=[False, False, False])
    masks = selected.binary().predictions

    assert len(selected) == 0
    assert tuple(masks.xyxy.shape) == (0, 4)
    assert tuple(masks.mask.shape) == (0, HEIGHT, WIDTH)
    assert masks.mask.dtype == torch.bool
    assert masks.class_id.dtype == torch.int64
    assert masks.bboxes_metadata == []


@pytest.mark.parametrize("pre_processing", [ODD_CROP, CENTRE_CROP])
def test_empty_grid_approximation_preserves_frame_without_interpolation(
    pre_processing, monkeypatch
) -> None:
    provider = _grid(pre_processing, metadata=_frame_metadata()).select(
        mask=[False, False, False]
    )

    def no_interpolation(*args, **kwargs):
        raise AssertionError("empty masks have no pixels to interpolate")

    monkeypatch.setattr(torch.nn.functional, "interpolate", no_interpolation)
    binary = provider.binary(fidelity="approximate")
    predictions = binary.predictions

    assert binary.fidelity == "approximate"
    assert tuple(predictions.mask.shape) == (0, HEIGHT, WIDTH)
    assert predictions.mask.dtype == torch.bool
    assert predictions.mask.device == provider.view.scores.device
    assert predictions.mask_size == predictions.mask_frame_size == (HEIGHT, WIDTH)
    assert predictions.image_size == (HEIGHT, WIDTH)
    assert predictions.image_metadata == provider.view.detections.image_metadata
    assert predictions.bboxes_metadata == []
    assert tuple(predictions.xyxy.shape) == (0, 4)
    assert provider.binary(fidelity="approximate").predictions is predictions


def test_selected_row_metadata_is_copied_not_shared() -> None:
    provider = DenseMaskProvider(_dense())

    selected = provider.select(indices=[0])
    selected.predictions.bboxes_metadata[0]["row"] = 99

    assert provider.predictions.bboxes_metadata[0] == {"row": 0}


@pytest.mark.parametrize(
    "selection,message",
    [
        ({"indices": [0, 3]}, r"outside \[0, 3\)"),
        ({"indices": [-1]}, r"outside \[0, 3\)"),
        ({"mask": [True]}, "expected 3 prediction rows"),
        ({}, "exactly one"),
        ({"indices": [0], "mask": [True, True, True]}, "exactly one"),
    ],
)
@pytest.mark.parametrize("kind", ["grid", "dense", "rle"])
def test_malformed_selection_is_a_contract_error(kind, selection, message) -> None:
    provider = _all_providers()[kind]

    with pytest.raises(ContractError, match=message):
        provider.select(**selection)


# ---------------------------------------------------------------------------
# Grid recipe: native helper, dtypes, threshold equality, geometry limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "pre_processing", [ODD_CROP, CENTRE_CROP], ids=["crop", "centre"]
)
def test_grid_reference_equals_the_native_helper_in_its_dtype(
    dtype, pre_processing
) -> None:
    provider = _grid(pre_processing, dtype=dtype, grid_hw=(4, 4))

    masks = provider.binary()

    assert masks.fidelity == "reference"
    assert masks.recipe.score_dtype == str(dtype)
    assert masks.recipe.comparison == "score > threshold"
    assert torch.equal(
        masks.predictions.mask, _native_grid_mask(provider, pre_processing)
    )
    assert tuple(masks.predictions.mask.shape) == (3, HEIGHT, WIDTH)


def test_grid_recipe_states_the_crop_paste_and_its_order() -> None:
    provider = _grid(ODD_CROP)

    recipe = provider.recipe
    mask = provider.binary().predictions.mask

    assert recipe.reference == providers.GRID_REFERENCE
    assert recipe.geometry is provider.view.geometry
    assert recipe.operation_order[2] == "threshold: score > 0.0"
    assert recipe.operation_order[3] == "paste into the (9, 11) image at (2, 1)"
    # Outside the static crop (2, 1, 7, 6) nothing is foreground.
    assert not mask[:, :1].any() and not mask[:, 7:].any()
    assert not mask[:, :, :2].any() and not mask[:, :, 9:].any()


def test_threshold_equality_is_background_and_reference_never_previews(
    monkeypatch,
) -> None:
    def no_preview(view):
        raise AssertionError("a reference request ran the nearest preview")

    monkeypatch.setattr(providers, "nearest_full_res", no_preview)
    provider = _grid(scores=torch.zeros((3, 4, 4)), threshold=0.0)

    exact_zero = provider.binary(fidelity="reference")
    below = provider.with_threshold(-1e-6).binary(fidelity="reference")

    assert not exact_zero.predictions.mask.any()
    assert below.predictions.mask[:, 1:7, 2:9].all()
    assert below.recipe.threshold == -1e-6


def test_float16_and_float32_scores_are_separate_recipes() -> None:
    scores = torch.randn((3, 4, 4), generator=torch.Generator().manual_seed(3))
    half = _grid(scores=scores.half())
    single = _grid(scores=scores.half().float())

    assert half.recipe.score_dtype == "torch.float16"
    assert single.recipe.score_dtype == "torch.float32"
    assert torch.equal(
        half.binary().predictions.mask, _native_grid_mask(half, ODD_CROP)
    )
    assert torch.equal(
        single.binary().predictions.mask, _native_grid_mask(single, ODD_CROP)
    )


def test_strided_grid_and_dense_inputs_give_contiguous_control_results() -> None:
    generator = torch.Generator().manual_seed(4)
    transposed = torch.randn((3, 4, 4), generator=generator).transpose(1, 2)
    strided_grid = _grid(scores=transposed)
    control_grid = _grid(scores=transposed.contiguous())
    masks = _analytic_masks()
    strided_mask = masks.transpose(1, 2).contiguous().transpose(1, 2)
    strided_dense = DenseMaskProvider(InstanceDetections(**_rows(), mask=strided_mask))

    selected = strided_dense.select(indices=[2, 0, 2])

    assert not transposed.is_contiguous() and not strided_mask.is_contiguous()
    assert torch.equal(
        strided_grid.binary().predictions.mask, control_grid.binary().predictions.mask
    )
    assert strided_dense.binary().predictions.mask is strided_mask
    assert torch.equal(selected.binary().predictions.mask, masks[[2, 0, 2]])


def test_origin_crop_and_non_square_geometry_use_current_native_rules() -> None:
    partial_crop_at_origin = _pre_processing(
        original_hw=(HEIGHT, WIDTH),
        resized_hw=(6, 7),
        inference_hw=(16, 16),
        padding_ltrb=(0, 1, 2, 1),
        crop_xywh=(0, 0, 7, 6),
    )
    non_square = _pre_processing(
        original_hw=(HEIGHT, WIDTH),
        resized_hw=(HEIGHT, WIDTH),
        inference_hw=(8, 8),
        padding_ltrb=(0, 0, 0, 0),
    )._replace(nonsquare_intermediate_size=ImageDimensions(8, 10))

    origin = MaskGridGeometry.from_pre_processing(
        partial_crop_at_origin, grid_size_hw=(4, 4)
    )
    non_square_geometry = MaskGridGeometry.from_pre_processing(
        non_square, grid_size_hw=(4, 4)
    )
    assert origin.static_crop_xywh == (0, 0, 7, 6)
    assert origin.original_size_hw == (HEIGHT, WIDTH)
    assert non_square_geometry.inference_size_hw == (8, 10)


def test_clamped_low_resolution_logits_do_not_reproduce_unclamped_masks() -> None:
    # SAM2 returns low-resolution logits clamped to [-32, 32], but thresholds
    # its full masks from unclamped logits. Interpolating the clamped values
    # moves the boundary: the grid recipe is exact only for stored scores.
    pre_processing = _pre_processing(
        original_hw=(8, 8),
        resized_hw=(8, 8),
        inference_hw=(2, 2),
        padding_ltrb=(0,) * 4,
    )
    unclamped = torch.tensor([[[-100.0, 40.0], [-100.0, 40.0]]])
    native = _grid(pre_processing, scores=unclamped.expand(3, 2, 2).clone())
    clamped = _grid(
        pre_processing, scores=unclamped.clamp(-32, 32).expand(3, 2, 2).clone()
    )

    native_mask = native.binary().predictions.mask
    clamped_mask = clamped.binary().predictions.mask

    assert torch.equal(clamped_mask, _native_grid_mask(clamped, pre_processing))
    assert int(clamped_mask[0].sum()) > int(native_mask[0].sum())


# ---------------------------------------------------------------------------
# Fidelity labels, capabilities and refusals
# ---------------------------------------------------------------------------


def test_grid_approximation_is_labelled_and_differs_from_the_reference() -> None:
    provider = _grid(ODD_CROP, grid_hw=(4, 4), seed=1)

    reference = provider.binary(fidelity="reference")
    approximate = provider.binary(fidelity="approximate")

    assert reference.fidelity == "reference"
    assert approximate.fidelity == "approximate"
    assert approximate.recipe is reference.recipe
    assert approximate.method.startswith("nearest_full_res")
    assert torch.equal(approximate.predictions.mask, nearest_full_res(provider.view))
    assert not torch.equal(approximate.predictions.mask, reference.predictions.mask)
    assert provider.binary(fidelity="approximate") is not approximate
    assert (
        provider.binary(fidelity="approximate").predictions is approximate.predictions
    )


@pytest.mark.parametrize("kind", ["dense", "rle"])
def test_exact_only_providers_answer_approximate_requests_with_their_reference(
    kind,
) -> None:
    provider = _all_providers()[kind]

    masks = provider.binary(fidelity="approximate")

    assert masks.fidelity == "reference"
    assert masks.predictions is provider.binary().predictions
    assert provider.capabilities == frozenset({BINARY})


@pytest.mark.parametrize("kind", ["dense", "rle"])
def test_dense_and_rle_refuse_scores_and_thresholds(kind) -> None:
    provider = _all_providers()[kind]

    with pytest.raises(UnsupportedMaskRequest, match="score grid"):
        provider.score_grid()
    with pytest.raises(UnsupportedMaskRequest, match="without scores"):
        provider.with_threshold(0.5)


def test_grid_capabilities_depend_on_its_scores() -> None:
    logits = _grid()
    binary_view = SegmentationView(
        detections=Detections(**_rows()),
        scores=torch.zeros((3, 4, 4), dtype=torch.bool),
        score_type="binary",
        threshold=0.0,
        geometry=logits.view.geometry,
    )
    binary = GridMaskProvider(binary_view)

    assert logits.capabilities == frozenset({BINARY, APPROXIMATE_BINARY, SCORE_GRID})
    assert logits.score_grid()[0] is logits.view.scores
    assert SCORE_GRID not in binary.capabilities
    with pytest.raises(UnsupportedMaskRequest, match="binary cells"):
        binary.score_grid()


def test_unknown_fidelity_is_rejected() -> None:
    with pytest.raises(ContractError, match="fidelity must be one of"):
        DenseMaskProvider(_dense()).binary(fidelity="exact")


def test_dense_fallback_view_is_a_dense_provider_not_a_grid() -> None:
    predictions = _dense()
    view = SegmentationView.from_dense(predictions)

    provider = mask_provider_of(view)

    assert isinstance(provider, DenseMaskProvider)
    assert provider.binary().predictions is predictions
    with pytest.raises(ContractError, match="dense-fallback"):
        GridMaskProvider(view)


@pytest.mark.parametrize(
    "build,message",
    [
        (lambda: DenseMaskProvider(_rle()), "use RLEMaskProvider"),
        (lambda: RLEMaskProvider(_dense()), "use DenseMaskProvider"),
        (lambda: GridMaskProvider(_dense()), "wraps a SegmentationView"),
        (lambda: mask_provider_of(Detections(**_rows())), "got Detections"),
        (
            lambda: DenseMaskProvider(
                InstanceDetections(
                    **_rows(), mask=torch.zeros((2, 4, 4), dtype=torch.bool)
                )
            ),
            "mask",
        ),
    ],
)
def test_wrong_representations_are_rejected(build, message) -> None:
    with pytest.raises(ContractError, match=message):
        build()


@pytest.mark.parametrize(
    "counts,message",
    [
        (b"zz", "not a compressed COCO count"),
        (b"", r"cover 0 pixels; image_size \(9, 11\) needs 99"),
        (b"0", r"cover 0 pixels"),
        (b"P", "end inside a run length"),
    ],
)
def test_malformed_rle_counts_fail_with_the_row(counts, message) -> None:
    predictions = _rle()
    predictions.mask.masks[1] = counts

    provider = RLEMaskProvider(predictions)

    with pytest.raises(ContractError, match=f"RLE row 1: .*{message}"):
        provider.binary()


def test_truncated_rle_that_pycocotools_accepts_is_rejected() -> None:
    predictions = _rle()
    predictions.mask.masks[0] = predictions.mask.masks[0][:-2]

    assert mask_utils.decode(predictions.mask.to_coco_rle_masks()).shape[:2] == (
        HEIGHT,
        WIDTH,
    )
    with pytest.raises(ContractError, match="RLE row 0: counts cover"):
        RLEMaskProvider(predictions).binary()


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["grid", "dense", "rle"])
def test_frame_comes_from_metadata_and_survives_materialization(kind) -> None:
    metadata = _frame_metadata()
    provider = _all_providers(metadata)[kind]

    frame = provider.frame

    assert frame.size_hw == (HEIGHT, WIDTH)
    assert frame.image_id == "frame"
    assert provider.binary().frame == frame


@pytest.mark.parametrize("kind", ["grid", "dense", "rle"])
def test_frame_of_another_size_or_missing_frame_is_an_error(kind) -> None:
    other = _all_providers(_frame_metadata(height=HEIGHT + 1))[kind]
    missing = _all_providers()[kind]

    with pytest.raises(ContractError, match=r"masks are \(9, 11\)"):
        other.frame
    with pytest.raises(ContractError, match="no image_metadata"):
        missing.frame


# ---------------------------------------------------------------------------
# Caches: once per provider, failures retry, no stale reuse
# ---------------------------------------------------------------------------


def _counting(monkeypatch, name, *, fail_first=False, delay=None):
    module = rle_module if name == "decode_rle_tensor" else providers
    original = getattr(module, name)
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(1)
        if delay is not None:
            delay.wait(timeout=5)
        if fail_first and len(calls) == 1:
            raise RuntimeError("transient failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapped)

    return calls


@pytest.mark.parametrize(
    "kind,function,fidelity",
    [
        ("rle", "decode_rle_tensor", "reference"),
        ("grid", "nearest_full_res", "approximate"),
    ],
)
def test_concurrent_requests_share_one_computation(
    kind, function, fidelity, monkeypatch
) -> None:
    release = threading.Event()
    calls = _counting(monkeypatch, function, delay=release)
    provider = _all_providers()[kind]
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(provider.binary(fidelity=fidelity))
        )
        for _ in range(4)
    ]

    for thread in threads:
        thread.start()
    time.sleep(0.05)  # let every thread reach the request before it completes
    release.set()
    for thread in threads:
        thread.join(timeout=10)

    assert len(calls) == 1
    assert len({id(masks.predictions) for masks in results}) == 1


@pytest.mark.parametrize(
    "kind,function,fidelity",
    [
        ("rle", "decode_rle_tensor", "reference"),
        ("grid", "nearest_full_res", "approximate"),
    ],
)
def test_a_failure_publishes_nothing_and_the_next_request_retries(
    kind, function, fidelity, monkeypatch
) -> None:
    calls = _counting(monkeypatch, function, fail_first=True)
    provider = _all_providers()[kind]

    with pytest.raises(RuntimeError, match="transient"):
        provider.binary(fidelity=fidelity)
    masks = provider.binary(fidelity=fidelity)

    assert len(calls) == 2
    assert tuple(masks.predictions.mask.shape) == (3, HEIGHT, WIDTH)
    assert provider.binary(fidelity=fidelity).predictions is masks.predictions


@pytest.mark.parametrize("kind", ["grid", "rle"])
def test_selection_and_threshold_changes_never_reuse_a_dense_cache(kind) -> None:
    provider = _all_providers()[kind]
    full = provider.binary().predictions.mask

    reordered = provider.select(indices=[1, 0])

    assert torch.equal(reordered.binary().predictions.mask, full[[1, 0]])
    if kind == "grid":
        stricter = provider.with_threshold(1.0)
        assert not stricter.view.is_full_res_materialized()
        assert torch.equal(
            stricter.binary().predictions.mask, _native_grid_mask(stricter, ODD_CROP)
        )
        assert int(stricter.binary().predictions.mask.sum()) < int(full.sum())


# ---------------------------------------------------------------------------
# Recording: providers are in-process helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["grid", "dense", "rle"])
def test_providers_have_no_recording_codec(kind) -> None:
    registry = CodecRegistry(create_catalogue().codecs.values())
    provider = _all_providers()[kind]

    with pytest.raises(RecordingCodecError, match="no recording codec"):
        registry.encoder(MemoryBlobs(), location="masks").encode(provider)
