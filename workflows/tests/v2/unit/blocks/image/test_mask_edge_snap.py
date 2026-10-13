"""``v2/mask_edge_snap@v1``: V1 parity, representations, rows and validation.

V1 parity runs the latest V1 implementation (``v1_tensor.refine_masks``) on the
same synthetic scenes. Scenes are drawn shapes with seeded noise; each mask is
offset from its shape, so snapping has to move outlines.
"""

import copy
import dataclasses

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch
from roboflow_workflows.core_steps.classical_cv.mask_edge_snap.v1_tensor import (
    refine_masks,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.image.mask_edge_snap import MaskEdgeSnapBlock
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.predictions import INSTANCE_SEGMENTATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    ResolvedParameterError,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

DEFAULTS = {
    "pixel_tolerance": 15,
    "sigma": 1.0,
    "min_contour_area": 50.0,
    "dilation_iterations": 2,
    "boundary_band_width": 15,
    "adaptive_window_size": 41,
}
SIZE_HW = (120, 160)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def _noisy(canvas: np.ndarray, seed: int = 0) -> np.ndarray:
    noise = np.random.default_rng(seed).integers(-6, 7, size=canvas.shape)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy


def _rectangle(xyxy, size_hw=SIZE_HW) -> np.ndarray:
    x0, y0, x1, y1 = xyxy
    mask = np.zeros(size_hw, dtype=bool)
    mask[y0:y1, x0:x1] = True

    return mask


def _ellipse(center, axes, size_hw=SIZE_HW) -> np.ndarray:
    canvas = np.zeros(size_hw, dtype=np.uint8)
    cv2.ellipse(canvas, center, axes, 0, 0, 360, color=1, thickness=-1)

    return canvas.astype(bool)


def _two_shapes() -> tuple:
    # Rectangle and ellipse; masks shifted / shrunk by about four pixels.
    gray = np.full(SIZE_HW, 30, dtype=np.uint8)
    gray[_rectangle((30, 25, 80, 85))] = 200
    gray[_ellipse((120, 60), (22, 30))] = 140
    masks = [_rectangle((34, 29, 84, 81)), _ellipse((120, 60), (18, 26))]

    return _noisy(gray), masks


def _border_shape() -> tuple:
    # Shape and mask both touch the top-left image border.
    gray = np.full(SIZE_HW, 40, dtype=np.uint8)
    gray[_rectangle((0, 0, 60, 50))] = 220
    masks = [_rectangle((0, 0, 55, 54))]

    return _noisy(gray, seed=1), masks


def _overlapping() -> tuple:
    # Three masks; two of them overlap and share one image edge.
    gray = np.full(SIZE_HW, 20, dtype=np.uint8)
    gray[_rectangle((20, 20, 70, 70))] = 180
    gray[_rectangle((90, 30, 140, 100))] = 120
    masks = [
        _rectangle((23, 17, 73, 67)),
        _rectangle((40, 40, 75, 75)),
        _rectangle((94, 34, 136, 96)),
    ]

    return _noisy(gray, seed=2), masks


def _color(gray: np.ndarray) -> np.ndarray:
    # RGB with different channels, so the gray conversion matters.
    rgb = np.stack([gray, (gray * 0.5).astype(np.uint8), 255 - gray], axis=-1)

    return rgb


def _image(pixels: np.ndarray, *, device: str = "cpu") -> ImageData:
    # RGB (H, W, 3) or gray (H, W) uint8 -> ImageData on device.
    tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    tensor = tensor.permute(2, 0, 1) if tensor.ndim == 3 else tensor[None]
    image = ImageData.from_tensor(tensor.contiguous().to(device), image_id="scene")

    return image


def _boxes(masks) -> list:
    boxes = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])

    return boxes


def _segmentation(
    masks, *, image: ImageData, family: str = "dense", boxes=None
) -> InstanceDetections:
    count = len(masks)
    device = image.device
    dense = (
        torch.from_numpy(np.stack(masks))
        if count
        else torch.zeros((0, *image.size_hw), dtype=torch.bool)
    )
    if family == "rle":
        encoded = [
            mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))["counts"]
            for m in masks
        ]
        mask = InstancesRLEMasks(image_size=image.size_hw, masks=encoded)
    else:
        mask = dense.to(device)
    xyxy = torch.tensor(
        boxes if boxes is not None else _boxes(masks), dtype=torch.float32
    ).reshape(-1, 4)
    predictions = InstanceDetections(
        xyxy=xyxy.to(device),
        class_id=torch.arange(count, dtype=torch.long, device=device),
        confidence=torch.linspace(0.5, 0.9, count, device=device),
        mask=mask,
        image_metadata={**image.prediction_metadata(), "class_names": {0: "a"}},
        bboxes_metadata=[
            {
                "detection_id": f"row-{i}",
                "_host_xyxy": [0, 0, 1, 1],
                "_host_class_id": -1,
                "_host_confidence": 0.0,
            }
            for i in range(count)
        ],
    )

    return predictions


def _grid_view(masks, *, image: ImageData, stride: int = 4) -> SegmentationView:
    # Grid-backed view: logits on a grid `stride` times coarser than the image.
    reference = _segmentation(masks, image=image)
    cells = torch.nn.functional.max_pool2d(
        reference.mask.float().unsqueeze(1), stride
    ).squeeze(1)
    height, width = image.size_hw
    geometry = MaskGridGeometry(
        grid_size_hw=(height // stride, width // stride),
        inference_size_hw=(height, width),
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(1.0, 1.0),
        crop_size_hw=(height, width),
        original_size_hw=(height, width),
        static_crop_xywh=(0, 0, width, height),
    )
    detections = Detections(
        xyxy=reference.xyxy,
        class_id=reference.class_id,
        confidence=reference.confidence,
        image_metadata=reference.image_metadata,
        bboxes_metadata=reference.bboxes_metadata,
    )
    view = SegmentationView(
        detections=detections,
        scores=cells * 16.0 - 8.0,
        score_type="logits",
        threshold=0.0,
        geometry=geometry,
    )

    return view


def _snap(image, segmentation, **options) -> dict:
    result = MaskEdgeSnapBlock().run(
        image=image, segmentation=segmentation, **{**DEFAULTS, **options}
    )

    return result


def _v1(pixels: np.ndarray, segmentation: InstanceDetections, **options) -> tuple:
    # Latest V1 on the same pixels: V1 images are BGR NumPy arrays.
    bgr = pixels if pixels.ndim == 2 else pixels[:, :, ::-1]
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="scene"),
        numpy_image=np.ascontiguousarray(bgr),
    )
    refined, edges = refine_masks(
        image=image, segmentation=segmentation, **{**DEFAULTS, **options}
    )

    return refined, edges


def _dense(predictions: InstanceDetections) -> np.ndarray:
    mask = predictions.mask
    if isinstance(mask, InstancesRLEMasks):
        if not mask.masks:
            return np.zeros((0, *mask.image_size), dtype=bool)
        decoded = mask_utils.decode(
            [{"size": list(mask.image_size), "counts": c} for c in mask.masks]
        )
        return np.moveaxis(decoded, -1, 0).astype(bool)

    return mask.cpu().numpy().astype(bool)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_contract_defaults_and_output_sources():
    params = spec_of(MaskEdgeSnapBlock).validate_params(
        {"image": "$inputs.image", "segmentation": "$steps.model.predictions"},
        step_path=("snap",),
    )
    assert {name: getattr(params, name) for name in DEFAULTS} == DEFAULTS
    spec = spec_of(MaskEdgeSnapBlock)
    assert spec.outputs["refined_segmentation"].source == "segmentation"
    assert spec.outputs["edges"].source == "image"


@pytest.mark.parametrize(
    "name,value",
    [
        ("pixel_tolerance", 0),
        ("pixel_tolerance", 101),
        ("pixel_tolerance", 2.5),
        ("pixel_tolerance", True),
        ("sigma", 0.0),
        ("sigma", 10.5),
        ("sigma", float("nan")),
        ("min_contour_area", -1.0),
        ("min_contour_area", 10001),
        ("dilation_iterations", -1),
        ("dilation_iterations", 21),
        ("boundary_band_width", 0),
        ("boundary_band_width", 101),
        ("adaptive_window_size", 2),
        ("adaptive_window_size", 202),
    ],
)
def test_literal_out_of_range_is_rejected(name, value):
    with pytest.raises(ParamsValidationError):
        spec_of(MaskEdgeSnapBlock).validate_params(
            {"image": "$inputs.image", "segmentation": "$inputs.seg", name: value},
            step_path=("snap",),
        )


def test_integer_literals_are_accepted_for_float_parameters():
    params = spec_of(MaskEdgeSnapBlock).validate_params(
        {
            "image": "$inputs.image",
            "segmentation": "$inputs.seg",
            "sigma": 2,
            "min_contour_area": 0,
        },
        step_path=("snap",),
    )
    assert (params.sigma, params.min_contour_area) == (2.0, 0.0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("pixel_tolerance", 0),
        ("sigma", 20.0),
        ("sigma", float("nan")),
        ("min_contour_area", -0.5),
        ("dilation_iterations", 25),
        ("boundary_band_width", 0),
        ("adaptive_window_size", 1),
        ("adaptive_window_size", True),
        ("adaptive_window_size", "41"),
    ],
)
def test_resolved_values_out_of_range_raise_with_field_path(name, value):
    # run() is an engine callback; validate selected payloads via BlockSpec.
    gray, masks = _two_shapes()
    image = _image(gray)
    spec = spec_of(MaskEdgeSnapBlock)
    params = spec.validate_params(
        {
            "image": "$inputs.image",
            "segmentation": "$inputs.seg",
            name: "$inputs.option",
        },
        step_path=("snap",),
    )
    arguments = {
        "image": image,
        "segmentation": _segmentation(masks, image=image),
        **DEFAULTS,
        name: value,
    }
    with pytest.raises(ResolvedParameterError) as error:
        spec.validate_resolved_arguments(params, arguments)
    assert error.value.field_path == (name,)


# ---------------------------------------------------------------------------
# V1 parity
# ---------------------------------------------------------------------------

PARITY_CASES = {
    "defaults_color": (_two_shapes, True, {}),
    "grayscale_one_channel": (_two_shapes, False, {}),
    # The default min_contour_area drops this scene's thin open edge (V1 too).
    "border": (_border_shape, True, {"min_contour_area": 0.0}),
    "overlapping_masks": (_overlapping, True, {}),
    "dilation_zero": (_overlapping, True, {"dilation_iterations": 0}),
    "tight_tolerance_even_window": (
        _two_shapes,
        True,
        {
            "pixel_tolerance": 3,
            "boundary_band_width": 5,
            "adaptive_window_size": 20,
            "sigma": 0.3,
            "min_contour_area": 0.0,
        },
    ),
}


@pytest.mark.parametrize("family", ["dense", "rle"])
@pytest.mark.parametrize("case", sorted(PARITY_CASES))
def test_v1_parity_of_masks_and_edges(case, family):
    scene, color, options = PARITY_CASES[case]
    gray, masks = scene()
    pixels = _color(gray) if color else gray
    image = _image(pixels)

    result = _snap(image, _segmentation(masks, image=image, family=family), **options)
    v1_refined, v1_edges = _v1(
        pixels, _segmentation(masks, image=image, family=family), **options
    )

    refined = _dense(result["refined_segmentation"])
    assert np.array_equal(refined, v1_refined.mask.numpy())
    assert np.array_equal(_dense(result["edges"]), v1_edges.mask.numpy())
    assert torch.equal(result["edges"].xyxy, v1_edges.xyxy)
    assert torch.equal(result["edges"].class_id, v1_edges.class_id)
    assert torch.equal(result["edges"].confidence, v1_edges.confidence)
    for key in ("class_names", "prediction_type"):
        assert result["edges"].image_metadata[key] == v1_edges.image_metadata[key]
    # The case is meaningful: snapping moved at least one outline.
    assert not np.array_equal(refined, np.stack(masks))


def test_v1_parity_without_rows_searches_the_whole_image():
    gray, _ = _two_shapes()
    image = _image(_color(gray))
    empty = _segmentation([], image=image)

    result = _snap(image, empty)
    _, v1_edges = _v1(_color(gray), _segmentation([], image=image))

    assert result["refined_segmentation"] is empty
    assert np.array_equal(_dense(result["edges"]), v1_edges.mask.numpy())
    assert torch.equal(result["edges"].xyxy, v1_edges.xyxy)


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------


def test_rle_input_stays_rle_and_matches_dense():
    gray, masks = _two_shapes()
    image = _image(_color(gray))

    dense = _snap(image, _segmentation(masks, image=image))["refined_segmentation"]
    rle = _snap(image, _segmentation(masks, image=image, family="rle"))[
        "refined_segmentation"
    ]

    assert isinstance(rle.mask, InstancesRLEMasks)
    assert tuple(rle.mask.image_size) == SIZE_HW
    assert np.array_equal(_dense(rle), _dense(dense))
    assert torch.equal(rle.xyxy, dense.xyxy)
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(rle)


def test_grid_view_uses_its_full_resolution_masks():
    gray, masks = _two_shapes()
    image = _image(_color(gray))
    view = _grid_view(masks, image=image)
    full_res = view.full_res()

    from_view = _snap(image, view)["refined_segmentation"]
    from_dense = _snap(image, full_res)["refined_segmentation"]

    assert isinstance(from_view, InstanceDetections)
    assert isinstance(from_view.mask, torch.Tensor)
    assert np.array_equal(_dense(from_view), _dense(from_dense))
    assert torch.equal(from_view.xyxy, from_dense.xyxy)
    assert [row["detection_id"] for row in from_view.bboxes_metadata] == [
        "row-0",
        "row-1",
    ]


def test_dense_fallback_view_matches_its_dense_predictions():
    gray, masks = _two_shapes()
    image = _image(_color(gray))
    dense = _segmentation(masks, image=image)

    from_view = _snap(image, SegmentationView.from_dense(dense))
    from_dense = _snap(image, dense)

    for name in ("refined_segmentation", "edges"):
        assert np.array_equal(_dense(from_view[name]), _dense(from_dense[name]))


def test_empty_grid_view_returns_empty_dense_predictions():
    gray, masks = _two_shapes()
    image = _image(_color(gray))
    view = _grid_view(masks, image=image).select(torch.zeros(0, dtype=torch.long))

    refined = _snap(image, view)["refined_segmentation"]

    assert len(refined) == 0
    assert tuple(refined.mask.shape) == (0, *SIZE_HW)


# ---------------------------------------------------------------------------
# Rows, boxes and frames
# ---------------------------------------------------------------------------


def test_boxes_follow_refined_masks_and_rows_keep_their_order():
    gray, masks = _overlapping()
    image = _image(_color(gray))
    segmentation = _segmentation(masks, image=image)

    refined = _snap(image, segmentation)["refined_segmentation"]

    assert torch.equal(
        refined.xyxy, torch.tensor(_boxes(_dense(refined)), dtype=torch.float32)
    )
    assert not torch.equal(refined.xyxy, segmentation.xyxy)
    assert torch.equal(refined.class_id, segmentation.class_id)
    assert torch.equal(refined.confidence, segmentation.confidence)
    assert refined.image_metadata == segmentation.image_metadata


def test_border_mask_snaps_to_the_shape_touching_the_border():
    gray, masks = _border_shape()
    image = _image(_color(gray))

    refined = _snap(image, _segmentation(masks, image=image), min_contour_area=0.0)

    assert refined["refined_segmentation"].xyxy.tolist() == [[0.0, 0.0, 60.0, 50.0]]


def test_row_with_empty_mask_keeps_its_box_and_empty_mask():
    gray, masks = _two_shapes()
    image = _image(_color(gray))
    empty = np.zeros(SIZE_HW, dtype=bool)
    boxes = [[1, 2, 3, 4]] + _boxes(masks)
    segmentation = _segmentation([empty, *masks], image=image, boxes=boxes)

    refined = _snap(image, segmentation)["refined_segmentation"]

    assert refined.xyxy[0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert not _dense(refined)[0].any()
    assert _dense(refined)[1:].any(axis=(1, 2)).all()


def test_host_mirrors_are_stripped_and_other_row_metadata_kept():
    gray, masks = _two_shapes()
    image = _image(_color(gray))

    refined = _snap(image, _segmentation(masks, image=image))["refined_segmentation"]

    assert refined.bboxes_metadata == [
        {"detection_id": "row-0"},
        {"detection_id": "row-1"},
    ]


def test_edges_row_is_on_the_image_frame():
    gray, masks = _two_shapes()
    image = _image(_color(gray))

    edges = _snap(image, _segmentation(masks, image=image))["edges"]

    assert len(edges) == 1
    assert edges.image_metadata["class_names"] == {0: "edges"}
    assert frame_of(edges) == FrameRef.from_image(image)
    assert edges.bboxes_metadata[0]["detection_id"]
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(edges)


def test_flat_image_gives_no_edges_row():
    image = _image(np.full((*SIZE_HW, 3), 90, dtype=np.uint8))
    _, masks = _two_shapes()

    result = _snap(image, _segmentation(masks, image=image))

    assert len(result["edges"]) == 0
    assert tuple(result["edges"].mask.shape) == (0, *SIZE_HW)
    assert np.array_equal(_dense(result["refined_segmentation"]), np.stack(masks))


@pytest.mark.parametrize("device", DEVICES)
def test_outputs_stay_on_the_inputs_device(device):
    gray, masks = _two_shapes()
    image = _image(_color(gray), device=device)

    result = _snap(image, _segmentation(masks, image=image))

    for name in ("refined_segmentation", "edges"):
        assert result[name].xyxy.device.type == device
        assert result[name].mask.device.type == device


# ---------------------------------------------------------------------------
# Rejections and immutability
# ---------------------------------------------------------------------------


def test_predictions_without_masks_are_rejected():
    gray, masks = _two_shapes()
    image = _image(_color(gray))
    no_masks = dataclasses.replace(_segmentation(masks, image=image), mask=None)

    with pytest.raises(ValueError, match="needs instance masks"):
        _snap(image, no_masks)


def test_masks_of_another_size_are_rejected():
    gray, masks = _two_shapes()
    image = _image(_color(gray))
    small = _image(_color(gray)[:60, :80])
    segmentation = _segmentation([masks[0][:60, :80]], image=small)

    with pytest.raises(ValueError, match="size"):
        _snap(image, segmentation)


@pytest.mark.parametrize("family", ["dense", "rle"])
def test_inputs_are_not_mutated(family):
    gray, masks = _two_shapes()
    image = _image(_color(gray))
    segmentation = _segmentation(masks, image=image, family=family)
    pixels = image.tensor_image.clone()
    xyxy = segmentation.xyxy.clone()
    mask = copy.deepcopy(segmentation.mask)
    rows = copy.deepcopy(segmentation.bboxes_metadata)
    metadata = copy.deepcopy(segmentation.image_metadata)

    _snap(image, segmentation)

    assert torch.equal(image.tensor_image, pixels)
    assert torch.equal(segmentation.xyxy, xyxy)
    if family == "rle":
        assert segmentation.mask.masks == mask.masks
    else:
        assert torch.equal(segmentation.mask, mask)
    assert segmentation.bboxes_metadata == rows
    assert segmentation.image_metadata == metadata


@pytest.mark.parametrize(
    "name,low,high",
    [
        ("pixel_tolerance", 1, 100),
        ("sigma", 0.01, 10.0),
        ("min_contour_area", 0.0, 10000.0),
        ("dilation_iterations", 0, 20),
        ("boundary_band_width", 1, 100),
        ("adaptive_window_size", 3, 201),
    ],
)
@pytest.mark.parametrize("boundary", ["low", "high"])
def test_literal_and_resolved_boundaries_are_accepted(name, low, high, boundary):
    value = low if boundary == "low" else high
    spec = spec_of(MaskEdgeSnapBlock)
    definition = {"image": "$inputs.image", "segmentation": "$inputs.seg"}
    literal = spec.validate_params({**definition, name: value}, step_path=("snap",))
    assert getattr(literal, name) == value
    selected = spec.validate_params(
        {**definition, name: "$inputs.option"}, step_path=("snap",)
    )
    gray, masks = _two_shapes()
    image = _image(gray)
    spec.validate_resolved_arguments(
        selected,
        {
            "image": image,
            "segmentation": _segmentation(masks, image=image),
            **DEFAULTS,
            name: value,
        },
    )


def test_unchanged_sam_style_mask_reboxes_from_inclusive_to_exclusive_bounds():
    # Q5 uses inclusive max bounds. Q2 deliberately recomputes max+1, even
    # when there are no edges and the mask itself does not change.
    image = _image(np.full(SIZE_HW, 90, dtype=np.uint8))
    mask = _rectangle((20, 30, 60, 70))
    segmentation = _segmentation([mask], image=image, boxes=[[20, 30, 59, 69]])
    refined = _snap(image, segmentation)["refined_segmentation"]
    assert np.array_equal(_dense(refined), np.stack([mask]))
    assert refined.xyxy.tolist() == [[20.0, 30.0, 60.0, 70.0]]


def test_same_size_wrong_frame_currently_passes_through_its_frame():
    # Evidence of the current compatibility gap, pending a scoped group
    # decision on typed-frame and legacy-metadata validation.
    gray, masks = _two_shapes()
    image = _image(gray)
    other = ImageData.from_tensor(image.tensor_image.clone(), image_id="other")
    segmentation = _segmentation(masks, image=other)
    result = _snap(image, segmentation)
    assert frame_of(result["refined_segmentation"]) == FrameRef.from_image(other)
    assert frame_of(result["edges"]) == FrameRef.from_image(image)
