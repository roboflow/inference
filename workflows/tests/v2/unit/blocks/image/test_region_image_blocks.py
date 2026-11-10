"""Unit tests of the ported region image blocks, called directly.

Expected values are worked out by hand from the V1 rules quoted in each block's
docstring (rounding, clipping, grid arithmetic), not taken from a V1 run.
"""

import math
from dataclasses import replace
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image import (
    AbsoluteStaticCropBlock,
    DynamicCropBlock,
    ImagePreprocessingBlock,
    ImageSlicerBlock,
    RelativeStaticCropBlock,
)
from roboflow_workflows.v2.blocks.image.dynamic_crop.v1 import parse_rgb
from roboflow_workflows.v2.blocks.image.image_slicer.v2 import slice_rectangles
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.row_identity import (
    RowRef,
    row_record,
    with_row_identity,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)


def image(height=80, width=100, *, fill=None):
    pixels = (
        torch.arange(3 * height * width, dtype=torch.int64).reshape(3, height, width)
        % 251
    ).to(torch.uint8)
    if fill is not None:
        pixels[:] = torch.tensor(fill, dtype=torch.uint8).reshape(3, 1, 1)

    return ImageData.from_tensor(pixels, image_id="root", video_metadata=VIDEO)


def detections_on(source, xyxy, **extra):
    fields = dict(
        xyxy=torch.tensor(xyxy, dtype=torch.float32),
        class_id=torch.arange(len(xyxy)),
        confidence=torch.full((len(xyxy),), 0.9),
        image_metadata={
            **source.prediction_metadata(),
            "class_names": {0: "a", 1: "b"},
        },
        bboxes_metadata=[{"row": row} for row in range(len(xyxy))],
    )
    fields.update(extra)
    carrier = InstanceDetections if "mask" in fields else Detections

    return carrier(**fields)


# --- static crops ------------------------------------------------------------


@pytest.mark.parametrize(
    ("x_center", "width", "expected_x"),
    [(10, 5, (8, 13)), (11, 5, (8, 13)), (13, 5, (10, 15)), (50, 4, (48, 52))],
)
def test_absolute_static_crop_rounds_half_to_even(x_center, width, expected_x):
    # 10 - 2.5 = 7.5 -> 8; 11 - 2.5 = 8.5 -> 8; 13 - 2.5 = 10.5 -> 10.
    crop = AbsoluteStaticCropBlock().run(
        image=image(), x_center=x_center, y_center=40, width=width, height=10
    )["crops"]

    assert crop.parent.offset_xy == (float(expected_x[0]), 35.0)
    assert crop.width == expected_x[1] - expected_x[0]


def test_absolute_static_crop_clips_and_keeps_source_and_time():
    source = image()

    crop = AbsoluteStaticCropBlock().run(
        image=source, x_center=95, y_center=5, width=20, height=20
    )["crops"]

    # x 85..105 -> 85..100; y -5..15 -> 0..15.
    assert crop.size_hw == (15, 15)
    assert crop.root.offset_xy == (85.0, 0.0) and crop.root.frame_id == "root"
    assert crop.video_metadata is VIDEO
    assert torch.equal(crop.tensor_image, source.tensor_image[:, 0:15, 85:100])


def test_absolute_static_crop_outside_the_image_has_no_crop():
    result = AbsoluteStaticCropBlock().run(
        image=image(), x_center=300, y_center=40, width=10, height=10
    )

    assert result == {"crops": None}


def test_relative_static_crop_clips_where_v1_wrapped():
    # W = 100: centre 10, width 50 -> x -15..35 -> clipped 0..35.
    crop = RelativeStaticCropBlock().run(
        image=image(), x_center=0.1, y_center=0.5, width=0.5, height=0.5
    )["crops"]

    assert crop.parent.offset_xy == (0.0, 20.0)
    assert crop.size_hw == (40, 35)


def test_relative_static_crop_of_zero_width_has_no_crop():
    result = RelativeStaticCropBlock().run(
        image=image(), x_center=0.5, y_center=0.5, width=0.0, height=0.5
    )

    assert result["crops"] is None


# --- slicer ------------------------------------------------------------------


def test_slice_grid_example_from_the_docstring():
    rectangles = slice_rectangles(
        (600, 1000), slice_wh=(640, 640), overlap_ratio_wh=(0.2, 0.2)
    )

    assert rectangles == [(0, 0, 640, 600), (360, 0, 1000, 600)]


def test_slice_grid_rows_run_y_outer_and_border_slices_move_inward():
    # 100 wide, 40-pixel slices, ratio 0.5: stride 20 -> 0, 20, 40, 60, 80 -> 60.
    rectangles = slice_rectangles(
        (40, 100), slice_wh=(40, 30), overlap_ratio_wh=(0.5, 0.0)
    )

    xs = sorted({r[0] for r in rectangles})
    assert xs == [0, 20, 40, 60]
    assert [r[1] for r in rectangles] == [0] * 4 + [10] * 4
    assert all(r[2] - r[0] == 40 and r[3] - r[1] == 30 for r in rectangles)


def test_slicer_returns_dense_indices_and_crop_provenance():
    slices = ImageSlicerBlock().run(
        image=image(),
        slice_width=60,
        slice_height=80,
        overlap_ratio_width=0.0,
        overlap_ratio_height=0.0,
    )["slices"]

    assert slices.indices == ((0,), (1,))
    assert [s.root.offset_xy for s in slices] == [(0.0, 0.0), (40.0, 0.0)]


# --- preprocessing -----------------------------------------------------------


def preprocess(source, **options):
    defaults = dict(
        task_type="resize",
        width=640,
        height=640,
        rotation_degrees=90,
        flip_type="vertical",
    )
    defaults.update(options)

    return ImagePreprocessingBlock().run(image=source, **defaults)["image"]


def test_resize_with_one_side_keeps_aspect_ratio_and_maps_back():
    resized = preprocess(image(60, 100), width=50, height=None)

    assert resized.size_hw == (30, 50)
    assert resized.parent.scale_xy == (2.0, 2.0)
    assert resized.root.frame_id == "root"


def test_resize_without_sizes_returns_the_input():
    source = image()

    assert preprocess(source, width=None, height=None) is source


def test_resize_to_zero_pixels_is_an_error():
    with pytest.raises(ValueError, match="aspect"):
        preprocess(image(1, 100), width=50, height=None)


def test_rotate_90_is_counter_clockwise_and_exact():
    source = image(4, 6, fill=(0, 0, 0))
    pixels = source.tensor_image.clone()
    pixels[:, 0, 5] = 255  # top-right
    rotated = preprocess(source.with_pixels(pixels), task_type="rotate")

    assert rotated.size_hw == (6, 4)
    assert torch.nonzero(rotated.tensor_image[0]).tolist() == [[0, 0]]


def test_rotation_starts_a_new_root_with_source_and_time():
    rotated = preprocess(image(), task_type="rotate", rotation_degrees=30)

    radians = math.radians(30)
    expected_width = int(80 * math.sin(radians) + 100 * math.cos(radians))
    expected_height = int(80 * math.cos(radians) + 100 * math.sin(radians))
    assert rotated.size_hw == (expected_height, expected_width)
    assert rotated.root.frame_id == rotated.image_id != "root"
    assert rotated.image_id.startswith("rotate-")
    assert rotated.video_metadata is VIDEO


def test_full_turn_returns_the_input_with_its_provenance():
    source = image()

    assert preprocess(source, task_type="rotate", rotation_degrees=-360) is source


@pytest.mark.parametrize(
    ("flip_type", "dims"), [("vertical", (1,)), ("horizontal", (2,)), ("both", (1, 2))]
)
def test_flip_mirrors_pixels_on_a_new_root(flip_type, dims):
    source = image()

    flipped = preprocess(source, task_type="flip", flip_type=flip_type)

    assert torch.equal(flipped.tensor_image, torch.flip(source.tensor_image, dims))
    assert flipped.image_id.startswith("flip-") and flipped.video_metadata is VIDEO


def test_selected_unknown_flip_type_is_an_error():
    with pytest.raises(ValueError, match="flip_type"):
        preprocess(image(), task_type="flip", flip_type="diagonal")


# --- dynamic crop ------------------------------------------------------------


def crop_rows(source, predictions, **options):
    defaults = dict(mask_opacity=0.0, background_color=(0, 0, 0))
    defaults.update(options)

    return DynamicCropBlock().run(image=source, predictions=predictions, **defaults)


def test_dynamic_crop_rounds_clamps_and_moves_the_row_into_the_crop():
    source = image()
    predictions = detections_on(source, [[-4.6, 10.5, 30.2, 90.0]])

    result = crop_rows(source, predictions)

    crop = result["crops"][0]
    assert crop.size_hw == (70, 30) and crop.parent.offset_xy == (0.0, 10.0)
    row = result["predictions"][0]
    # V1 moves the unrounded box by the crop offset.
    assert row.xyxy[0].tolist() == pytest.approx([-4.6, 0.5, 30.2, 80.0])
    assert frame_of(row).image_id == crop.image_id
    assert row.bboxes_metadata == [{"row": 0}]


def test_dynamic_crop_keeps_a_none_slot_for_every_empty_box():
    source = image()
    predictions = detections_on(
        source, [[1, 1, 5, 5], [200, 0, 210, 10], [2, 2, 4, 4], [0, 90, 9, 99]]
    )

    result = crop_rows(source, predictions)

    # Rows 1 and 3 (trailing) miss the 100x80 image: None slots keep 4 slots.
    crops, rows = result["crops"], result["predictions"]
    assert crops.indices == ((0,), (1,), (2,), (3,))
    assert [crop is None for crop in crops] == [False, True, False, True]
    assert [row is None for row in rows] == [False, True, False, True]
    assert rows[2].bboxes_metadata[0]["row"] == 2


def test_dynamic_crop_without_rows_returns_empty_batches():
    source = image()

    result = crop_rows(source, detections_on(source, torch.zeros((0, 4)).tolist()))

    assert len(result["crops"]) == 0 and len(result["predictions"]) == 0


def test_dynamic_crop_moves_keypoints_with_their_row():
    source = image()
    boxes = detections_on(source, [[10, 20, 30, 40], [50, 50, 60, 60]])
    key_points = KeyPoints(
        xy=torch.tensor([[[12.0, 22.0]], [[55.0, 57.0]]]),
        class_id=torch.tensor([0, 0]),
        confidence=torch.ones((2, 1)),
        image_metadata=source.prediction_metadata(),
    )

    rows = crop_rows(source, (key_points, boxes))["predictions"]

    assert rows[1][0].xy.tolist() == [[[5.0, 7.0]]]
    assert rows[1][1].xyxy.tolist() == [[0.0, 0.0, 10.0, 10.0]]


@pytest.mark.parametrize("encoding", ["dense", "rle"])
def test_dynamic_crop_fades_outside_the_mask(encoding):
    source = image(10, 10, fill=(100, 100, 100))
    mask = torch.zeros((1, 10, 10), dtype=torch.bool)
    mask[0, 2:4, 2:4] = True
    if encoding == "rle":
        mask = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(10, 10), masks=[torch_mask_to_coco_rle(mask[0])]
        )
    predictions = detections_on(source, [[1, 1, 5, 5]], mask=mask)

    crop = crop_rows(source, predictions, mask_opacity=0.5, background_color="#00FF00")[
        "crops"
    ][0]

    # Outside: 0.5 * (0, 255, 0) + 0.5 * 100 = (50, 177.5 -> 178 half-even, 50).
    assert crop.tensor_image[:, 0, 0].tolist() == [50, 178, 50]
    assert crop.tensor_image[:, 1, 1].tolist() == [100, 100, 100]
    assert source.tensor_image[:, 1, 1].tolist() == [100, 100, 100]


@pytest.mark.parametrize(
    ("value", "rgb"),
    [
        ("#431112", (67, 17, 18)),
        ("#0f8", (0, 255, 136)),
        ("(1, 2, 3)", (1, 2, 3)),
        ([4, 5, 6], (4, 5, 6)),
    ],
)
def test_parse_rgb_accepts_v1_notations(value, rgb):
    assert parse_rgb(value) == rgb


@pytest.mark.parametrize("value", ["red", "#12345", (1, 2), (1, 2, 300), "(a,b,c)"])
def test_parse_rgb_rejects_other_values(value):
    with pytest.raises(ValueError):
        parse_rgb(value)


# --- area resize, row identity, segmentation views ---------------------------


def test_resize_equals_opencv_area_interpolation():
    source = image(60, 100)

    resized = preprocess(source, width=40, height=30)

    pixels = source.tensor_image.permute(1, 2, 0).numpy()
    expected = cv2.resize(pixels, (40, 30), interpolation=cv2.INTER_AREA)
    assert np.array_equal(resized.tensor_image.permute(1, 2, 0).numpy(), expected)


@pytest.mark.parametrize(
    "options",
    [
        dict(task_type="resize", width=20, height=None),
        dict(task_type="rotate", rotation_degrees=90),
        dict(task_type="rotate", rotation_degrees=30),
        dict(task_type="flip", flip_type="both"),
    ],
)
def test_preprocessing_keeps_the_parent_row_of_a_crop(options):
    parent_row = RowRef(namespace="detections", id=3)
    crop = image().crop((0, 0, 40, 40)).with_parent_row(parent_row)

    assert preprocess(crop, **options).parent_row == parent_row


def test_dynamic_crop_labels_each_crop_with_its_detection_row():
    source = image()
    minted = with_row_identity(
        detections_on(source, [[1, 1, 5, 5], [200, 0, 210, 10], [10, 10, 20, 20]])
    )
    namespace = row_record(minted).namespace
    # Sorted and reordered before cropping: slot k is no longer row k.
    reordered = select_predictions(minted, indices=[2, 1, 0])

    result = crop_rows(source, reordered)

    crops, rows = result["crops"], result["predictions"]
    assert crops[0].parent_row == RowRef(namespace=namespace, id=2)
    assert crops[1] is None
    assert crops[2].parent_row == RowRef(namespace=namespace, id=0)
    # The one-row prediction of a crop keeps that row's key.
    assert row_record(rows[0]).namespace == namespace
    assert row_record(rows[0]).host_ids() == [2]


def test_dynamic_crop_without_row_identity_gives_crops_without_parent_row():
    # The source is itself a crop with a parent row; its crops must not inherit
    # a row they were not cut for.
    source = image().with_parent_row(RowRef(namespace="outer", id=5))

    result = crop_rows(source, detections_on(source, [[1, 1, 5, 5]]))

    assert result["crops"][0].parent_row is None


def masked_detections(source):
    mask = torch.zeros((2, *source.size_hw), dtype=torch.bool)
    mask[0, 2:4, 2:4] = True
    mask[1, 6:9, 5:8] = True
    predictions = detections_on(source, [[1, 1, 5, 5], [5, 6, 8, 9]], mask=mask)

    return with_row_identity(predictions)


def test_dynamic_crop_of_a_view_reads_boxes_without_materializing(monkeypatch):
    source = image(10, 10)
    predictions = masked_detections(source)
    view = SegmentationView.from_dense(predictions)

    def refuse(_view):
        raise AssertionError("box-only cropping requested full-resolution masks")

    monkeypatch.setattr(SegmentationView, "full_res", refuse)

    from_view = crop_rows(source, view)
    from_dense = crop_rows(source, predictions)

    for view_row, dense_row in zip(from_view["predictions"], from_dense["predictions"]):
        assert type(view_row) is Detections
        assert torch.equal(view_row.xyxy, dense_row.xyxy)
        assert row_record(view_row).host_ids() == row_record(dense_row).host_ids()
    assert [crop.parent_row for crop in from_view["crops"]] == [
        crop.parent_row for crop in from_dense["crops"]
    ]


def test_dynamic_crop_fade_from_a_view_equals_the_dense_fade():
    source = image(10, 10, fill=(100, 100, 100))
    predictions = masked_detections(source)
    options = dict(mask_opacity=0.5, background_color="#00FF00")

    from_view = crop_rows(source, SegmentationView.from_dense(predictions), **options)
    from_dense = crop_rows(source, predictions, **options)

    for view_crop, dense_crop in zip(from_view["crops"], from_dense["crops"]):
        assert torch.equal(view_crop.tensor_image, dense_crop.tensor_image)


# --- prediction frames and required boxes (decision 021) ---------------------


def test_dynamic_crop_maps_boxes_detected_on_a_resize_of_the_image():
    source = image()
    resized = source.resize((40, 50))
    predictions = with_row_identity(detections_on(resized, [[5, 10, 15, 20]]))

    result = crop_rows(source, predictions)

    crop = result["crops"][0]
    assert crop.parent.offset_xy == (10.0, 20.0) and crop.size_hw == (20, 20)
    assert torch.equal(crop.tensor_image, source.tensor_image[:, 20:40, 10:30])
    assert crop.video_metadata is VIDEO
    namespace = row_record(predictions).namespace
    assert crop.parent_row == RowRef(namespace=namespace, id=0)
    row = result["predictions"][0]
    assert row.xyxy.tolist() == [[0.0, 0.0, 20.0, 20.0]]
    assert row.bboxes_metadata == [{"row": 0}]
    assert row_record(row).host_ids() == [0]
    assert frame_of(row).image_id == crop.image_id
    assert frame_of(row).video_metadata == VIDEO


@pytest.mark.parametrize(
    ("predicted_on", "cropped", "box", "expected_root_xy"),
    [
        # Detected on a static crop at (40, 0), cropped from the image.
        ("crop", "image", [10, 10, 30, 30], (50.0, 10.0)),
        # Detected on the image, cropped from its static crop at (40, 0).
        ("image", "crop", [50, 10, 70, 30], (50.0, 10.0)),
    ],
)
def test_dynamic_crop_maps_boxes_between_an_image_and_its_static_crop(
    predicted_on, cropped, box, expected_root_xy
):
    frames = {"image": image()}
    frames["crop"] = frames["image"].crop((40, 0, 90, 80))
    predictions = detections_on(frames[predicted_on], [box])

    crop = crop_rows(frames[cropped], predictions)["crops"][0]

    assert crop.root.offset_xy == expected_root_xy and crop.size_hw == (20, 20)


def test_dynamic_crop_maps_keypoints_with_their_boxes():
    source = image()
    resized = source.resize((40, 50))
    key_points = KeyPoints(
        xy=torch.tensor([[[6.0, 11.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
        image_metadata=resized.prediction_metadata(),
    )
    boxes = detections_on(resized, [[5, 10, 15, 20]])

    (row,) = crop_rows(source, (key_points, boxes))["predictions"]

    # (6, 11) in the resize is (12, 22) in the image, (2, 2) in the crop.
    assert row[0].xy.tolist() == [[[2.0, 2.0]]]
    assert row[1].xyxy.tolist() == [[0.0, 0.0, 20.0, 20.0]]


def test_dynamic_crop_fades_with_masks_mapped_from_a_resize():
    source = image(20, 20, fill=(100, 100, 100))
    resized = source.resize((10, 10))
    mask = torch.zeros((1, 10, 10), dtype=torch.bool)
    mask[0, 1:3, 1:3] = True
    predictions = detections_on(resized, [[1, 1, 4, 4]], mask=mask)

    crop = crop_rows(source, predictions, mask_opacity=1.0, background_color="#000")[
        "crops"
    ][0]

    # Box 2..8 in the image; the mask covers 2..6, so the last two pixels fade.
    assert crop.size_hw == (6, 6)
    assert crop.tensor_image[0, 0].tolist() == [100, 100, 100, 100, 0, 0]


def grid_view_on(source, xyxy):
    # A compact view: a 2x-coarser grid, box-shaped probabilities.
    height, width = source.size_hw
    grid = torch.zeros((len(xyxy), height // 2, width // 2))
    for row, (x0, y0, x1, y1) in enumerate(xyxy):
        grid[row, y0 // 2 : y1 // 2, x0 // 2 : x1 // 2] = 1.0
    geometry = MaskGridGeometry(
        grid_size_hw=(height // 2, width // 2),
        inference_size_hw=(height, width),
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(1.0, 1.0),
        crop_size_hw=(height, width),
        original_size_hw=(height, width),
        static_crop_xywh=(0, 0, width, height),
    )

    return SegmentationView(
        detections=with_row_identity(detections_on(source, xyxy)),
        scores=grid,
        score_type="probabilities",
        threshold=0.5,
        geometry=geometry,
    )


def test_dynamic_crop_maps_grid_view_boxes_without_materializing(monkeypatch):
    source = image()
    view = grid_view_on(source.resize((40, 50)), [[4, 10, 14, 20]])

    def refuse(_view):
        raise AssertionError("box-only cropping requested full-resolution masks")

    monkeypatch.setattr(SegmentationView, "full_res", refuse)

    result = crop_rows(source, view)

    crop, row = result["crops"][0], result["predictions"][0]
    assert crop.parent.offset_xy == (8.0, 20.0) and crop.size_hw == (20, 20)
    assert type(row) is Detections and row.xyxy.tolist() == [[0.0, 0.0, 20.0, 20.0]]
    assert view.materialization_count == 0


def test_dynamic_crop_refuses_boxes_of_another_input_or_a_rotation():
    source = image()
    other = ImageData.from_tensor(source.tensor_image, image_id="other")
    rotated = preprocess(source, task_type="rotate", rotation_degrees=90)

    for predicted_on in (other, rotated):
        predictions = detections_on(predicted_on, [[1, 1, 5, 5]])
        with pytest.raises(ContractError, match="Bind 'image' to the image"):
            crop_rows(source, predictions)


def legacy_detections(xyxy, metadata):
    return Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32),
        class_id=torch.zeros(len(xyxy), dtype=torch.int64),
        confidence=torch.full((len(xyxy),), 0.9),
        image_metadata=metadata,
    )


def test_dynamic_crop_rules_for_metadata_without_a_typed_frame():
    source = image()

    def crop(metadata):
        predictions = legacy_detections([[10, 20, 30, 40]], metadata)
        return crop_rows(source, predictions)["crops"][0]

    # No frame metadata at all: documented as made on the cropped image.
    assert crop(None).parent.offset_xy == (10.0, 20.0)
    # Legacy keys naming the cropped image.
    assert crop({"parent_id": "root"}).parent.offset_xy == (10.0, 20.0)
    # Legacy keys of another image without a root placement cannot be mapped.
    with pytest.raises(ContractError, match="does not name the workflow input"):
        crop({"parent_id": "elsewhere"})


def test_dynamic_crop_never_matches_two_unnamed_workflow_inputs():
    # Imported V1 images may not name their root; equal sizes prove nothing.
    unnamed_root = FrameMapping(frame_id=None, frame_size_hw=(80, 100))
    source = ImageData(
        tensor_image=image().tensor_image,
        image_id="imported",
        parent=unnamed_root,
        root=unnamed_root,
    )
    other = replace(source, image_id="other-import")
    predictions = detections_on(other, [[10, 20, 30, 40]])

    with pytest.raises(ContractError, match="does not name the workflow input"):
        crop_rows(source, predictions)


@pytest.mark.parametrize("count", [4, 0])
def test_dynamic_crop_refuses_keypoints_without_boxes(count):
    source = image()
    key_points = KeyPoints(
        xy=torch.ones((count, 1, 2)),
        class_id=torch.zeros(count, dtype=torch.int64),
        confidence=torch.ones((count, 1)),
        image_metadata=source.prediction_metadata(),
    )

    with pytest.raises(ContractError, match=f"no bounding boxes \\({count} skeletons"):
        crop_rows(source, (key_points, None))


def test_dynamic_crop_of_absent_predictions_is_empty():
    result = crop_rows(image(), None)

    assert len(result["crops"]) == 0 and len(result["predictions"]) == 0
