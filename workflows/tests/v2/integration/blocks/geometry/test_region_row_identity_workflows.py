"""Row keys, known child positions and segmentation views through real workflows.

Flows (decisions 015, 016, 018):

* detect -> select/reorder rows -> crop -> classify -> replace on the ORIGINAL
  detections: each class lands on the row its crop was cut for.
* The same with plain-text results, joined through the bound ``crops``.
* A gated crop leaves a ``None`` slot in the collapsed list.
* Instance segmentation as a ``SegmentationView``: cropping reads boxes only;
  bounding rect and stitch read reference masks equal to the dense ones.

The image has two markers: marker 1 (red crop) and marker 2 (green crop), see
``colored_image``. A positional join after reordering would swap the classes.
"""

import torch
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record

from tests.v2.integration.blocks.geometry.deterministic import painted_image
from tests.v2.integration.blocks.geometry.test_region_workflows import (
    MARKERS,
    colored_image,
    run,
)

DETECT = {"type": "test/marker_detector@v1", "name": "detect", "image": "$inputs.image"}


def select(rows):
    return {
        "type": "test/row_selection@v1",
        "name": "select",
        "predictions": "$steps.detect.predictions",
        "rows": rows,
    }


CROP_SELECTED = {
    "type": "roboflow_core/dynamic_crop@v1",
    "name": "crop",
    "image": "$inputs.image",
    "predictions": "$steps.select.predictions",
}
CLASSIFY = {
    "type": "test/color_classifier@v1",
    "name": "classify",
    "image": "$steps.crop.crops",
}


def replace(classifications, **extra):
    return {
        "type": "roboflow_core/detections_classes_replacement@v1",
        "name": "replace",
        "object_detection_predictions": "$steps.detect.predictions",
        "classification_predictions": classifications,
        "fallback_class_name": "unknown",
        "fallback_class_id": 7,
        **extra,
    }


def classes(predictions):
    return [row["class"] for row in predictions.bboxes_metadata]


def test_reordered_crops_label_the_rows_they_were_cut_from() -> None:
    steps = [
        DETECT,
        select([1, 0]),
        CROP_SELECTED,
        CLASSIFY,
        replace("$steps.classify.predictions"),
    ]

    result = run(
        steps,
        {"replaced": "$steps.replace.predictions"},
        inputs={"image": [colored_image()]},
    )

    replaced = result.rows()[0]["replaced"]
    # Crop 0 is marker 2 (green), crop 1 is marker 1 (red); rows keep their order.
    assert [row["marker"] for row in replaced.bboxes_metadata] == [1, 2]
    assert classes(replaced) == ["red", "green"]


def test_rows_filtered_before_cropping_take_the_fallback() -> None:
    steps = [
        DETECT,
        select([1]),
        CROP_SELECTED,
        CLASSIFY,
        replace("$steps.classify.predictions"),
    ]

    result = run(
        steps,
        {"replaced": "$steps.replace.predictions"},
        inputs={"image": [colored_image()]},
    )

    assert classes(result.rows()[0]["replaced"]) == ["unknown", "green"]


def test_text_results_join_through_their_crops() -> None:
    steps = [
        DETECT,
        select([1, 0]),
        CROP_SELECTED,
        {"type": "test/color_name@v1", "name": "name", "image": "$steps.crop.crops"},
        replace("$steps.name.text", crops="$steps.crop.crops"),
    ]

    result = run(
        steps,
        {"replaced": "$steps.replace.predictions"},
        inputs={"image": [colored_image()]},
    )

    replaced = result.rows()[0]["replaced"]
    assert classes(replaced) == ["red", "green"]
    # Text results all share class id 0, as in V1.
    assert replaced.class_id.tolist() == [0, 0]


def test_collapse_keeps_a_none_slot_for_a_gated_crop() -> None:
    steps = [
        DETECT,
        select([0, 1]),
        CROP_SELECTED,
        {
            "type": "roboflow_core/has_brightness@v1",
            "name": "bright",
            "image": "$steps.crop.crops",
            "minimum": 100.0,
        },
        {
            "type": "roboflow_core/continue_if_boolean@v1",
            "name": "gate",
            "condition": "$steps.bright.keep",
            "next_steps": ["$steps.classify"],
        },
        CLASSIFY,
        {
            "type": "roboflow_core/dimension_collapse@v1",
            "name": "collect",
            "data": "$steps.classify.predictions",
        },
    ]

    result = run(
        steps,
        {"collected": "$steps.collect.output"},
        inputs={"image": [colored_image()]},
    )

    # Marker 1's crop (brightness 85) is gated off; marker 2's (148) is classified.
    first, second = result.rows()[0]["collected"]
    assert first is None
    assert int(second.class_id[0]) == 1


SEGMENT = {
    "type": "test/masked_marker_detector@v1",
    "name": "detect",
    "image": "$inputs.image",
}


def test_cropping_a_view_never_requests_full_resolution_masks(monkeypatch) -> None:
    def refuse(_view):
        raise AssertionError("a box-only chain requested full-resolution masks")

    monkeypatch.setattr(SegmentationView, "full_res", refuse)
    steps = [
        SEGMENT,
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.detect.segmentation",
        },
        CLASSIFY,
        replace("$steps.classify.predictions"),
    ]

    result = run(
        steps,
        {"replaced": "$steps.replace.predictions"},
        inputs={"image": [colored_image()]},
    )

    replaced = result.rows()[0]["replaced"]
    assert classes(replaced) == ["red", "green"]
    # The targets are the dense predictions; the view shares their row keys.
    assert row_record(replaced).rows == 2


def test_bounding_rect_of_a_view_equals_that_of_its_predictions() -> None:
    steps = [
        SEGMENT,
        {
            "type": "roboflow_core/bounding_rect@v1",
            "name": "from_view",
            "predictions": "$steps.detect.segmentation",
        },
        {
            "type": "roboflow_core/bounding_rect@v1",
            "name": "from_dense",
            "predictions": "$steps.detect.predictions",
        },
    ]

    result = run(
        steps,
        {
            "from_view": "$steps.from_view.predictions",
            "from_dense": "$steps.from_dense.predictions",
        },
        inputs={"image": [colored_image()]},
    )

    row = result.rows()[0]
    assert torch.equal(row["from_view"].mask, row["from_dense"].mask)
    assert row["from_view"].bboxes_metadata == row["from_dense"].bboxes_metadata


def test_stitch_of_slice_views_equals_stitch_of_slice_predictions() -> None:
    def stitch(name, output):
        return {
            "type": "roboflow_core/detections_stitch@v1",
            "name": name,
            "reference_image": "$inputs.image",
            "predictions": f"$steps.detect.{output}",
            "overlap_filtering_strategy": "nms",
            "iou_threshold": 0.3,
        }

    steps = [
        {
            "type": "roboflow_core/image_slicer@v2",
            "name": "slicer",
            "image": "$inputs.image",
            "slice_width": 80,
            "slice_height": 64,
            "overlap_ratio_width": 0.4,
            "overlap_ratio_height": 0.0,
        },
        {**SEGMENT, "image": "$steps.slicer.slices"},
        stitch("from_views", "segmentation"),
        stitch("from_dense", "predictions"),
    ]

    result = run(
        steps,
        {
            "from_views": "$steps.from_views.predictions",
            "from_dense": "$steps.from_dense.predictions",
        },
        inputs={"image": [painted_image((64, 128), MARKERS)]},
    )

    row = result.rows()[0]
    assert torch.equal(row["from_views"].xyxy, row["from_dense"].xyxy)
    assert torch.equal(row["from_views"].mask, row["from_dense"].mask)
    # Stitched rows are a new result with their own row identity.
    assert (
        row_record(row["from_views"]).namespace
        != row_record(row["from_dense"]).namespace
    )
