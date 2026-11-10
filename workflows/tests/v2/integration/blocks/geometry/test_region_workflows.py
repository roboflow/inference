"""Region blocks composed through the real V2 compiler and session.

Flows:
* SAHI: slice -> detect each slice -> stitch, boxes back in the input image.
* Two-stage: detect -> dynamic crop -> classify each crop -> replace classes.
* Gated, empty and nested variants of both.
Detection and classification are the deterministic stand-ins of
``deterministic.py``; expected values below are worked out from the painted
markers, not read from a previous run.
"""

import pytest
import torch
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.compilation import compile_workflow

from tests.v2.integration.blocks.geometry.deterministic import (
    painted_image,
    regions_catalogue,
)

# 64x128 image; slices 80 wide with ratio 0.4 start at x = 0 and x = 48.
#   1: only in slice 0.            2: whole in both slices (duplicate).
#   3: x 74..90, cut at 80 in slice 0 (area 60), whole in slice 1 (area 160):
#      IoU 60 / 160 = 0.375.        4: only in slice 1.
MARKERS = [
    (10, 10, 20, 30, 1, 0),
    (60, 40, 70, 50, 2, 1),
    (74, 5, 90, 15, 3, 0),
    (100, 5, 120, 15, 4, 1),
]
WHOLE = {(10, 10, 20, 30), (60, 40, 70, 50), (74, 5, 90, 15), (100, 5, 120, 15)}


def confidence(area):
    return 0.5 + area / 1000


def run(steps, outputs, *, inputs, extra_inputs=()):
    workflow = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            *({"type": "WorkflowParameter", "name": name} for name in extra_inputs),
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    plan = compile_workflow(workflow, catalogue=regions_catalogue())
    result = plan.create_session().run(inputs)

    return result


def sahi_steps(*, image="$inputs.image", strategy="$inputs.strategy"):
    steps = [
        {
            "type": "roboflow_core/image_slicer@v2",
            "name": "slicer",
            "image": image,
            "slice_width": 80,
            "slice_height": 64,
            "overlap_ratio_width": 0.4,
            "overlap_ratio_height": 0.0,
        },
        {
            "type": "test/marker_detector@v1",
            "name": "detect",
            "image": "$steps.slicer.slices",
        },
        {
            "type": "roboflow_core/detections_stitch@v1",
            "name": "stitch",
            "reference_image": image,
            "predictions": "$steps.detect.predictions",
            "overlap_filtering_strategy": strategy,
            "iou_threshold": 0.3,
        },
    ]

    return steps


def boxes_of(prediction):
    return sorted(tuple(int(v) for v in box) for box in prediction.xyxy.tolist())


def test_sahi_none_keeps_every_slice_row_in_input_coordinates() -> None:
    result = run(
        sahi_steps(),
        {"stitched": "$steps.stitch.predictions", "slices": "$steps.slicer.slices"},
        inputs={"image": [painted_image((64, 128), MARKERS)], "strategy": "none"},
        extra_inputs=["strategy"],
    )

    slices = result.outputs.data["slices"][0]
    assert [one.root.offset_xy for one in slices] == [(0.0, 0.0), (48.0, 0.0)]
    stitched = result.rows()[0]["stitched"]
    assert boxes_of(stitched) == sorted([*WHOLE, (60, 40, 70, 50), (74, 5, 80, 15)])
    assert stitched.image_metadata["class_names"] == {0: "box", 1: "disc"}
    markers = sorted(row["marker"] for row in stitched.bboxes_metadata)
    assert markers == [1, 2, 2, 3, 3, 4]


def test_sahi_nms_keeps_the_whole_copy_of_each_marker() -> None:
    result = run(
        sahi_steps(strategy="nms"),
        {"stitched": "$steps.stitch.predictions"},
        inputs={"image": [painted_image((64, 128), MARKERS)]},
    )

    stitched = result.rows()[0]["stitched"]
    assert set(boxes_of(stitched)) == WHOLE and len(stitched.xyxy) == 4
    # Rows stay aligned: marker numbers follow their boxes.
    by_marker = {
        row["marker"]: tuple(int(v) for v in box)
        for row, box in zip(stitched.bboxes_metadata, stitched.xyxy.tolist())
    }
    assert by_marker[3] == (74, 5, 90, 15)


def test_sahi_nmm_merges_cut_and_whole_copies_with_area_weighted_confidence() -> None:
    result = run(
        sahi_steps(strategy="nmm"),
        {"stitched": "$steps.stitch.predictions"},
        inputs={"image": [painted_image((64, 128), MARKERS)]},
    )

    stitched = result.rows()[0]["stitched"]
    rows = {
        tuple(int(v) for v in box): score
        for box, score in zip(stitched.xyxy.tolist(), stitched.confidence.tolist())
    }
    assert set(rows) == WHOLE
    cut, whole = confidence(60), confidence(160)
    expected = (60 * cut + 160 * whole) / (60 + 160)
    assert rows[(74, 5, 90, 15)] == pytest.approx(expected, abs=1e-6)
    assert rows[(60, 40, 70, 50)] == pytest.approx(confidence(100), abs=1e-6)


def test_stitch_into_a_crop_that_was_sliced_again() -> None:
    # Crop x 40..128 of the input, then slice that crop; stitch into the crop.
    steps = [
        {
            "type": "roboflow_core/absolute_static_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "x_center": 84,
            "y_center": 32,
            "width": 88,
            "height": 64,
        },
        *sahi_steps(image="$steps.crop.crops", strategy="nms"),
    ]

    result = run(
        steps,
        {"stitched": "$steps.stitch.predictions"},
        inputs={"image": [painted_image((64, 128), MARKERS)]},
    )

    stitched = result.rows()[0]["stitched"]
    # Workflow output converts to the input image ("parent" coordinates).
    assert set(boxes_of(stitched)) == {
        (60, 40, 70, 50),
        (74, 5, 90, 15),
        (100, 5, 120, 15),
    }


def test_stitch_of_an_image_without_detections_is_empty() -> None:
    result = run(
        sahi_steps(strategy="nms"),
        {"stitched": "$steps.stitch.predictions"},
        inputs={"image": [painted_image((64, 128), [])]},
    )

    stitched = result.rows()[0]["stitched"]
    assert stitched.xyxy.shape == (0, 4)


TWO_STAGE_STEPS = [
    {"type": "test/marker_detector@v1", "name": "detect", "image": "$inputs.image"},
    {
        "type": "roboflow_core/dynamic_crop@v1",
        "name": "crop",
        "image": "$inputs.image",
        "predictions": "$steps.detect.predictions",
    },
    {
        "type": "test/color_classifier@v1",
        "name": "classify",
        "image": "$steps.crop.crops",
    },
    {
        "type": "roboflow_core/detections_classes_replacement@v1",
        "name": "replace",
        "object_detection_predictions": "$steps.detect.predictions",
        "classification_predictions": "$steps.classify.predictions",
        "fallback_class_name": "unknown",
        "fallback_class_id": 7,
    },
]


def colored_image():
    # Marker 1 is pure marker colour (255, 1, 0): red, mean brightness 85.3.
    # Marker 2 (20x20) gets a (100, 250, 200) 16x16 centre; its crop has channel
    # means R 155.8, G 160.7, B 128.0 -> green, brightness 148.2.
    pixels = painted_image((40, 60), [(5, 5, 15, 15, 1, 0), (30, 10, 50, 30, 2, 0)])
    pixels[12:28, 32:48] = (100, 250, 200)

    return pixels


def test_detect_crop_classify_replace_classes() -> None:
    result = run(
        TWO_STAGE_STEPS,
        {"replaced": "$steps.replace.predictions", "crops": "$steps.crop.crops"},
        inputs={"image": [colored_image()]},
    )

    crops = result.outputs.data["crops"][0]
    assert crops.indices == ((0, 0), (0, 1))
    replaced = result.rows()[0]["replaced"]
    assert [row["class"] for row in replaced.bboxes_metadata] == ["red", "green"]
    assert replaced.class_id.tolist() == [0, 1]
    assert replaced.image_metadata["class_names"] == {0: "red", 1: "green"}
    assert [row["marker"] for row in replaced.bboxes_metadata] == [1, 2]


def test_gated_crop_falls_back_and_ungated_rows_keep_their_classes() -> None:
    steps = [
        *TWO_STAGE_STEPS[:2],
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
        *TWO_STAGE_STEPS[2:],
    ]

    result = run(
        steps,
        {"replaced": "$steps.replace.predictions"},
        inputs={"image": [colored_image()]},
    )

    # Crop 0 (brightness 85.3 < 100) is gated off; crop 1 (148.2) is classified.
    replaced = result.rows()[0]["replaced"]
    assert [row["class"] for row in replaced.bboxes_metadata] == ["unknown", "green"]
    assert replaced.class_id.tolist() == [7, 1]
    assert replaced.confidence[0].item() == 0.0


def test_two_stage_on_a_batch_with_an_image_without_detections() -> None:
    result = run(
        TWO_STAGE_STEPS,
        {"replaced": "$steps.replace.predictions"},
        inputs={"image": [painted_image((20, 20), []), colored_image()]},
    )

    first, second = (row["replaced"] for row in result.rows())
    assert first.xyxy.shape == (0, 4)
    assert [row["class"] for row in second.bboxes_metadata] == ["red", "green"]


def test_dimension_collapse_lists_crop_results_per_image() -> None:
    steps = [
        *TWO_STAGE_STEPS[:3],
        {
            "type": "roboflow_core/dimension_collapse@v1",
            "name": "collect",
            "data": "$steps.classify.predictions",
        },
    ]

    result = run(
        steps,
        {"collected": "$steps.collect.output"},
        inputs={"image": [colored_image(), painted_image((20, 20), [])]},
    )

    first, second = (row["collected"] for row in result.rows())
    assert [int(p.class_id[0]) for p in first] == [0, 1]
    assert second == []


def test_dynamic_crop_selected_options_and_mask_free_detections() -> None:
    steps = [dict(step) for step in TWO_STAGE_STEPS[:2]]
    steps[1]["mask_opacity"] = "$inputs.opacity"
    steps[1]["background_color"] = "$inputs.color"

    result = run(
        steps,
        {"crops": "$steps.crop.crops", "rows": "$steps.crop.predictions"},
        inputs={"image": [colored_image()], "opacity": 0.5, "color": "#00FF00"},
        extra_inputs=["opacity", "color"],
    )

    # Detections without masks are never faded.
    crop = result.outputs.data["crops"][0][0]
    assert torch.equal(
        crop.tensor_image[0], torch.full((10, 10), 255, dtype=torch.uint8)
    )
    # Raw values are in crop coordinates; workflow rows convert to the input.
    rows = result.outputs.data["rows"][0]
    assert [boxes_of(row) for row in rows] == [[(0, 0, 10, 10)], [(0, 0, 20, 20)]]
    converted = result.rows()[0]["rows"]
    assert [boxes_of(row) for row in converted] == [
        [(5, 5, 15, 15)],
        [(30, 10, 50, 30)],
    ]


def gate_all(source_step, guarded_step):
    # A mean of 255 needs an all-white image; no crop or slice here is one, so
    # every child is filtered.
    return [
        {
            "type": "roboflow_core/has_brightness@v1",
            "name": "bright",
            "image": source_step,
            "minimum": 255.0,
        },
        {
            "type": "roboflow_core/continue_if_boolean@v1",
            "name": "gate",
            "condition": "$steps.bright.keep",
            "next_steps": [guarded_step],
        },
    ]


def test_all_children_filtered_still_reach_the_group_blocks() -> None:
    steps = [
        *TWO_STAGE_STEPS[:2],
        *gate_all("$steps.crop.crops", "$steps.classify"),
        *TWO_STAGE_STEPS[2:],
        {
            "type": "roboflow_core/dimension_collapse@v1",
            "name": "collect",
            "data": "$steps.classify.predictions",
        },
    ]

    result = run(
        steps,
        {
            "replaced": "$steps.replace.predictions",
            "collected": "$steps.collect.output",
        },
        inputs={"image": [colored_image()]},
    )

    row = result.rows()[0]
    # Every crop was gated off: both rows take the fallback, and the list keeps
    # one None slot per skipped crop (decision 016).
    assert [r["class"] for r in row["replaced"].bboxes_metadata] == [
        "unknown",
        "unknown",
    ]
    assert row["collected"] == [None, None]


def test_stitch_with_every_slice_filtered_is_empty() -> None:
    steps = sahi_steps(strategy="nms")
    steps = [
        steps[0],
        *gate_all("$steps.slicer.slices", "$steps.detect"),
        *steps[1:],
    ]

    result = run(
        steps,
        {"stitched": "$steps.stitch.predictions"},
        inputs={"image": [painted_image((64, 128), MARKERS)]},
    )

    stitched = result.rows()[0]["stitched"]
    assert stitched.xyxy.shape == (0, 4)


def test_resize_then_detect_maps_boxes_back_to_the_input() -> None:
    steps = [
        {
            "type": "roboflow_core/image_preprocessing@v1",
            "name": "resize",
            "image": "$inputs.image",
            "task_type": "resize",
            "width": 64,
            "height": None,
        },
        {
            "type": "test/marker_detector@v1",
            "name": "detect",
            "image": "$steps.resize.image",
        },
    ]

    result = run(
        steps,
        {"boxes": "$steps.detect.predictions"},
        inputs={"image": [painted_image((64, 128), MARKERS)]},
    )

    # Raw output is in resized pixels; workflow rows are in input pixels, twice
    # as large (V1 kept the input's metadata and reported resized pixels).
    raw = boxes_of(result.outputs.data["boxes"][0])
    converted = boxes_of(result.rows()[0]["boxes"])
    assert len(raw) == 4
    assert converted == [tuple(2 * v for v in box) for box in raw]


def test_stitch_refuses_resized_slices_with_a_clear_error() -> None:
    steps = sahi_steps(strategy="none")
    steps.insert(
        1,
        {
            "type": "roboflow_core/image_preprocessing@v1",
            "name": "resize",
            "image": "$steps.slicer.slices",
            "task_type": "resize",
            "width": 40,
            "height": 32,
        },
    )
    steps[2]["image"] = "$steps.resize.image"

    with pytest.raises(Exception, match="only crops and slices without resizing"):
        run(
            steps,
            {"stitched": "$steps.stitch.predictions"},
            inputs={"image": [painted_image((64, 128), MARKERS)]},
        )


@pytest.mark.parametrize("with_hole", [False, True])
@pytest.mark.parametrize("output", ["predictions", "segmentation"])
def test_compiled_bounding_rect_follows_the_outline(with_hole, output):
    pixels = painted_image((90, 120), [(20, 20, 100, 70, 1, 0)])
    if with_hole:
        ys, xs = torch.meshgrid(torch.arange(90), torch.arange(120), indexing="ij")
        pixels[((xs - 60) ** 2 + (ys - 45) ** 2 <= 12**2).numpy()] = 0
    result = run(
        [
            {
                "type": "test/masked_marker_detector@v1",
                "name": "detect",
                "image": "$inputs.image",
            },
            {
                "type": "roboflow_core/bounding_rect@v1",
                "name": "rect",
                "predictions": f"$steps.detect.{output}",
            },
        ],
        {"fitted": "$steps.rect.predictions", "original": "$steps.detect.predictions"},
        inputs={"image": [pixels]},
    )

    row = result.rows()[0]
    fitted, original = row["fitted"], row["original"]
    assert fitted.xyxy.tolist() == [[20, 20, 99, 69]]
    expected = torch.zeros((1, 90, 120), dtype=torch.bool)
    expected[0, 20:70, 20:100] = True
    assert torch.equal(fitted.mask, expected)
    assert sorted(map(tuple, fitted.bboxes_metadata[0]["rect"])) == [
        (20, 20),
        (20, 69),
        (99, 20),
        (99, 69),
    ]
    assert fitted.bboxes_metadata[0]["marker"] == 1
    assert original.bboxes_metadata == [{"marker": 1}]
    # The integer radius-12 disc contains 441 pixels.
    expected_area = 3559 if with_hole else 4000
    assert int(original.mask.sum()) == expected_area
    assert torch.equal(fitted.class_id, original.class_id)
    assert torch.equal(fitted.confidence, original.confidence)
    assert fitted.image_metadata == original.image_metadata
    assert row_record(fitted).host_ids() == row_record(original).host_ids()
    assert row_record(fitted).namespace == row_record(original).namespace
