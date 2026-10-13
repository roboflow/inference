"""Region frames and required boxes through the real V2 compiler and session.

Decision 021. Dynamic crop reads boxes in the frame of the image it crops;
predictions made on a resize or static crop of the same input are mapped::

    input 100x100, object [50, 40, 70, 60]
      ├─ resize 50x50 ─────────── detect: [25, 20, 35, 30]  (scale 2)
      ├─ static crop x 40..90 ─── detect: [10, 40, 30, 60]  (offset 40, 0)
      └─────────────────────────────────── dynamic crop of the input:
                                           offset (50, 40), size 20x20

Detections from another input or a rotation (a new root) are refused. The box
consumers (dynamic crop, detection offset, classes replacement) refuse
keypoints without boxes (decision 018); gated and genuinely empty predictions
still pass. Detectors are the stand-ins of ``deterministic.py`` and the fake
models of ``tests/v2/support/models``, not real inference.
"""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.models import KeypointDetectionBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.row_identity import RowRef, row_record
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError

from tests.v2.integration.blocks.geometry.deterministic import (
    painted_image,
    regions_catalogue,
)
from tests.v2.support.models.fake_task_models import KeypointModel
from tests.v2.support.models.grid_segmentation import (
    GridInstanceModel,
    GridInstanceSegmentationBlock,
)

OBJECT = (50, 40, 70, 60)
BRIGHT, DARK = 200, 20
MISSING_BOXES = "keypoint prediction has no bounding boxes"

PREPARE = {
    "resize": {
        "type": "v2/image_preprocessing@v1",
        "name": "prepare",
        "image": "$inputs.image",
        "task_type": "resize",
        "width": 50,
        "height": 50,
    },
    "static_crop": {
        "type": "v2/absolute_static_crop@v1",
        "name": "prepare",
        "image": "$inputs.image",
        "x_center": 65,
        "y_center": 50,
        "width": 50,
        "height": 100,
    },
    "rotate": {
        "type": "v2/image_preprocessing@v1",
        "name": "prepare",
        "image": "$inputs.image",
        "task_type": "rotate",
        "rotation_degrees": 90,
    },
}
PREPARED_IMAGE = {
    "resize": "$steps.prepare.image",
    "static_crop": "$steps.prepare.crops",
    "rotate": "$steps.prepare.image",
}


def run(steps, outputs, *, inputs, models=None):
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    catalogue = regions_catalogue().with_blocks(
        [KeypointDetectionBlock, GridInstanceSegmentationBlock]
    )
    plan = compile_workflow(workflow, catalogue=catalogue)
    result = plan.create_session({"models": models or {}}).run(inputs)

    return result


def object_image():
    return painted_image((100, 100), [(*OBJECT, 1, 0)])


def crop_of_input(prepare, *, detector="test/marker_detector@v1", **crop_options):
    steps = [
        PREPARE[prepare],
        {"type": detector, "name": "detect", "image": PREPARED_IMAGE[prepare]},
        {
            "type": "v2/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.detect.predictions",
            **crop_options,
        },
        {"type": "test/color_name@v1", "name": "text", "image": "$steps.crop.crops"},
    ]

    return steps


CROP_OUTPUTS = {
    "boxes": "$steps.detect.predictions",
    "crops": "$steps.crop.crops",
    "rows": "$steps.crop.predictions",
    "text": "$steps.text.text",
}


@pytest.mark.parametrize("prepare", ["resize", "static_crop"])
def test_crop_of_the_input_maps_boxes_detected_on_a_resize_or_static_crop(prepare):
    result = run(
        crop_of_input(prepare), CROP_OUTPUTS, inputs={"image": [object_image()]}
    )

    raw = result.outputs.data
    detections = raw["boxes"][0]
    (crop,) = raw["crops"][0]
    assert crop.root.offset_xy == (50.0, 40.0) and crop.size_hw == (20, 20)
    # Every crop pixel is the painted object: (255, 1, 0).
    assert torch.equal(
        crop.tensor_image,
        torch.tensor([255, 1, 0], dtype=torch.uint8).reshape(3, 1, 1).expand(3, 20, 20),
    )
    record = row_record(detections)
    assert crop.parent_row == RowRef(namespace=record.namespace, id=0)
    (row,) = raw["rows"][0]
    assert row.xyxy.tolist() == [[0.0, 0.0, 20.0, 20.0]]
    assert row.bboxes_metadata == [{"marker": 1}]
    assert row_record(row).host_ids() == [0]
    assert result.rows()[0]["text"] == ["red"]
    # Workflow rows convert the per-crop box back to the input.
    (converted,) = result.rows()[0]["rows"]
    assert converted.xyxy.tolist() == [[50.0, 40.0, 70.0, 60.0]]


def test_crop_of_a_static_crop_maps_boxes_detected_on_the_input():
    steps = [
        PREPARE["static_crop"],
        {"type": "test/marker_detector@v1", "name": "detect", "image": "$inputs.image"},
        {
            "type": "v2/dynamic_crop@v1",
            "name": "crop",
            "image": "$steps.prepare.crops",
            "predictions": "$steps.detect.predictions",
        },
    ]

    result = run(
        steps, {"crops": "$steps.crop.crops"}, inputs={"image": [object_image()]}
    )

    (crop,) = result.outputs.data["crops"][0]
    assert crop.parent.offset_xy == (10.0, 40.0)
    assert crop.root.offset_xy == (50.0, 40.0) and crop.size_hw == (20, 20)


def test_crop_of_the_input_refuses_boxes_detected_on_a_rotation():
    # A rotation starts a new root: no mapping into the input exists.
    with pytest.raises(StepExecutionError, match="Bind 'image' to the image"):
        run(
            crop_of_input("rotate"),
            {"crops": "$steps.crop.crops"},
            inputs={"image": [object_image()]},
        )


def test_masks_detected_on_a_resize_fade_the_crop_of_the_input():
    steps = crop_of_input(
        "resize",
        detector="test/masked_marker_detector@v1",
        mask_opacity=1.0,
        background_color="#0000FF",
    )

    result = run(steps, CROP_OUTPUTS, inputs={"image": [object_image()]})

    (crop,) = result.outputs.data["crops"][0]
    # The mask moved with its box: nothing inside the object faded to blue.
    assert crop.root.offset_xy == (50.0, 40.0)
    assert torch.equal(
        crop.tensor_image[0], torch.full((20, 20), 255, dtype=torch.uint8)
    )


@pytest.fixture
def materializations(monkeypatch) -> List[int]:
    """Every full-resolution upscale of any SegmentationView in the test."""
    calls: List[int] = []
    materialize = SegmentationView._materialize

    def counted(view):
        calls.append(len(view))
        return materialize(view)

    monkeypatch.setattr(SegmentationView, "_materialize", counted)

    return calls


def test_grid_view_boxes_from_a_resize_crop_the_input_without_masks(materializations):
    image = ImageData.from_tensor(
        torch.full((3, 40, 60), BRIGHT, dtype=torch.uint8), image_id="input"
    )
    steps = [
        {**PREPARE["resize"], "width": 30, "height": 20},
        {
            "type": GridInstanceSegmentationBlock.type,
            "name": "segmenter",
            "image": "$steps.prepare.image",
            "model_id": "seg",
            "confidence": 0.0,
        },
        {
            "type": "v2/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.segmenter.segmentation",
        },
    ]

    result = run(
        steps,
        {"crops": "$steps.crop.crops"},
        inputs={"image": [image]},
        models={"seg": GridInstanceModel()},
    )

    # Fake rows i = 0, 1, 2 on 30x20: [i, i, 30 - i, 20 - i], twice as large.
    crops = result.outputs.data["crops"][0]
    assert [crop.root.offset_xy for crop in crops] == [(0, 0), (2, 2), (4, 4)]
    assert [crop.size_hw for crop in crops] == [(40, 60), (36, 56), (32, 52)]
    assert materializations == []


# --- box consumers and keypoints without boxes ------------------------------


KEY_POINTS_STEP = {
    "type": "v2/keypoint_detection@v1",
    "name": "kp",
    "image": "$inputs.image",
    "model_id": "kp",
    "confidence": 0.0,
}
CROP_STEP = {
    "type": "v2/dynamic_crop@v1",
    "name": "crop",
    "image": "$inputs.image",
    "predictions": "$steps.kp.predictions",
}
OFFSET_STEP = {
    "type": "v2/detection_offset@v1",
    "name": "offset",
    "predictions": "$steps.kp.predictions",
    "offset_width": 4,
    "offset_height": 4,
}


def replace_step(text_source):
    step = {
        "type": "v2/detections_classes_replacement@v1",
        "name": "replace",
        "object_detection_predictions": "$steps.kp.predictions",
        "classification_predictions": "$steps.text.text",
        "crops": text_source,
    }

    return step


def text_step(images):
    return {"type": "test/color_name@v1", "name": "text", "image": images}


# Without boxes there are no crops to classify: the replacement is fed text
# of slices instead, so that its own check is reached.
SLICE_STEP = {
    "type": "v2/image_slicer@v1",
    "name": "slicer",
    "image": "$inputs.image",
    "slice_width": 16,
    "slice_height": 20,
}
WITHOUT_BOXES_STEPS = {
    "crop": [KEY_POINTS_STEP, CROP_STEP],
    "offset": [KEY_POINTS_STEP, OFFSET_STEP],
    "replace": [
        KEY_POINTS_STEP,
        SLICE_STEP,
        text_step("$steps.slicer.slices"),
        replace_step("$steps.slicer.slices"),
    ],
}


def gray_image(value):
    return np.full((20, 30, 3), value, dtype=np.uint8)


def test_keypoints_with_boxes_reach_every_box_consumer():
    steps = [
        KEY_POINTS_STEP,
        CROP_STEP,
        text_step("$steps.crop.crops"),
        replace_step("$steps.crop.crops"),
        OFFSET_STEP,
    ]

    result = run(
        steps,
        {
            "crops": "$steps.crop.crops",
            "replaced": "$steps.replace.predictions",
            "offset": "$steps.offset.predictions",
        },
        inputs={"image": [gray_image(BRIGHT)]},
        models={"kp": KeypointModel()},
    )

    raw = result.outputs.data
    # Fake rows i = 0, 1, 2 on 30x20: [i, i, 30 - i, 20 - i].
    assert [crop.size_hw for crop in raw["crops"][0]] == [(20, 30), (18, 28), (16, 26)]
    key_points, replaced = raw["replaced"][0]
    assert [row["class"] for row in replaced.bboxes_metadata] == ["red"] * 3
    assert len(key_points) == 3
    _, padded = raw["offset"][0]
    assert padded.xyxy[1].tolist() == [0.0, 0.0, 30.0, 20.0]


@pytest.mark.parametrize(
    ("consumer", "step_type"),
    [
        ("crop", "v2/dynamic_crop@v1"),
        ("offset", "v2/detection_offset@v1"),
        ("replace", "v2/detections_classes_replacement@v1"),
    ],
)
@pytest.mark.parametrize("brightness", [BRIGHT, DARK])
def test_box_consumers_refuse_keypoints_without_boxes(consumer, step_type, brightness):
    # Dark images give no skeletons: the boxes are missing all the same.
    with pytest.raises(StepExecutionError, match=MISSING_BOXES) as raised:
        run(
            WITHOUT_BOXES_STEPS[consumer],
            {"result": f"$steps.{consumer}.predictions"},
            inputs={"image": [gray_image(brightness)]},
            models={"kp": KeypointModel(with_boxes=False)},
        )

    assert step_type in str(raised.value)


def test_gated_keypoints_without_boxes_reach_no_box_consumer():
    steps = [
        {
            "type": "v2/has_brightness",
            "name": "bright",
            "image": "$inputs.image",
            "minimum": 255.0,
        },
        {
            "type": "v2/continue_if",
            "name": "gate",
            "condition": "$steps.bright.keep",
            "next_steps": ["$steps.kp"],
        },
        KEY_POINTS_STEP,
        OFFSET_STEP,
    ]

    result = run(
        steps,
        {"offset": "$steps.offset.predictions"},
        inputs={"image": [gray_image(BRIGHT)]},
        models={"kp": KeypointModel(with_boxes=False)},
    )

    assert result.rows()[0]["offset"] is None


def test_empty_detections_pass_every_box_consumer():
    steps = [
        {"type": "test/marker_detector@v1", "name": "detect", "image": "$inputs.image"},
        {
            "type": "v2/detection_offset@v1",
            "name": "offset",
            "predictions": "$steps.detect.predictions",
            "offset_width": 4,
            "offset_height": 4,
        },
        {
            "type": "v2/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.offset.predictions",
        },
    ]
    outputs: Dict[str, Any] = {
        "offset": "$steps.offset.predictions",
        "crops": "$steps.crop.crops",
    }

    result = run(steps, outputs, inputs={"image": [painted_image((20, 20), [])]})

    row = result.rows()[0]
    assert row["offset"].xyxy.shape == (0, 4) and row["crops"] == []
