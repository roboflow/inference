"""Crop, detect, collapse and roll up through the real compiler/session."""

import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.fusion.rollup import DetectionsListRollUpBlock
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import (
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from tests.v2.integration.blocks.geometry.deterministic import (
    MarkerDetectorBlock,
    painted_image,
    regions_catalogue,
)


class OverlappingRegionsBlock(Block):
    """Two overlapping proposals, both containing the painted marker."""

    type = "test/rollup_regions@v1"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to propose crops on.")

    def run(self, *, image):
        predictions = Detections(
            xyxy=torch.tensor([[2, 2, 24, 24], [10, 2, 30, 24]], dtype=torch.float32),
            class_id=torch.zeros(2, dtype=torch.int64),
            confidence=torch.ones(2),
            image_metadata=FrameRef.from_image(image).prediction_metadata(),
        )
        return {"predictions": with_row_identity(predictions)}


class KeypointMarkerBlock(Block):
    """One keypoint at the centre of each detected marker."""

    type = "test/rollup_keypoints@v1"
    outputs = {
        "predictions": Output(KEYPOINT_DETECTION_PREDICTION_KIND, source="image")
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to detect.")

    def run(self, *, image):
        boxes = MarkerDetectorBlock().run(image=image)["predictions"]
        points = KeyPoints(
            xy=((boxes.xyxy[:, :2] + boxes.xyxy[:, 2:]) / 2).unsqueeze(1),
            class_id=boxes.class_id,
            confidence=torch.ones((len(boxes.xyxy), 1)),
            image_metadata={
                key: value
                for key, value in boxes.image_metadata.items()
                if key != "workflows_v2_rows"
            },
        )
        return {"predictions": (points, boxes)}


def definition(detector="test/marker_detector@v1", *, nested=False, proposals=False):
    steps = []
    source = "$inputs.image"
    if nested:
        steps.append(
            dict(
                type="v2/absolute_static_crop@v1",
                name="outer",
                image=source,
                x_center=30,
                y_center=25,
                width=40,
                height=30,
            )
        )
        source = "$steps.outer.crops"
    steps.extend(
        [
            dict(
                type=(
                    "test/rollup_regions@v1" if proposals else "test/marker_detector@v1"
                ),
                name="parents",
                image=source,
            ),
            dict(
                type="v2/dynamic_crop@v1",
                name="crop",
                image=source,
                predictions="$steps.parents.predictions",
            ),
            dict(type=detector, name="child", image="$steps.crop.crops"),
            dict(
                type="v2/dimension_collapse@v1",
                name="collect",
                data="$steps.child.predictions",
            ),
            dict(
                type=DetectionsListRollUpBlock.type,
                name="rollup",
                parent_detection="$steps.parents.predictions",
                child_detections="$steps.collect.output",
                confidence_strategy="$inputs.strategy",
                overlap_threshold="$inputs.overlap",
                keypoint_merge_threshold="$inputs.distance",
            ),
        ]
    )
    return {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            *[
                {"type": "WorkflowParameter", "name": name}
                for name in ("strategy", "overlap", "distance")
            ],
        ],
        "steps": steps,
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.rollup.rolled_up_detections",
            },
            {
                "type": "JsonField",
                "name": "zones",
                "selector": "$steps.rollup.crop_zones",
            },
        ],
    }


def execute(workflow, inputs, mode):
    catalogue = regions_catalogue().with_blocks(
        [DetectionsListRollUpBlock, OverlappingRegionsBlock, KeypointMarkerBlock]
    )
    plan = compile_workflow(
        workflow, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()
    try:
        return session.run(
            {"strategy": "mean", "overlap": 0.0, "distance": 10.0, **inputs}
        )
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize(
    "detector",
    [
        "test/marker_detector@v1",
        "test/masked_marker_detector@v1",
        "test/rollup_keypoints@v1",
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_composed_crop_rollup_and_empty_batch(mode, detector, nested):
    pixels = painted_image((60, 80), [(16, 14, 24, 22, 1, 0), (36, 24, 44, 30, 2, 1)])
    result = execute(
        definition(detector, nested=nested),
        {"image": [pixels, painted_image((60, 80), [])]},
        mode,
    )
    first, empty = [row["result"] for row in result.rows()]
    boxes = first[1] if isinstance(first, tuple) else first
    empty_boxes = empty[1] if isinstance(empty, tuple) else empty
    # rows() converts to root; raw output remains in the nested parent's frame.
    assert boxes.xyxy.tolist() == [[16, 14, 24, 22], [36, 24, 44, 30]]
    assert empty_boxes.xyxy.shape == (0, 4)
    raw = result.outputs.data["result"][0]
    raw_boxes = raw[1] if isinstance(raw, tuple) else raw
    assert frame_of(raw_boxes).size_hw == ((30, 40) if nested else (60, 80))
    assert raw_boxes.image_metadata["class_names"] == {0: "box", 1: "disc"}
    if isinstance(first, tuple):
        assert first[0].xy.tolist() == [[[20, 18]], [[40, 27]]]
    elif hasattr(first, "mask"):
        assert int(first.mask.sum()) == 112


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize(
    "detector",
    [
        "test/marker_detector@v1",
        "test/masked_marker_detector@v1",
        "test/rollup_keypoints@v1",
    ],
)
def test_overlapping_crop_duplicate_is_merged(mode, detector):
    pixels = painted_image((40, 40), [(14, 8, 20, 16, 1, 0)])
    result = execute(definition(detector, proposals=True), {"image": [pixels]}, mode)
    prediction = result.rows()[0]["result"]
    boxes = prediction[1] if isinstance(prediction, tuple) else prediction
    assert boxes.xyxy.tolist() == [[14, 8, 20, 16]]
    assert boxes.confidence.tolist() == pytest.approx([0.548])
    assert len(result.rows()[0]["zones"]) == 2
