"""Compiled workflow outputs restore root coordinates from typed-only frames.

The fake model writes only the typed ``FRAME_RECORD_KEY`` record, with no
legacy keys. The workflow crops a 29x47 root at x=7..27, y=3..13 and resizes
the 10x20 crop to 5x10, so ``root_xy = 2 * local_xy + (7, 3)``.
"""

import datetime

import pytest
import torch
from pydantic import Field
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image.primitives.resize.v1 import ResizeBlock
from roboflow_workflows.v2.blocks.image.primitives.static_crop.v1 import StaticCropBlock
from roboflow_workflows.v2.data.boundaries import MEDIA_WILDCARD_KIND
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY, FrameRef, frame_of
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import WILDCARD_KIND
from roboflow_workflows.v2.data.predictions import (
    NATIVE_KINDS,
    NATIVE_SEMANTIC_SEGMENTATION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.rows import PREDICTION_ROWS_KIND, PredictionRows
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.errors import WorkflowExecutionError

from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

VIDEO = VideoMetadata(
    video_identifier="camera-7",
    frame_number=12,
    frame_timestamp=datetime.datetime(
        2026, 10, 12, 9, 15, tzinfo=datetime.timezone.utc
    ),
    fps=30.0,
)
ROOT_HW = (29, 47)
LOCAL_BOXES = [[2.0, 1.0, 4.0, 3.0], [0.0, 0.0, 10.0, 5.0]]
ROOT_BOXES = [[11.0, 5.0, 15.0, 9.0], [7.0, 3.0, 27.0, 13.0]]


def _typed_only(image):
    return {FRAME_RECORD_KEY: FrameRef.from_image(image).to_record()}


class _TypedOnlyDetector(Block):
    type = "data_test/typed_only_detector"
    outputs = {
        "detections": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image"),
        "rows": Output(PREDICTION_ROWS_KIND, source="image"),
        "media": Output(WILDCARD_KIND, source="image"),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to detect on.")

    def run(self, *, image):
        detections = Detections(
            xyxy=torch.tensor(LOCAL_BOXES),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([0.9, 0.6]),
            image_metadata=_typed_only(image),
            bboxes_metadata=[{"name": "a"}, {"name": "b"}],
        )
        rows = PredictionRows(
            native=detections, columns={"tracker_id": torch.tensor([100, 101])}
        )
        return {
            "detections": detections,
            "rows": rows,
            "media": {"boxes": detections, "rows": [rows]},
        }


class _TypedOnlySegmenter(Block):
    type = "data_test/typed_only_segmenter"
    outputs = {
        "grid": Output(NATIVE_SEMANTIC_SEGMENTATION_KIND, source="image"),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to segment.")

    def run(self, *, image):
        height, width = image.size_hw
        grid = SemanticSegmentationResult(
            segmentation_map=torch.zeros((height, width), dtype=torch.int64),
            confidence=torch.ones((height, width)),
            image_metadata=_typed_only(image),
        )
        return {"grid": grid}


class _AuthorDetector(Block):
    """Attaches metadata the way the example 03 README teaches."""

    type = "data_test/author_detector"
    outputs = {
        "detections": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image"),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to detect on.")

    def run(self, *, image):
        detections = Detections(
            xyxy=torch.tensor(LOCAL_BOXES),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([0.9, 0.6]),
            image_metadata={
                **image.prediction_metadata(),
                "class_names": {0: "a", 1: "b"},
            },
        )
        return {"detections": detections}


def _plan(model, *, outputs, transformed=True):
    catalogue = Catalogue(
        [StaticCropBlock, ResizeBlock, model],
        kinds=[IMAGE_KIND, MEDIA_WILDCARD_KIND, *NATIVE_KINDS, PREDICTION_ROWS_KIND],
    )
    steps = [
        {
            "type": "roboflow_core/static_crop@v1",
            "name": "crop",
            "image": "$inputs.images",
            "regions": [[7, 3, 27, 13]],
        },
        {
            "type": "roboflow_core/resize@v1",
            "name": "resize",
            "image": "$steps.crop.crops",
            "width": 10,
            "height": 5,
        },
    ]
    model_image = "$steps.resize.image" if transformed else "$inputs.images"
    steps.append({"type": model.type, "name": "model", "image": model_image})
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "images"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector, **options}
            for name, selector, options in outputs
        ],
    }
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


def _root():
    return ImageData.from_tensor(
        torch.zeros((3, *ROOT_HW), dtype=torch.uint8),
        image_id="root",
        video_metadata=VIDEO,
    )


def _expected_root_frame():
    identity = FrameMapping(frame_id="root", frame_size_hw=ROOT_HW)
    return FrameRef(
        image_id="root",
        size_hw=ROOT_HW,
        parent=identity,
        root=identity,
        video_metadata=VIDEO,
    )


def _single(value):
    # Values after static crop are nested one level per root image.
    return value[0] if isinstance(value, list) else value


def test_typed_only_detections_and_rows_reach_the_root_through_the_engine():
    plan = _plan(
        _TypedOnlyDetector,
        outputs=[
            ("own", "$steps.model.detections", {"coordinates_system": "own"}),
            ("detections", "$steps.model.detections", {}),
            ("rows", "$steps.model.rows", {"coordinates_system": "root"}),
        ],
    )

    (row,) = plan.create_session().run({"images": [_root()]}).rows()
    own, detections, rows = (
        _single(row[name]) for name in ("own", "detections", "rows")
    )

    assert own.xyxy.tolist() == LOCAL_BOXES
    assert frame_of(own).size_hw == (5, 10)
    assert frame_of(own).root.offset_xy == (7.0, 3.0)
    assert detections.xyxy.tolist() == ROOT_BOXES
    assert frame_of(detections) == _expected_root_frame()
    assert set(detections.image_metadata) == {FRAME_RECORD_KEY}
    assert rows.native.xyxy.tolist() == ROOT_BOXES
    assert rows.frame == _expected_root_frame()
    assert rows.columns["tracker_id"].tolist() == [100, 101]
    assert [entry["name"] for entry in rows.native.bboxes_metadata] == ["a", "b"]


def test_typed_only_media_wildcard_output_converts_nested_predictions():
    plan = _plan(
        _TypedOnlyDetector,
        outputs=[("media", "$steps.model.media", {"coordinates_system": "parent"})],
    )

    (row,) = plan.create_session().run({"images": [_root()]}).rows()
    media = _single(row["media"])

    assert media["boxes"].xyxy.tolist() == ROOT_BOXES
    assert frame_of(media["boxes"]) == _expected_root_frame()
    (rows,) = media["rows"]
    assert rows.native.xyxy.tolist() == ROOT_BOXES
    assert rows.frame == _expected_root_frame()


def test_typed_only_semantic_grid_of_a_resized_crop_refuses_root_output():
    own_plan = _plan(
        _TypedOnlySegmenter,
        outputs=[("grid", "$steps.model.grid", {"coordinates_system": "own"})],
    )
    root_plan = _plan(
        _TypedOnlySegmenter,
        outputs=[("grid", "$steps.model.grid", {"coordinates_system": "root"})],
    )

    (row,) = own_plan.create_session().run({"images": [_root()]}).rows()
    assert _single(row["grid"]).segmentation_map.shape == (5, 10)
    with pytest.raises(WorkflowExecutionError, match="no lossless") as raised:
        root_plan.create_session().run({"images": [_root()]}).rows()
    assert isinstance(raised.value.__cause__, ValueError)


def test_typed_only_semantic_grid_on_the_root_is_returned_through_the_engine():
    plan = _plan(
        _TypedOnlySegmenter,
        outputs=[("grid", "$steps.model.grid", {"coordinates_system": "root"})],
        transformed=False,
    )

    (row,) = plan.create_session().run({"images": [_root()]}).rows()
    grid = _single(row["grid"])

    assert grid.segmentation_map.shape == ROOT_HW
    assert frame_of(grid) == _expected_root_frame()


def test_author_block_keeps_source_and_time_in_own_and_root_outputs():
    plan = _plan(
        _AuthorDetector,
        outputs=[
            ("own", "$steps.model.detections", {"coordinates_system": "own"}),
            ("root", "$steps.model.detections", {"coordinates_system": "root"}),
        ],
    )

    (row,) = plan.create_session().run({"images": [_root()]}).rows()
    own, root = (_single(row[name]) for name in ("own", "root"))

    assert frame_of(own).video_metadata == VIDEO
    assert frame_of(own).root.offset_xy == (7.0, 3.0)
    assert root.xyxy.tolist() == ROOT_BOXES
    assert frame_of(root) == _expected_root_frame()
    assert root.image_metadata["class_names"] == {0: "a", 1: "b"}
