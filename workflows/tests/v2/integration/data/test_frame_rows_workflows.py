"""Compiled crop -> resize -> rows -> select keeps frames and aligned rows."""

import json

import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.image.primitives.resize.v1 import ResizeBlock
from roboflow_workflows.v2.blocks.image.primitives.static_crop.v1 import StaticCropBlock
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.data.rows import PREDICTION_ROWS_KIND, PredictionRows
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)

from inference_models.models.base.object_detection import Detections

# Rows per root image: an empty, a single and a three-row result.
ROW_COUNTS = {"empty": 0, "single": 1, "triple": 3}


class _FakeDetector(Block):
    type = "data_test/fake_rows_detector"
    outputs = {"rows": Output(PREDICTION_ROWS_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to detect on.")

    def run(self, *, image):
        count = ROW_COUNTS[image.root.frame_id]
        offsets = torch.arange(count, dtype=torch.float32)[:, None]
        detections = Detections(
            xyxy=torch.tensor([[2.0, 1.0, 4.0, 3.0]]) + offsets,
            class_id=torch.arange(count),
            confidence=torch.full((count,), 0.5),
            image_metadata=FrameRef.from_image(image).prediction_metadata(),
            bboxes_metadata=[{"name": f"row{k}"} for k in range(count)],
        )
        rows = PredictionRows(
            native=detections, columns={"tracker_id": torch.arange(count) + 100}
        )
        return {"rows": rows}


class _SelectTwoZeroTwo(Block):
    type = "data_test/select_two_zero_two"
    outputs = {"rows": Output(PREDICTION_ROWS_KIND, source="rows")}

    class Params(BlockParams):
        rows: Ref(PREDICTION_ROWS_KIND) = Field(description="Rows to select.")

    def run(self, *, rows):
        indices = [index for index in (2, 0, 2) if index < len(rows)]
        return {"rows": rows.select(indices=indices)}


def _plan():
    catalogue = Catalogue(
        [StaticCropBlock, ResizeBlock, _FakeDetector, _SelectTwoZeroTwo],
        kinds=[IMAGE_KIND, *NATIVE_KINDS, PREDICTION_ROWS_KIND],
    )
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "images"}],
        "steps": [
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
            {
                "type": _FakeDetector.type,
                "name": "detect",
                "image": "$steps.resize.image",
            },
            {
                "type": _SelectTwoZeroTwo.type,
                "name": "select",
                "rows": "$steps.detect.rows",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "own",
                "selector": "$steps.select.rows",
                "coordinates_system": "own",
            },
            {"type": "JsonField", "name": "root", "selector": "$steps.select.rows"},
        ],
    }
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


def _roots():
    sizes = {"empty": (20, 30), "single": (29, 47), "triple": (40, 61)}
    roots = [
        ImageData.from_tensor(torch.zeros((3, *size), dtype=torch.uint8), image_id=name)
        for name, size in sizes.items()
    ]

    return roots


def test_ragged_rows_keep_frames_order_repeats_and_columns_through_the_engine():
    result = _plan().create_session().run({"images": _roots()})
    rows = result.rows()

    (empty_own,), (single_own,), (triple_own,) = (row["own"] for row in rows)
    (empty_root,), (single_root,), (triple_root,) = (row["root"] for row in rows)

    assert len(empty_own) == 0
    assert empty_own.native.xyxy.shape == (0, 4)
    assert empty_own.columns["tracker_id"].shape == (0,)
    assert len(empty_root) == 0

    assert single_own.columns["tracker_id"].tolist() == [100]
    assert single_root.native.xyxy.tolist() == [[11.0, 5.0, 15.0, 9.0]]

    assert triple_own.native.xyxy.tolist() == [
        [4.0, 3.0, 6.0, 5.0],
        [2.0, 1.0, 4.0, 3.0],
        [4.0, 3.0, 6.0, 5.0],
    ]
    assert triple_own.columns["tracker_id"].tolist() == [102, 100, 102]
    assert [row["name"] for row in triple_own.native.bboxes_metadata] == [
        "row2",
        "row0",
        "row2",
    ]
    # local (x, y) -> root (2x + 7, 2y + 3)
    assert triple_root.native.xyxy.tolist() == [
        [15.0, 9.0, 19.0, 13.0],
        [11.0, 5.0, 15.0, 9.0],
        [15.0, 9.0, 19.0, 13.0],
    ]
    assert triple_root.columns["tracker_id"] is triple_own.columns["tracker_id"]

    frame = triple_own.frame
    assert frame.size_hw == (5, 10)
    assert frame.parent.frame_size_hw == (10, 20)
    assert frame.root.frame_id == "triple"
    assert frame.root.frame_size_hw == (40, 61)
    assert frame.root.map_xy(2, 1) == (11.0, 5.0)
    assert single_own.frame.root.frame_size_hw == (29, 47)
    assert triple_root.frame.image_id == "triple"
    assert triple_root.frame.size_hw == (40, 61)


def test_serialized_rows_are_json_and_keep_columns_and_frame():
    result = _plan().create_session().run({"images": _roots()})
    triple_own = result.rows()[2]["own"][0]

    wire = json.loads(json.dumps(result.rows(serialize=True)[2]["own"][0]))
    restored = PREDICTION_ROWS_KIND.deserialize(wire)

    assert restored.columns["tracker_id"].tolist() == [102, 100, 102]
    assert restored.frame == triple_own.frame
