"""Compiled native semantic results keep labels, confidence and frame on the wire."""

import json

import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.image.blocks import ResizeBlock
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import NATIVE_SEMANTIC_SEGMENTATION_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.errors import WorkflowExecutionError

from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

KIND = NATIVE_SEMANTIC_SEGMENTATION_KIND


class _FakeSegmenter(Block):
    """Labels pixels by red value modulo 3; confidence is distinct per pixel."""

    type = "data_test/fake_semantic"
    outputs = {"segmentation": Output(KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to segment.")

    def run(self, *, image):
        red = image.tensor_image[0]
        result = SemanticSegmentationResult(
            segmentation_map=(red % 3).to(torch.int64),
            confidence=red.to(torch.float16) / 256 + 0.001,
            image_metadata=FrameRef.from_image(image).prediction_metadata(),
        )
        return {"segmentation": result}


class _Echo(Block):
    type = "data_test/semantic_echo"
    outputs = {"segmentation": Output(KIND, source="segmentation")}

    class Params(BlockParams):
        segmentation: Ref(KIND) = Field(description="Result to forward.")

    def run(self, *, segmentation):
        return {"segmentation": segmentation}


def _catalogue():
    return Catalogue([ResizeBlock, _FakeSegmenter, _Echo], kinds=[IMAGE_KIND, KIND])


def _segmenting(*, resize, coordinates_system):
    source = "$inputs.images"
    steps = []
    if resize:
        steps.append(
            {
                "type": "v2/resize",
                "name": "resize",
                "image": source,
                "width": 3,
                "height": 2,
            }
        )
        source = "$steps.resize.image"
    steps.append({"type": _FakeSegmenter.type, "name": "segment", "image": source})
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "images"}],
        "steps": steps,
        "outputs": [
            {
                "type": "JsonField",
                "name": "segmentation",
                "selector": "$steps.segment.segmentation",
                "coordinates_system": coordinates_system,
            }
        ],
    }
    plan = compile_workflow(definition, catalogue=_catalogue())

    return plan


def _echo():
    definition = {
        "version": "2.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "segmentations",
                "kind": [KIND.name],
            }
        ],
        "steps": [
            {
                "type": _Echo.type,
                "name": "echo",
                "segmentation": "$inputs.segmentations",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "segmentation",
                "selector": "$steps.echo.segmentation",
                "coordinates_system": "own",
            }
        ],
    }
    plan = compile_workflow(definition, catalogue=_catalogue())

    return plan


def _image():
    red = torch.tensor([[0, 1, 2, 3], [5, 0, 7, 9], [2, 2, 0, 4]], dtype=torch.uint8)
    pixels = torch.stack([red, red, red])
    return ImageData.from_tensor(pixels, image_id="frame")


def test_semantic_result_round_trips_through_serialized_workflow_outputs():
    plan = _segmenting(resize=False, coordinates_system="parent")
    result = plan.create_session().run({"images": [_image()]})
    original = result.rows()[0]["segmentation"]

    wire = json.loads(json.dumps(result.rows(serialize=True)[0]["segmentation"]))
    restored = (
        _echo()
        .create_session()
        .run({"segmentations": [wire]})
        .rows()[0]["segmentation"]
    )

    assert original.segmentation_map.tolist() == [
        [0, 1, 2, 0],
        [2, 0, 1, 0],
        [2, 2, 0, 1],
    ]
    assert torch.equal(restored.segmentation_map, original.segmentation_map)
    assert restored.segmentation_map.dtype == torch.int64
    assert torch.equal(restored.confidence, original.confidence)
    assert restored.confidence.dtype == torch.float16
    assert frame_of(restored) == frame_of(original)
    assert frame_of(restored).image_id == "frame"


def test_resized_semantic_grid_stays_own_and_refuses_silent_root_projection():
    image = _image()
    own = (
        _segmenting(resize=True, coordinates_system="own")
        .create_session()
        .run({"images": [image]})
        .rows()[0]["segmentation"]
    )

    assert own.segmentation_map.shape == (2, 3)
    assert frame_of(own).root.frame_id == "frame"
    refused = (
        _segmenting(resize=True, coordinates_system="parent")
        .create_session()
        .run({"images": [image]})
    )
    with pytest.raises(WorkflowExecutionError, match="no lossless workflow-root form"):
        refused.rows()
