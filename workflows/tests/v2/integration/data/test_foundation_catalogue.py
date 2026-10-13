"""``create_catalogue`` wires native semantic grids and ``PredictionRows``.

Both families travel through the media wildcard (``*``): native ingress keeps
the object, serialization uses each family's declared kind, and output
coordinates follow the family's own hook. A resized semantic grid refuses a
lossy root form; rows convert their boxes and keep their columns.
"""

import json

import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.data.boundaries import MEDIA_WILDCARD_KIND
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import WILDCARD_KIND
from roboflow_workflows.v2.data.predictions import NATIVE_SEMANTIC_SEGMENTATION_KIND
from roboflow_workflows.v2.data.rows import PREDICTION_ROWS_KIND, PredictionRows
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

HOST_COPY_METHODS = ("cpu", "numpy", "tolist", "item")


def _semantic(image):
    height, width = image.tensor_image.shape[1:]
    labels = torch.arange(height * width, dtype=torch.int64).reshape(height, width)
    result = SemanticSegmentationResult(
        segmentation_map=labels % 3,
        confidence=torch.full((height, width), 0.75, dtype=torch.float16),
        image_metadata=FrameRef.from_image(image).prediction_metadata(),
    )

    return result


def _rows(image):
    detections = Detections(
        xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 2.0, 1.0]]),
        class_id=torch.tensor([0, 2]),
        confidence=torch.tensor([0.9, 0.6]),
        image_metadata=FrameRef.from_image(image).prediction_metadata(),
    )
    rows = PredictionRows(
        native=detections, columns={"tracker_id": torch.tensor([7, 9])}
    )

    return rows


class _NativeFamilies(Block):
    type = "foundation_test/native_families"
    outputs = {
        "semantic": Output(WILDCARD_KIND, source="image"),
        "rows": Output(WILDCARD_KIND, source="image"),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to describe.")

    def run(self, *, image):
        return {"semantic": _semantic(image), "rows": _rows(image)}


def _image():
    pixels = torch.zeros((3, 10, 20), dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id="frame")

    return image


def _forwarding():
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "kind": ["*"]}],
        "steps": [],
        "outputs": [
            {
                "type": "JsonField",
                "name": "value",
                "selector": "$inputs.value",
                "coordinates_system": "own",
            }
        ],
    }
    plan = compile_workflow(definition, catalogue=create_catalogue())

    return plan


def _producing(*, resize, coordinates_system, names=("semantic", "rows")):
    source = "$inputs.images"
    steps = []
    if resize:
        steps.append(
            {
                "type": "v2/resize",
                "name": "resize",
                "image": source,
                "width": 10,
                "height": 5,
            }
        )
        source = "$steps.resize.image"
    steps.append({"type": _NativeFamilies.type, "name": "native", "image": source})
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "images"}],
        "steps": steps,
        "outputs": [
            {
                "type": "JsonField",
                "name": name,
                "selector": f"$steps.native.{name}",
                "coordinates_system": coordinates_system,
            }
            for name in names
        ],
    }
    catalogue = create_catalogue().with_blocks([_NativeFamilies])
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


def _raw(result, name):
    (entry,) = result.selections[name].values()

    return result.outputs.data[entry]


def _raw_row(result, name):
    (value,) = _raw(result, name).content

    return value


def _forbid_host_copies(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("tensor copied to the host")

    for method in HOST_COPY_METHODS:
        monkeypatch.setattr(torch.Tensor, method, _refuse)


def _assert_same_semantic(restored, original):
    assert isinstance(restored, SemanticSegmentationResult)
    assert torch.equal(restored.segmentation_map, original.segmentation_map)
    assert restored.segmentation_map.dtype == original.segmentation_map.dtype
    assert torch.equal(restored.confidence, original.confidence)
    assert restored.confidence.dtype == original.confidence.dtype
    assert frame_of(restored) == frame_of(original)


def _assert_same_rows(restored, original):
    assert isinstance(restored, PredictionRows)
    assert torch.equal(restored.native.xyxy, original.native.xyxy)
    assert torch.equal(restored.native.class_id, original.native.class_id)
    assert torch.equal(restored.native.confidence, original.native.confidence)
    assert set(restored.columns) == {"tracker_id"}
    assert torch.equal(restored.columns["tracker_id"], original.columns["tracker_id"])
    assert restored.columns["tracker_id"].dtype == torch.int64
    assert restored.frame == original.frame


FAMILIES = [
    ("semantic", NATIVE_SEMANTIC_SEGMENTATION_KIND, _semantic, _assert_same_semantic),
    ("rows", PREDICTION_ROWS_KIND, _rows, _assert_same_rows),
]


def test_catalogue_registers_both_families_under_their_declared_kinds():
    catalogue = create_catalogue()

    assert catalogue.kinds["native_semantic_segmentation"] is (
        NATIVE_SEMANTIC_SEGMENTATION_KIND
    )
    assert catalogue.kinds["prediction_rows"] is PREDICTION_ROWS_KIND
    assert catalogue.kinds["*"] is MEDIA_WILDCARD_KIND


@pytest.mark.parametrize(
    "kind,make,assert_same",
    [family[1:] for family in FAMILIES],
    ids=[family[0] for family in FAMILIES],
)
def test_media_wildcard_round_trips_each_family_through_its_declared_kind(
    kind, make, assert_same
):
    payload = make(_image())
    session = _forwarding().create_session()

    result = session.run({"value": {"items": [payload]}})
    wire = json.loads(json.dumps(result.rows(serialize=True)[0]["value"]))
    decoded = session.run({"value": wire})

    assert _raw(result, "value")["items"][0] is payload
    (envelope,) = wire["items"]
    assert envelope["__workflows_v2_media__"] == 1
    assert envelope["kind"] == kind.name
    assert envelope["value"] == kind.to_serialized(payload)
    assert_same(_raw(decoded, "value")["items"][0], payload)
    assert decoded.rows(serialize=True)[0]["value"] == wire


def test_media_wildcard_rejects_a_rows_envelope_holding_a_semantic_payload():
    semantic = _semantic(_image())
    wire = {
        "__workflows_v2_media__": 1,
        "kind": "prediction_rows",
        "value": NATIVE_SEMANTIC_SEGMENTATION_KIND.to_serialized(semantic),
    }

    with pytest.raises(WorkflowExecutionError):
        _forwarding().create_session().run({"value": wire})


def test_own_coordinates_keep_both_families_unchanged_in_the_resized_frame():
    result = (
        _producing(resize=True, coordinates_system="own")
        .create_session()
        .run({"images": [_image()]})
    )
    row = result.rows()[0]

    assert row["semantic"] is _raw_row(result, "semantic")
    assert row["rows"] is _raw_row(result, "rows")
    assert row["semantic"].segmentation_map.shape == (5, 10)
    assert frame_of(row["semantic"]).root.frame_id == "frame"
    assert row["rows"].native.xyxy.tolist()[0] == [1.0, 2.0, 3.0, 4.0]


def test_root_coordinates_scale_row_boxes_and_keep_columns():
    result = (
        _producing(resize=True, coordinates_system="parent", names=("rows",))
        .create_session()
        .run({"images": [_image()]})
    )
    raw = _raw_row(result, "rows")

    (row,) = result.rows()

    converted = row["rows"]
    assert isinstance(converted, PredictionRows)
    assert converted.native.xyxy.tolist() == [
        [2.0, 4.0, 6.0, 8.0],
        [0.0, 0.0, 4.0, 2.0],
    ]
    assert converted.columns["tracker_id"] is raw.columns["tracker_id"]
    assert converted.frame.size_hw == (10, 20)
    assert raw.native.xyxy.tolist()[0] == [1.0, 2.0, 3.0, 4.0]


def test_root_coordinates_refuse_a_resized_semantic_grid():
    result = (
        _producing(resize=True, coordinates_system="parent")
        .create_session()
        .run({"images": [_image()]})
    )

    with pytest.raises(WorkflowExecutionError, match="no lossless workflow-root form"):
        result.rows()


def test_root_coordinates_keep_an_unresized_semantic_grid_as_is():
    result = (
        _producing(resize=False, coordinates_system="parent")
        .create_session()
        .run({"images": [_image()]})
    )

    (row,) = result.rows()

    assert row["semantic"] is _raw_row(result, "semantic")


@pytest.mark.parametrize(
    "resize,names,coordinates_system",
    [
        (False, ("semantic", "rows"), "own"),
        (False, ("semantic", "rows"), "parent"),
        (True, ("rows",), "parent"),
    ],
)
def test_runtime_and_output_conversion_copy_no_tensor_to_the_host(
    resize, names, coordinates_system, monkeypatch
):
    plan = _producing(resize=resize, coordinates_system=coordinates_system, names=names)
    session = plan.create_session()
    image = _image()

    _forbid_host_copies(monkeypatch)
    with pytest.raises(AssertionError, match="tensor copied to the host"):
        image.tensor_image.cpu()
    result = session.run({"images": [image]})
    (row,) = result.rows()
    monkeypatch.undo()

    raw_rows = _raw_row(result, "rows")
    assert row["rows"].columns["tracker_id"] is raw_rows.columns["tracker_id"]
    assert row["rows"].native.xyxy.device == raw_rows.native.xyxy.device
    if "semantic" in names:
        raw_semantic = _raw_row(result, "semantic")
        assert row["semantic"].segmentation_map is raw_semantic.segmentation_map
