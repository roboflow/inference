"""Compiled native-media ingress and wire boundaries."""

import json

import numpy as np
import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import WILDCARD_KIND
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)

from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)


class _ForwardNative(Block):
    type = "native_test/forward"
    outputs = {"value": Output(WILDCARD_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(WILDCARD_KIND) = Field(
            description="Native payload to forward unchanged."
        )

    def run(self, *, value):
        return {"value": value}


def _raw(result, name):
    (entry,) = result.selections[name].values()
    return result.outputs.data[entry]


def _forwarding(kind, *, coordinates="own", nested=False):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "kind": [kind]}],
        "steps": [],
        "outputs": [
            {
                "type": "JsonField",
                "name": "value",
                "selector": "$inputs.value",
                "coordinates_system": coordinates,
            }
        ],
    }
    if nested:
        child = _forwarding(kind, coordinates="own")
        child["steps"] = [
            {"type": "native_test/forward", "name": "forward", "value": "$inputs.value"}
        ]
        child["outputs"][0]["selector"] = "$steps.forward.value"
        definition["steps"] = [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {"value": "$inputs.value"},
            }
        ]
        definition["outputs"][0]["selector"] = "$steps.child.value"
    return definition


@pytest.mark.parametrize(
    "device",
    ["cpu"]
    + (["mps"] if torch.backends.mps.is_available() else [])
    + (["cuda"] if torch.cuda.is_available() else []),
)
def test_nested_native_ingress_preserves_identity_device_and_storage_without_host_export(
    monkeypatch, device
):
    image = ImageData.from_tensor(
        torch.zeros((3, 12, 16), dtype=torch.uint8, device=device), image_id="native"
    )
    prediction = Detections(
        xyxy=torch.tensor([[1, 2, 3, 4]], dtype=torch.float32, device=device),
        class_id=torch.tensor([0], device=device),
        confidence=torch.tensor([0.9], device=device),
        image_metadata=image.prediction_metadata(),
    )
    payloads = [
        ("image", image),
        ("object_detection_prediction", prediction),
        ("tensor", image.tensor_image),
    ]

    def reject_export(*args, **kwargs):
        raise AssertionError(
            "Native forwarding must not materialize host pixels or predictions"
        )

    monkeypatch.setattr(torch.Tensor, "cpu", reject_export)
    monkeypatch.setattr(torch.Tensor, "numpy", reject_export)
    for kind, payload in payloads:
        plan = compile_workflow(
            _forwarding(kind, nested=True),
            catalogue=create_catalogue().with_blocks([_ForwardNative]),
        )
        result = plan.create_session().run({"value": payload})
        assert _raw(result, "value") is payload
        assert result.rows()[0]["value"] is payload
    assert image.tensor_image.device.type == device
    assert prediction.xyxy.device.type == device


GALLERY_IMAGE = ImageData.from_tensor(
    torch.zeros((3, 8, 10), dtype=torch.uint8), image_id="gallery"
)


def test_semantic_grid_has_faithful_compiled_wire_roundtrip():
    payload = SemanticSegmentationResult(
        segmentation_map=torch.arange(80, dtype=torch.int64).reshape(8, 10) % 3,
        confidence=torch.full((8, 10), 0.5, dtype=torch.float16),
        image_metadata=GALLERY_IMAGE.prediction_metadata(),
    )
    plan = compile_workflow(
        _forwarding("native_semantic_segmentation"),
        catalogue=create_catalogue().with_blocks([_ForwardNative]),
    )
    session = plan.create_session()
    result = session.run({"value": payload})
    assert _raw(result, "value") is payload
    wire = json.loads(json.dumps(result.rows(serialize=True)[0]["value"]))
    decoded = session.run({"value": wire})
    assert decoded.rows(serialize=True)[0]["value"] == wire

    restored = _raw(decoded, "value")
    assert torch.equal(restored.segmentation_map, payload.segmentation_map)
    assert torch.equal(restored.confidence, payload.confidence)
    assert restored.confidence.dtype == torch.float16
    assert restored.image_metadata == payload.image_metadata


def test_catalogue_has_complete_native_inventory_and_legacy_array_boundary():
    catalogue = create_catalogue()
    assert len(NATIVE_KINDS) == 12
    for kind in (*NATIVE_KINDS,):
        assert catalogue.kinds[kind.name] is kind
    plan = compile_workflow(_forwarding("numpy_array"), catalogue=catalogue)
    result = plan.create_session().run({"value": np.empty((0, 8), dtype=np.float16)})
    native = _raw(result, "value")
    assert isinstance(native, torch.Tensor)
    assert native.shape == (0, 8)
    assert native.dtype == torch.float16
    wire = json.loads(json.dumps(result.rows(serialize=True)[0]["value"]))
    restored = _raw(plan.create_session().run({"value": wire}), "value")
    assert restored.shape == native.shape
    assert restored.dtype == native.dtype


def test_media_wildcard_serializes_native_values_inside_ordinary_containers():
    source = {
        "image": GALLERY_IMAGE,
        "empty": torch.empty((0, 4), dtype=torch.float16),
        "label": "ordinary",
    }
    plan = compile_workflow(
        _forwarding("*", nested=True),
        catalogue=create_catalogue().with_blocks([_ForwardNative]),
    )
    result = plan.create_session().run({"value": source})
    assert _raw(result, "value") is source
    wire = json.loads(json.dumps(result.rows(serialize=True)[0]["value"]))
    restored = _raw(plan.create_session().run({"value": wire}), "value")
    assert restored["image"].image_id == "gallery"
    assert restored["empty"].shape == (0, 4)
    assert restored["empty"].dtype == torch.float16
    assert restored["label"] == "ordinary"
