"""Compiled native-media ingress, nested geometry, filtering and wire boundaries."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import Field
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.predictions import NATIVE_KINDS
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import WILDCARD_KIND

from inference_models.models.base.object_detection import Detections

REPOSITORY = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "development/workflows-2.0").is_dir()
)
EXAMPLE = REPOSITORY / "development/workflows-2.0/03-tensor-native"


def _example_module(name):
    spec = importlib.util.spec_from_file_location(
        f"tensor_native_example_{name}", EXAMPLE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUTHOR_BLOCKS = _example_module("author_blocks")
FIXTURES = _example_module("fixtures")


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


@pytest.mark.parametrize("root_first", [False, True])
def test_nested_geometry_keeps_sparse_groups_and_independent_own_root_rows(root_first):
    definition = json.loads((EXAMPLE / "workflows/nested_geometry.json").read_text())
    if root_first:
        definition["outputs"].sort(key=lambda output: output["name"] != "root")
    roots = [
        ImageData.from_tensor(pixels, image_id=name)
        for pixels, name in zip(
            FIXTURES.make_root_pixels(), ["bright", "filtered", "empty"]
        )
    ]
    plan = compile_workflow(definition, catalogue=AUTHOR_BLOCKS.create_demo_catalogue())
    result = plan.create_session().run({"images": roots})

    crops = _raw(result, "crops")
    assert [group.indices for group in crops] == [
        ((0, 0), (0, 2)),
        ((1, 0), (1, 2)),
        (),
    ]
    assert crops[2].parent_index == (2,)
    assert list(_raw(result, "count")) == [2, 2, 0]
    own_raw = _raw(result, "own")[0][0][0]
    assert own_raw is _raw(result, "root")[0][0][0]
    assert own_raw is _raw(result, "selected")[0][0][0]
    assert own_raw.xyxy.device == roots[0].device
    assert own_raw.xyxy.dtype == torch.float32

    rows = result.rows()
    own = rows[0]["own"][0][0]
    root = rows[0]["root"][0][0]
    assert own is own_raw
    assert root is not own
    assert own.xyxy.tolist() == [[2, 4, 6, 8]]
    assert root.xyxy.tolist() == [[44, 98, 52, 114]]
    assert own.image_metadata["scaling_relative_to_root_parent"] == [0.5, 0.25]
    assert own.image_metadata["root_parent_coordinates"] == [40.0, 82.0]
    assert root.image_metadata["parent_id"] == "bright"
    assert rows[0]["own"][2] == [None]
    assert rows[1]["own"] == [[None], [], [None]]
    assert rows[2]["own"] == []
    assert rows[2]["mosaic"].is_composite
    assert rows[2]["mosaic"].composite_sources == ()
    assert rows[2]["mosaic"].size_hw == (64, 64)

    wire = json.loads(json.dumps(result.rows(serialize=True)))
    forward = compile_workflow(
        _forwarding("object_detection_prediction"),
        catalogue=create_catalogue().with_blocks([_ForwardNative]),
    ).create_session()
    own_copy = forward.run({"value": wire[0]["own"][0][0]}).rows()[0]["value"]
    root_copy = forward.run({"value": wire[0]["root"][0][0]}).rows()[0]["value"]
    assert own_copy.xyxy.tolist() == [[2, 4, 6, 8]]
    assert root_copy.xyxy.tolist() == [[44, 98, 52, 114]]
    assert own_copy.image_metadata == own.image_metadata
    assert root_copy.image_metadata == root.image_metadata
    assert own.bboxes_metadata == root.bboxes_metadata
    assert own_raw.xyxy.tolist() == [[2, 4, 6, 8]]
    assert result.rows()[0]["own"][0][0] is own_raw


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
GALLERY = FIXTURES.make_native_fixtures(GALLERY_IMAGE.prediction_metadata())


@pytest.mark.parametrize(
    "name,kind,payload", GALLERY, ids=[item[0] for item in GALLERY]
)
def test_all_native_carriers_have_faithful_compiled_wire_roundtrips(
    name, kind, payload
):
    plan = compile_workflow(
        _forwarding(kind), catalogue=create_catalogue().with_blocks([_ForwardNative])
    )
    session = plan.create_session()
    result = session.run({"value": payload})
    assert _raw(result, "value") is payload
    wire = json.loads(json.dumps(result.rows(serialize=True)[0]["value"]))
    decoded = session.run({"value": wire})
    assert decoded.rows(serialize=True)[0]["value"] == wire

    restored = _raw(decoded, "value")
    if name == "empty-float16":
        assert restored.shape == (0, 4)
        assert restored.dtype == torch.float16
    if name == "dense-mask":
        assert torch.equal(restored.mask, payload.mask)
        assert restored.image_metadata == payload.image_metadata
    if name in ("disconnected-rle", "disconnected-semantic"):
        assert restored.mask.masks == payload.mask.masks
        assert restored.mask.image_size == (8, 10)
    if name == "keypoints-without-boxes":
        assert restored[1] is None
        assert torch.equal(restored[0].covariance, payload[0].covariance)
        assert torch.equal(
            restored[0].detection_confidence, payload[0].detection_confidence
        )
        assert restored[0].key_points_metadata == payload[0].key_points_metadata
    if name == "empty-boxes":
        assert restored.xyxy.shape == (0, 4)
        assert restored.xyxy.dtype == torch.float16
        assert restored.image_metadata == payload.image_metadata


def test_catalogue_has_complete_native_inventory_and_legacy_array_boundary():
    catalogue = create_catalogue()
    assert len(NATIVE_KINDS) == 11
    assert {kind for _, kind, _ in GALLERY} == {kind.name for kind in NATIVE_KINDS}
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
