"""Compiled depth demand, provenance, requests and phase execution."""

import datetime

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.models import depth
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.unit.blocks.models.test_depth import backend_types, make_image


def plan(*, outputs=("normalized_depth", "image"), mode="run"):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {"type": "WorkflowBatchInput", "name": "model", "kind": ["string"]},
            {"type": "WorkflowBatchInput", "name": "package", "kind": ["string"]},
        ],
        "steps": [
            {
                "type": depth.DepthEstimationBlock.type,
                "name": "depth",
                "image": "$inputs.image",
                "model_id": "$inputs.model",
                "model_package_id": "$inputs.package",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": key, "selector": f"$steps.depth.{key}"}
            for key in outputs
        ],
    }
    return compile_workflow(
        definition,
        catalogue=Catalogue([depth.DepthEstimationBlock]),
        options=CompileOptions(block_execution=mode),
    )


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize(
    "outputs", [("normalized_depth",), ("image",), ("normalized_depth", "image")]
)
def test_compiled_demand_and_frame_linkage(backend_types, monkeypatch, mode, outputs):
    original = depth._visualization
    rendered = []

    def visualize(image, *, normalized):
        rendered.append(image.image_id)
        return original(image, normalized=normalized)

    monkeypatch.setattr(depth, "_visualization", visualize)
    model = backend_types[0]()
    session = plan(outputs=outputs, mode=mode).create_session(
        {"models": {"local": model}}
    )
    video = VideoMetadata(
        video_identifier="depth-camera",
        frame_number=23,
        frame_timestamp=datetime.datetime(2026, 10, 13, tzinfo=datetime.timezone.utc),
        fps=10.0,
    )
    root = ImageData.from_tensor(
        torch.arange(48, dtype=torch.uint8).reshape(3, 4, 4),
        image_id="root",
        video_metadata=video,
    )
    cropped = root.crop((1, 1, 3, 3)).resize((3, 4))
    images = [cropped, make_image(image_id="second")]
    rows = session.run(
        {"image": images, "model": ["local", "local"], "package": [None, None]}
    ).rows()
    assert rendered == ([cropped.image_id, "second"] if "image" in outputs else [])
    for image, row in zip(images, rows):
        assert set(row) == set(outputs)
        if "normalized_depth" in outputs:
            actual = row["normalized_depth"]
            raw = image.tensor_image[0].float()
            assert torch.equal(actual, (raw - raw.min()) / (raw.max() - raw.min()))
            assert tuple(actual.shape) == image.size_hw
            assert actual.device == image.device
        if "image" in outputs:
            actual = row["image"]
            assert actual.parent == image.parent
            assert actual.root == image.root
            assert actual.image_id == image.image_id
            assert actual.video_metadata == image.video_metadata
            assert actual.parent_row == image.parent_row
            assert actual.tensor_image.device == model.last_maps[0].device


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_compiled_multiple_requests_shape_batches_and_model_reuse(backend_types, mode):
    loaded, models = [], {}

    def load(request):
        loaded.append(request)
        models[request] = backend_types[
            1 if request.model_package_id == "dav3" else 0
        ]()
        return models[request]

    session = plan(mode=mode).create_session({"models": ModelCache(load)})
    assert loaded == []
    images = [
        make_image((1, 4, 2, 3), image_id="zero"),
        make_image(size=(1, 4), image_id="one"),
        make_image(image_id="two"),
        make_image((4, 1, 3, 2), image_id="three"),
    ]
    for _ in range(2):
        rows = session.run(
            {
                "image": images,
                "model": ["a", "a", "b", "a"],
                "package": ["v2", "v2", "dav3", "v2"],
            }
        ).rows()
        assert [r["image"].image_id for r in rows] == [im.image_id for im in images]
        assert [tuple(r["normalized_depth"].shape) for r in rows] == [
            im.size_hw for im in images
        ]
    assert loaded == [ModelRequest("a", "v2"), ModelRequest("b", "dav3")]
    assert models[loaded[0]].calls.count(("forward", 2)) == 2
    assert models[loaded[0]].calls.count(("forward", 1)) == 2
    assert models[loaded[1]].calls.count(("forward", 1)) == 2
