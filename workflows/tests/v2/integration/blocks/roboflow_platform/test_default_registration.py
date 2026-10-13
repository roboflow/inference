"""Public platform discovery and compiled in-process ports, with no providers."""

import json
import re
import sys

import pytest
import torch
from roboflow_workflows.v2 import blocks
from roboflow_workflows.v2.blocks import roboflow_platform
from roboflow_workflows.v2.blocks.roboflow_platform.vision_events import EventBundle
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import UnknownBlockError
from roboflow_workflows.v2.engine.plan import CompileOptions

_ADDED = (
    ("RoboflowCustomMetadataBlock", "v2/roboflow_custom_metadata@v1"),
    ("VisualSearchBlock", "v2/visual_search@v1"),
    ("VisualSearchClassifierBlock", "v2/visual_search_classifier@v1"),
    ("RoboflowVisionEventsBlock", "v2/roboflow_vision_events@v1"),
    ("VisionEventBundleBlock", "v2/vision_event_bundle@v1"),
)


def _definition():
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "name": "search",
                "type": "v2/visual_search@v1",
                "image": "$inputs.image",
                "workspace": "shop",
                "target_project": "references",
            },
            {
                "name": "classify",
                "type": "v2/visual_search_classifier@v1",
                "image": "$inputs.image",
                "target_project": "references",
            },
            {
                "name": "tag",
                "type": "v2/roboflow_custom_metadata@v1",
                "predictions": "$steps.classify.predictions",
                "inference_id": "$steps.classify.inference_id",
                "field_name": "source",
                "field_value": "reference",
            },
            {
                "name": "event",
                "type": "v2/roboflow_vision_events@v1",
                "event_type": "custom",
                "solution": "use-case",
                "input_image": "$inputs.image",
                "custom_value": "$steps.classify.inference_id",
                "cooldown_seconds": 0,
                "fire_and_forget": False,
            },
            {
                "name": "bundle",
                "type": "v2/vision_event_bundle@v1",
                "event_type": "operator_feedback",
                "related_event_id": "$steps.event.event_id",
                "feedback": "correct",
                "output_image": "$inputs.image",
                "target_directory": "/invented-bundles",
                "file_name": "checked",
                "cooldown_seconds": 0,
                "fire_and_forget": False,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in (
                ("best", "$steps.search.best_candidate"),
                ("id", "$steps.classify.inference_id"),
                ("prediction", "$steps.classify.predictions"),
                ("tagged", "$steps.tag.message"),
                ("event", "$steps.event.event_id"),
                ("bundle", "$steps.bundle.bundle_path"),
            )
        ],
    }

    return definition


class _Port:
    def __init__(self):
        self.calls = []

    def get_workspace(self, *, api_key):
        assert isinstance(api_key, Secret)
        self.calls.append(("workspace", None))
        return "workspace"

    def search_project_images(self, **arguments):
        assert isinstance(arguments["api_key"], Secret)
        self.calls.append(("search", arguments))
        return {
            "results": [
                {
                    "id": "reference",
                    "name": "reference.jpg",
                    "score": 1.5,
                    "classification": {"class": "apple", "class_id": 1},
                }
            ]
        }

    def load_candidate_image(self, url):
        raise AssertionError("An undemanded candidate image must not be downloaded")

    def add_custom_metadata(self, **arguments):
        self.calls.append(("metadata", arguments))

    def submit(self, request, **arguments):
        self.calls.append(("event", request))
        return f"event-{len(self.calls)}"

    def prepare_destination(self, **arguments):
        self.calls.append(("prepare", arguments))

    def create_bundle(self, request, **arguments):
        self.calls.append(("bundle", request))
        return EventBundle(
            event_id="bundle", bundle_path="/invented-bundles/checked.tar.gz"
        )


def _resources(port, **extra):
    resources = {
        "api_key": "invented-final-registration-key",
        "visual_search_port": port,
        "custom_metadata_port": port,
        "vision_events_port": port,
        "vision_event_bundle_port": port,
        **extra,
    }

    return resources


@pytest.mark.parametrize("route", ["create", "plugin", "factory"])
def test_default_exports_and_family_tuple_share_exact_classes(route):
    catalogue = (
        blocks.create_catalogue()
        if route == "create"
        else Catalogue.from_modules(
            ["roboflow_workflows.v2.blocks"],
            **({"attribute": "create_catalogue"} if route == "factory" else {}),
        )
    )
    assert roboflow_platform.BLOCKS is roboflow_platform.PLATFORM_BLOCKS
    assert len(set(roboflow_platform.BLOCKS)) == 8
    for name, identity in _ADDED:
        block = getattr(blocks, name)
        assert getattr(roboflow_platform, name) is block
        assert name in blocks.__all__ and name in roboflow_platform.__all__
        assert roboflow_platform.BLOCKS.count(block) == 1
        assert catalogue.block_types.count(identity) == 1
        assert catalogue.entry(identity).spec is spec_of(block)
    for identity in (
        "v2/asset_library_attributes@v1",
        "v2/roboflow_dataset_upload@v2",
        "v2/model_monitoring_inference_aggregator@v1",
        "v2/action_recognition@v1",
        "v2/cosmos3_edge@v1",
        "v2/florence_2@v2",
        "v2/google_vision_ocr@v1",
        "v2/moondream2@v1",
        "v2/qwen_vlm@v4",
    ):
        assert identity in catalogue.block_types


def test_public_schema_and_compilation_construct_no_block_or_host(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Discovery/schema/compilation must not construct a block")

    for name, identity in _ADDED:
        monkeypatch.setattr(getattr(blocks, name), "__init__", unexpected)
    catalogue = blocks.create_catalogue()
    description = catalogue.describe()
    json.dumps(description)
    descriptions = {row["type"]: row for row in description["blocks"]}
    fields = {
        "RoboflowCustomMetadataBlock": {
            "predictions",
            "field_name",
            "field_value",
            "inference_id",
        },
        "VisualSearchBlock": {"image", "workspace", "target_project"},
        "VisualSearchClassifierBlock": {"image", "target_project", "max_image_size"},
        "RoboflowVisionEventsBlock": {"event_type", "solution", "input_image"},
        "VisionEventBundleBlock": {
            "event_type",
            "related_event_id",
            "target_directory",
            "file_name",
        },
    }
    for name, identity in _ADDED:
        assert fields[name].issubset(
            descriptions[identity]["params_schema"]["properties"]
        )
    compile_workflow(_definition(), catalogue=catalogue)
    assert not any(
        name in sys.modules
        for name in (
            "inference.roboflow_workflows_plugin.v2.platform_adapter",
            "inference.roboflow_workflows_plugin.v2.platform_events_adapter",
        )
    )


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_default_compiled_search_metadata_and_event_chain(mode):
    port = _Port()
    plan = compile_workflow(
        _definition(),
        catalogue=blocks.create_catalogue(),
        options=CompileOptions(block_execution=mode),
    )
    assert port.calls == []
    images = [
        ImageData.from_tensor(
            torch.zeros((3, 16, 24), dtype=torch.uint8), image_id=f"camera-{index}"
        )
        for index in range(2)
    ]
    with plan.create_session(resources=_resources(port)) as session:
        rows = session.run({"image": images}).rows()
    assert len(rows) == 2
    assert all(
        row["best"]["image_id"] == "reference"
        and row["prediction"].class_id.tolist() == [1]
        and row["tagged"] == "Custom metadata upload was successful"
        and row["bundle"] == "/invented-bundles/checked.tar.gz"
        for row in rows
    )
    metadata = [value for name, value in port.calls if name == "metadata"]
    events = [value for name, value in port.calls if name == "event"]
    bundles = [value for name, value in port.calls if name == "bundle"]
    assert [value["inference_ids"] for value in metadata] == [
        [row["id"]] for row in rows
    ]
    assert [value.event_data["value"] for value in events] == [
        row["id"] for row in rows
    ]
    assert [value.event_data["relatedEventId"] for value in bundles] == [
        row["event"] for row in rows
    ]
    assert [value.output_image.image_id for value in bundles] == [
        image.image_id for image in images
    ]
    assert sum(name == "search" for name, value in port.calls) == 4
    assert "invented-final-registration-key" not in repr(rows) + repr(port.calls)


def test_empty_batch_has_no_port_effect():
    port = _Port()
    plan = compile_workflow(_definition(), catalogue=blocks.create_catalogue())
    with plan.create_session(resources=_resources(port)) as session:
        rows = session.run({"image": []}).rows()
    assert rows == [] and port.calls == []


def test_disabled_sinks_leave_search_active_without_sink_effects():
    port = _Port()
    plan = compile_workflow(_definition(), catalogue=blocks.create_catalogue())
    image = ImageData.from_tensor(torch.zeros((3, 16, 24), dtype=torch.uint8))
    with plan.create_session(resources=_resources(port, disable_sinks=True)) as session:
        (row,) = session.run({"image": [image]}).rows()
    assert row["tagged"] == "Sink was disabled by workflow execution policy"
    assert row["event"] == row["bundle"] == ""
    assert sorted(name for name, value in port.calls) == [
        "search",
        "search",
        "workspace",
    ]


def test_retirement_and_q3_deferral_remain_default_rejections():
    from tests.v2.unit.blocks.test_not_ported_v1_types import RETIRED_V2_TYPES

    catalogue = blocks.create_catalogue()
    for identity in (*RETIRED_V2_TYPES, "v2/qwen3_5vl@v2"):
        assert catalogue.find(identity) is None
        definition = {
            "version": "2.0",
            "inputs": [],
            "steps": [{"type": identity, "name": "excluded"}],
            "outputs": [],
        }
        with pytest.raises(UnknownBlockError, match=re.escape(identity)) as error:
            compile_workflow(definition, catalogue=catalogue)
        assert error.value.step_path == ("excluded",)
