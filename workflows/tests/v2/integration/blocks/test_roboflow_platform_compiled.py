"""R1 platform blocks through the real compiler and session, with fake ports.

The blocks are registered explicitly on top of the V2 catalogue; the shared
catalogue does not list them yet. Ports are in-process fakes, so these tests
prove wiring, demand and failure routing, not HTTP behaviour::

    image ─▶ v2/visual_search_classifier@v1 ─▶ predictions, inference_id
                     │                                    │
                     └─▶ visual_search_port (fake)        ▼
                                          v2/roboflow_custom_metadata@v1
                                                    └─▶ custom_metadata_port (fake)

Every credential is an invented sentinel.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.custom_metadata import (
    RoboflowCustomMetadataBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.blocks.roboflow_platform.visual_search import (
    VisualSearchBlock,
    VisualSearchClassifierBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError

from inference_models.models.base.classification import ClassificationPrediction

SENTINEL = "rf-test-KEY-r1-compiled-7c41"
CATALOGUE = create_catalogue().with_blocks(
    [RoboflowCustomMetadataBlock, VisualSearchBlock, VisualSearchClassifierBlock]
)
CANDIDATE = {
    "id": "ref-7",
    "url": "https://images.test/ref-7.jpg",
    "name": "ref-7.jpg",
    "tags": ["shelf"],
    "user_metadata": {"aisle": 3},
    "score": 1.5,
    "classification": {"class": "apple", "class_id": 1},
}


class FakePlatform:
    """One fake for both ports; keeps calls, never reveals the key."""

    def __init__(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        *,
        add_error: Optional[Exception] = None,
    ) -> None:
        self.results = [CANDIDATE] if results is None else results
        self.add_error = add_error
        self.calls: List[Dict[str, Any]] = []

    def get_workspace(self, *, api_key: Secret) -> str:
        self.calls.append({"operation": "get_workspace", "api_key": api_key})
        return "fake-workspace"

    def add_custom_metadata(self, **arguments: Any) -> None:
        self.calls.append({"operation": "add_custom_metadata", **arguments})
        if self.add_error is not None:
            raise self.add_error

    def search_project_images(self, **arguments: Any) -> Dict[str, Any]:
        self.calls.append({"operation": "search", **arguments})
        return {"results": [dict(result) for result in self.results]}

    def load_candidate_image(self, url: str) -> np.ndarray:
        self.calls.append({"operation": "load", "url": url})
        return np.full((3, 5, 3), 200, dtype=np.uint8)

    def operations(self) -> List[str]:
        return [call["operation"] for call in self.calls]


def _image(image_id: str) -> ImageData:
    pixels = torch.full((3, 16, 24), 90, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _tagging_plan(outputs: List[Dict[str, str]]):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "v2/visual_search_classifier@v1",
                "name": "classify",
                "image": "$inputs.image",
                "target_project": "shelf-refs",
            },
            {
                "type": "RoboflowCustomMetadata",
                "name": "tag",
                "predictions": "$steps.classify.predictions",
                "inference_id": "$steps.classify.inference_id",
                "field_name": "label_source",
                "field_value": "visual_search",
            },
        ],
        "outputs": outputs,
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def _search_plan(outputs: List[Dict[str, str]], **step: Any):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "top_k", "default_value": 2},
        ],
        "steps": [
            {
                "type": "v2/visual_search@v1",
                "name": "search",
                "image": "$inputs.image",
                "workspace": "shop",
                "target_project": "shelf-refs",
                "top_k": "$inputs.top_k",
                **step,
            }
        ],
        "outputs": outputs,
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def _resources(platform: FakePlatform, api_key: Any = SENTINEL, **extra: Any) -> dict:
    resources = {
        "visual_search_port": platform,
        "custom_metadata_port": platform,
        "api_key": api_key,
        **extra,
    }

    return resources


def _chain_text(error: Optional[BaseException]) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def test_classifier_output_is_tagged_with_its_own_inference_id() -> None:
    # given
    plan = _tagging_plan(
        [
            _output("predictions", "$steps.classify.predictions"),
            _output("inference_id", "$steps.classify.inference_id"),
            _output("tagged", "$steps.tag.message"),
        ]
    )
    platform = FakePlatform()
    session = plan.create_session(resources=_resources(platform))

    # when
    rows = session.run({"image": [_image("a"), _image("b")]}).rows()

    # then
    assert len(rows) == 2
    tagged_ids = [
        call["inference_ids"]
        for call in platform.calls
        if call["operation"] == "add_custom_metadata"
    ]
    assert tagged_ids == [[rows[0]["inference_id"]], [rows[1]["inference_id"]]]
    for row in rows:
        assert isinstance(row["predictions"], ClassificationPrediction)
        assert row["predictions"].class_id.tolist() == [1]
        assert row["tagged"] == "Custom metadata upload was successful"
    assert "load" not in platform.operations()


def test_search_outputs_serialize_and_contain_no_credential(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    plan = _search_plan(
        [
            _output("best", "$steps.search.best_candidate"),
            _output("image", "$steps.search.best_candidate_image"),
            _output("tags", "$steps.search.best_candidate_tags"),
        ]
    )
    platform = FakePlatform()
    session = plan.create_session(
        resources=_resources(platform, api_key=Secret(SENTINEL, name="RF_KEY"))
    )

    # when
    with caplog.at_level(logging.DEBUG):
        (row,) = session.run({"image": [_image("q")], "top_k": 4}).rows(serialize=True)

    # then
    assert row["best"]["image_id"] == "ref-7"
    assert row["tags"] == ["shelf"]
    assert row["image"]["type"] == "base64"
    assert row["image"]["image_id"] == "ref-7"
    assert platform.calls[0]["limit"] == 4
    assert platform.operations() == ["search", "load"]
    assert SENTINEL not in repr(row) + caplog.text + repr(platform.calls)


def test_an_undemanded_candidate_image_is_never_downloaded() -> None:
    # given
    plan = _search_plan([_output("best", "$steps.search.best_candidate")])
    platform = FakePlatform()
    session = plan.create_session(resources=_resources(platform))

    # when
    session.run({"image": [_image("q")]})

    # then
    assert platform.operations() == ["search"]


@pytest.mark.parametrize("top_k", ["three", 0, 51])
def test_an_invalid_runtime_top_k_fails_before_any_request(top_k: Any) -> None:
    # given
    plan = _search_plan([_output("best", "$steps.search.best_candidate")])
    platform = FakePlatform()
    session = plan.create_session(resources=_resources(platform))

    # when
    with pytest.raises(StepExecutionError):
        session.run({"image": [_image("q")], "top_k": top_k})

    # then
    assert platform.calls == []


def test_without_host_ports_the_step_fails_with_an_actionable_error() -> None:
    # given
    plan = _search_plan([_output("best", "$steps.search.best_candidate")])
    session = plan.create_session(resources={"api_key": SENTINEL})

    # when
    with pytest.raises(StepExecutionError) as raised:
        session.run({"image": [_image("q")]})

    # then
    text = _chain_text(raised.value)
    assert PlatformUnavailableError.__name__ in text
    assert "'visual_search_port'" in text
    assert SENTINEL not in text


def test_a_replayed_api_key_fails_the_step_before_dispatch() -> None:
    # given
    plan = _tagging_plan([_output("tagged", "$steps.tag.message")])
    platform = FakePlatform()
    session = plan.create_session(
        resources=_resources(platform, api_key=Secret.unavailable("RF_KEY"))
    )

    # when
    with pytest.raises(StepExecutionError) as raised:
        session.run({"image": [_image("q")]})

    # then
    assert SecretUnavailableError.__name__ in _chain_text(raised.value)
    assert platform.calls == []


def test_upload_failure_and_disabled_sinks_reach_outputs() -> None:
    # given
    plan = _tagging_plan(
        [
            _output("status", "$steps.tag.error_status"),
            _output("message", "$steps.tag.message"),
        ]
    )
    failing = FakePlatform(
        add_error=PlatformRequestError("code 503", error_type="builtins.RuntimeError")
    )
    failed_session = plan.create_session(resources=_resources(failing))
    disabled = FakePlatform()
    disabled_session = plan.create_session(
        resources=_resources(disabled, disable_sinks=True)
    )

    # when
    (failed,) = failed_session.run({"image": [_image("q")]}).rows()
    (skipped,) = disabled_session.run({"image": [_image("q")]}).rows()

    # then
    assert failed["status"] is True
    assert "Details: code 503" in failed["message"]
    assert skipped == {
        "status": False,
        "message": "Sink was disabled by workflow execution policy",
    }
    assert "add_custom_metadata" not in disabled.operations()
