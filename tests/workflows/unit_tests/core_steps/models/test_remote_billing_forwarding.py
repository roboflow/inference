"""End-to-end guards that an authenticated `countinference=false` reaches the remote server.

A block running in ``StepExecutionMode.REMOTE`` records no usage locally - the
server it calls runs the model and bills it. The opt-out therefore has to travel
as request parameters, and it only works if every layer cooperates: the block
spreads the scope into its ``InferenceConfiguration``, and the SDK serializes
those fields onto the request.

The client is real and the assertions are on the actual outgoing URL, so a
regression in either layer fails here. Both SDK request builders are covered -
``infer()``, used by the Roboflow model blocks, and ``_post_images``, used by
the core-model and VLM blocks - because only the former serialized the
configuration before.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from requests_mock.mocker import Mocker

from inference.core.env import HOSTED_CORE_MODEL_URL, HOSTED_DETECT_URL
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v2 import (
    SegmentAnything3BlockV2,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.usage_tracking import billable_scope
from inference.usage_tracking.billable_scope import billing_suppressed

SERVICE_SECRET = "shared-secret"


@pytest.fixture
def workflow_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    )


@pytest.fixture
def configured_service_secret(monkeypatch) -> str:
    monkeypatch.setattr(billable_scope, "ROBOFLOW_SERVICE_SECRET", SERVICE_SECRET)
    return SERVICE_SECRET


def _run_object_detection(image: WorkflowImageData) -> None:
    block = RoboflowObjectDetectionModelBlockV1(
        model_manager=MagicMock(),
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    block.run(
        images=[image],
        model_id="some/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.4,
        iou_threshold=0.5,
        max_detections=100,
        max_candidates=1000,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )


def _run_sam3(image: WorkflowImageData) -> None:
    block = SegmentAnything3BlockV2(
        model_manager=MagicMock(),
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    block.run(
        images=[image],
        model_id="sam3/sam3_final",
        class_names=["cat"],
        confidence=0.5,
    )


def test_infer_path_forwards_the_opt_out(
    requests_mock: Mocker,
    workflow_image: WorkflowImageData,
    configured_service_secret: str,
) -> None:
    requests_mock.post(
        f"{HOSTED_DETECT_URL}/some/1",
        json={
            "image": {"width": 640, "height": 480},
            "predictions": [],
            "time": 0.1,
        },
    )

    with billing_suppressed(True):
        _run_object_detection(workflow_image)

    sent_url = requests_mock.request_history[0].url.lower()
    assert "countinference=false" in sent_url
    assert f"service_secret={configured_service_secret}" in sent_url


def test_post_images_path_forwards_the_opt_out(
    requests_mock: Mocker,
    workflow_image: WorkflowImageData,
    configured_service_secret: str,
) -> None:
    # The core-model builder used to pass parameters=None, dropping the fields
    # the block had already put on its InferenceConfiguration.
    requests_mock.post(
        f"{HOSTED_CORE_MODEL_URL}/sam3/concept_segment",
        json={"prompt_results": []},
    )

    with billing_suppressed(True):
        _run_sam3(workflow_image)

    sent_url = requests_mock.request_history[0].url.lower()
    assert "countinference=false" in sent_url
    assert f"service_secret={configured_service_secret}" in sent_url


@pytest.mark.parametrize(
    "run_block, url, response",
    [
        (
            _run_object_detection,
            f"{HOSTED_DETECT_URL}/some/1",
            {
                "image": {"width": 640, "height": 480},
                "predictions": [],
                "time": 0.1,
            },
        ),
        (
            _run_sam3,
            f"{HOSTED_CORE_MODEL_URL}/sam3/concept_segment",
            {"prompt_results": []},
        ),
    ],
)
def test_nothing_is_forwarded_for_an_ordinary_billable_caller(
    requests_mock: Mocker,
    workflow_image: WorkflowImageData,
    configured_service_secret: str,
    run_block,
    url: str,
    response: dict,
) -> None:
    # The secret must never leave the process on a request nobody opted out of.
    requests_mock.post(url, json=response)

    run_block(workflow_image)

    sent_url = requests_mock.request_history[0].url.lower()
    assert "countinference" not in sent_url
    assert "service_secret" not in sent_url
