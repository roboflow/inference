"""Regression tests for WORKFLOWS_REMOTE_API_KEY_TRANSPORT in remote execution.

Guards against the transport selection being discarded by a later
`client.configure(...)` call inside `run_remotely` (PR #2810 review finding):
the client is real and the assertion is on the actual outgoing request, so a
reordering that wipes the transport fails here.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from requests_mock.mocker import Mocker

from inference.core.env import HOSTED_DETECT_URL
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.fixture
def workflow_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    )


def test_remote_object_detection_sends_bearer_header_under_default_transport(
    requests_mock: Mocker,
    workflow_image: WorkflowImageData,
) -> None:
    # given - default WORKFLOWS_REMOTE_API_KEY_TRANSPORT is "both": the legacy
    # v0 query param must stay AND the Authorization header must be attached.
    # The header assertion is the regression guard: it fails whenever the
    # transport selection is applied before (and therefore wiped by) the
    # client.configure(...) call inside run_remotely.
    requests_mock.post(
        f"{HOSTED_DETECT_URL}/some/1",
        json={
            "image": {"width": 640, "height": 480},
            "predictions": [],
            "time": 0.1,
        },
    )
    block = RoboflowObjectDetectionModelBlockV1(
        model_manager=MagicMock(),
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    # when
    _ = block.run(
        images=[workflow_image],
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

    # then
    sent_request = requests_mock.request_history[0]
    assert (
        "api_key=test_api_key" in sent_request.url
    ), "legacy v0 query param must be preserved in 'both' mode"
    assert (
        sent_request.headers["Authorization"] == "Bearer test_api_key"
    ), "Authorization header must survive client.configure() in run_remotely"
