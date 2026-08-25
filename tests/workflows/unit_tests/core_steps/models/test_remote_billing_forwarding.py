"""End-to-end guards that an authenticated `countinference=false` reaches the remote server.

A block running in ``StepExecutionMode.REMOTE`` records no usage locally - the
server it calls runs the model and bills it. The opt-out therefore has to travel
as request parameters, and it only works if the SDK cooperates: a bare
``InferenceHTTPClient`` forwards the outbound forwarding-authority context (set
by the usage decorator for a call it proved carries an authenticated opt-out)
with no per-block code at all.

The client is real and the assertions are on the actual outgoing URL, so a
regression in the SDK's forwarding fails here. One case per distinct SDK request
builder reachable from a block - ``infer()``, ``_post_images``, ``clip_compare``
- since each builds its request by hand and any of them can forget the
parameters. ``infer_lmm`` shares ``_post_images`` but is covered too, because
the block behind it configures its client differently from every other block.
"""

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import MagicMock

import numpy as np
import pytest
from requests_mock.mocker import Mocker

from inference.core.env import HOSTED_CORE_MODEL_URL, HOSTED_DETECT_URL
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v2 import (
    ClipComparisonBlockV2,
)
from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v2 import (
    QwenVlmBlockV2,
)
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
from inference_sdk.config import outbound_service_secret

SERVICE_SECRET = "shared-secret"
QWEN_MODEL_ID = "qwen/qwen2.5-vl-7b"


@pytest.fixture
def workflow_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    )


@contextmanager
def outbound_authority_granted(service_secret: str) -> Iterator[None]:
    """Publish the SDK's outbound forwarding authority for the scope.

    Exactly what the usage decorator does for a call it proved carries an
    authenticated `countinference=false` - a bare `InferenceHTTPClient` built
    anywhere inside the scope forwards it with no code of its own.
    """
    token = outbound_service_secret.set(service_secret)
    try:
        yield
    finally:
        outbound_service_secret.reset(token)


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


def _run_clip_comparison(image: WorkflowImageData) -> None:
    block = ClipComparisonBlockV2(
        model_manager=MagicMock(),
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    block.run(images=[image], classes=["cat"], version="ViT-B-16")


def _run_qwen_vlm(image: WorkflowImageData) -> None:
    block = QwenVlmBlockV2(
        model_manager=MagicMock(),
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    block._run_native_remotely(
        images=[image],
        model_id=QWEN_MODEL_ID,
        combined_prompt="what is this",
        enable_thinking=False,
        max_new_tokens=None,
    )


BUILDER_CASES = [
    pytest.param(
        _run_object_detection,
        f"{HOSTED_DETECT_URL}/some/1",
        {"image": {"width": 640, "height": 480}, "predictions": [], "time": 0.1},
        id="infer",
    ),
    pytest.param(
        _run_sam3,
        f"{HOSTED_CORE_MODEL_URL}/sam3/concept_segment",
        {"prompt_results": []},
        id="post_images",
    ),
    pytest.param(
        _run_clip_comparison,
        f"{HOSTED_CORE_MODEL_URL}/clip/compare",
        {"similarity": [0.5]},
        id="clip_compare",
    ),
    pytest.param(
        _run_qwen_vlm,
        f"{HOSTED_CORE_MODEL_URL}/infer/lmm/{QWEN_MODEL_ID}",
        {"response": "a cat"},
        id="infer_lmm",
    ),
]


@pytest.mark.parametrize("run_block, url, response", BUILDER_CASES)
def test_the_opt_out_reaches_the_server_that_bills_the_model(
    requests_mock: Mocker,
    workflow_image: WorkflowImageData,
    run_block,
    url: str,
    response: dict,
) -> None:
    requests_mock.post(url, json=response)

    with outbound_authority_granted(SERVICE_SECRET):
        run_block(workflow_image)

    sent_url = requests_mock.request_history[0].url.lower()
    assert "countinference=false" in sent_url
    assert f"service_secret={SERVICE_SECRET}" in sent_url


@pytest.mark.parametrize("run_block, url, response", BUILDER_CASES)
def test_nothing_is_forwarded_for_an_ordinary_billable_caller(
    requests_mock: Mocker,
    workflow_image: WorkflowImageData,
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
