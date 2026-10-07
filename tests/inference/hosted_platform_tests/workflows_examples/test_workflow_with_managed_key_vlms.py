"""Hosted E2E coverage for the latest VLM blocks on the Roboflow-managed key.

Every block below defaults its `api_key` to `rf_key:account`, so the hosted
platform proxies the provider call and bills Roboflow credits. These tests need
only `HOSTED_PLATFORM_TESTS_API_KEY`, no provider-specific secrets.
"""

from typing import Any, Dict

import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    apply_auth_mode,
)

MANAGED_KEY_VLM_STEPS = {
    "openai_v7": {"type": "roboflow_core/open_ai@v7"},
    "anthropic_claude_v5": {"type": "roboflow_core/anthropic_claude@v5"},
    "google_gemini_v6": {"type": "roboflow_core/google_gemini@v6"},
    "spacexai_v3": {"type": "roboflow_core/spacexai@v3"},
    "qwen_vlm_v4_openrouter": {
        "type": "roboflow_core/qwen_vlm@v4",
        "backend": "openrouter",
    },
    "meta_vlm_v3": {"type": "roboflow_core/meta_vlm@v3"},
    "zai_vlm_v2": {"type": "roboflow_core/zai_vlm@v2"},
    "kimi_openrouter_v3": {"type": "roboflow_core/kimi_openrouter@v3"},
    "google_gemma_v4": {"type": "roboflow_core/google_gemma@v4"},
    "openrouter_v3": {
        "type": "roboflow_core/openrouter@v3",
        "model_id": "openai/gpt-4o-mini",
    },
}


def _build_workflow(step: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    vlm_step = {
        **step,
        "name": "vlm",
        "images": "$inputs.image",
        "task_type": task_type,
        "api_key": "rf_key:account",
    }
    if task_type in {"classification", "object-detection"}:
        vlm_step["classes"] = "$inputs.classes"
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "classes"},
        ],
        "steps": [vlm_step],
        "outputs": [
            {"type": "JsonField", "name": "output", "selector": "$steps.vlm.output"},
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.vlm.predictions",
            },
            {
                "type": "JsonField",
                "name": "error_status",
                "selector": "$steps.vlm.error_status",
            },
        ],
    }


def _run(
    service_url: str,
    auth_mode: str,
    step: Dict[str, Any],
    task_type: str,
    image: np.ndarray,
) -> Dict[str, Any]:
    client = InferenceHTTPClient(api_url=service_url, api_key=ROBOFLOW_API_KEY)
    client = apply_auth_mode(client, auth_mode)
    result = client.run_workflow(
        specification=_build_workflow(step=step, task_type=task_type),
        images={"image": image},
        parameters={"classes": ["cat", "dog"]},
    )
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "output",
        "predictions",
        "error_status",
    }, "Expected all outputs to be delivered"
    return result[0]


@pytest.mark.parametrize(
    "step", MANAGED_KEY_VLM_STEPS.values(), ids=MANAGED_KEY_VLM_STEPS.keys()
)
@pytest.mark.flaky(retries=4, delay=1)
def test_managed_key_vlm_unconstrained(
    object_detection_service_url: str,
    dogs_image: np.ndarray,
    auth_mode: str,
    step: Dict[str, Any],
) -> None:
    # given
    step = {**step, "prompt": "Describe the animals in this image."}

    # when
    result = _run(
        service_url=object_detection_service_url,
        auth_mode=auth_mode,
        step=step,
        task_type="unconstrained",
        image=dogs_image,
    )

    # then
    assert (
        isinstance(result["output"], str) and len(result["output"]) > 0
    ), "Expected non-empty string generated"


@pytest.mark.parametrize(
    "step", MANAGED_KEY_VLM_STEPS.values(), ids=MANAGED_KEY_VLM_STEPS.keys()
)
@pytest.mark.flaky(retries=4, delay=1)
def test_managed_key_vlm_classification(
    object_detection_service_url: str,
    dogs_image: np.ndarray,
    auth_mode: str,
    step: Dict[str, Any],
) -> None:
    # when
    result = _run(
        service_url=object_detection_service_url,
        auth_mode=auth_mode,
        step=step,
        task_type="classification",
        image=dogs_image,
    )

    # then
    assert result["error_status"] is False, "Expected VLM output to be parsed"
    assert result["predictions"]["top"] == "dog"


@pytest.mark.parametrize(
    "step", MANAGED_KEY_VLM_STEPS.values(), ids=MANAGED_KEY_VLM_STEPS.keys()
)
@pytest.mark.flaky(retries=4, delay=1)
def test_managed_key_vlm_object_detection(
    object_detection_service_url: str,
    dogs_image: np.ndarray,
    auth_mode: str,
    step: Dict[str, Any],
) -> None:
    # when
    result = _run(
        service_url=object_detection_service_url,
        auth_mode=auth_mode,
        step=step,
        task_type="object-detection",
        image=dogs_image,
    )

    # then
    assert result["error_status"] is False, "Expected VLM output to be parsed"
    detected_classes = [e["class"] for e in result["predictions"]["predictions"]]
    assert "dog" in detected_classes, "Expected at least one dog to be detected"
    assert set(detected_classes) <= {"cat", "dog"}, "Expected only prompted classes"
