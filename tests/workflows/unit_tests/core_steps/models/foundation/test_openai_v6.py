"""Tests for the OpenAI v6 block (v5 + token-usage outputs).

The v5 behavior suite lives in ``test_openai_v5.py``; this file covers the
v6 additions: token-usage outputs on both execution paths and optional
detection criteria that preserve the default Auto Label request.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from inference.core.workflows.core_steps.models.foundation.openai.v6 import (
    BlockManifest,
    _execute_direct_openai_request,
    _execute_proxied_openai_request,
    prepare_object_detection_prompt,
    run_openai_prompting,
)

_OPENAI_OK = {
    "status": "completed",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ],
}


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai.v6.post_to_roboflow_api"
)
def test_proxied_request_returns_usage_and_none_when_omitted(mock_post: Mock) -> None:
    def call():
        return _execute_proxied_openai_request(
            roboflow_api_key="rf_api_key",
            openai_api_key="rf_key:account",
            instructions="test",
            input_content=[],
            model_version="gpt-5.1",
            reasoning_effort=None,
            max_tokens=None,
            temperature=None,
        )

    mock_post.return_value = {
        **_OPENAI_OK,
        "usage": {"input_tokens": 21, "output_tokens": 6},
    }
    assert call() == ("ok", 21, 6)

    mock_post.return_value = _OPENAI_OK
    assert call() == ("ok", None, None)


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai.v6._get_openai_client"
)
def test_direct_request_returns_usage(mock_get_client: Mock) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.output_text = "response"
    mock_response.usage = MagicMock(input_tokens=14, output_tokens=3)
    mock_client.responses.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = _execute_direct_openai_request(
        openai_api_key="sk-test",
        instructions="test",
        input_content=[{"role": "user", "content": []}],
        model_version="gpt-5.1",
        reasoning_effort=None,
        max_tokens=None,
        temperature=None,
    )

    assert result == ("response", 14, 3)


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.1", "gpt-4.1"])
@pytest.mark.parametrize(
    "criteria", [None, "", "Only people wearing helmets, excluding cyclists"]
)
def test_detection_criteria_are_separate_from_labels(model, criteria):
    with (
        patch(
            "inference.core.workflows.core_steps.models.foundation.openai.v6.load_image",
            return_value=(object(), None),
        ),
        patch(
            "inference.core.workflows.core_steps.models.foundation.openai.v6.encode_image_for_task",
            return_value=("encoded", 640, 480),
        ),
        patch(
            "inference.core.workflows.core_steps.models.foundation.openai.v6.execute_openai_requests",
            return_value=[],
        ) as execute,
    ):
        run_openai_prompting(
            roboflow_api_key="test",
            images=[{}],
            task_type="object-detection",
            prompt=None,
            output_structure=None,
            classes=["person"],
            openai_api_key="rf_key:account",
            model_version=model,
            reasoning_effort=None,
            image_detail="auto",
            max_tokens=None,
            temperature=None,
            max_concurrent_requests=1,
            detection_instructions=criteria,
        )
    content = execute.call_args.kwargs["openai_prompts"][0]["input"][0]["content"]
    texts = [item["text"] for item in content if item["type"] == "input_text"]
    assert "person" in texts[0]
    if criteria:
        assert criteria not in texts[0]
        assert criteria in texts[1]
        assert "not class names" in texts[1]
        assert "empty detections list" in texts[1]
    else:
        assert len(texts) == 1
        assert execute.call_args.kwargs["openai_prompts"][
            0
        ] == prepare_object_detection_prompt(
            base64_image="encoded",
            classes=["person"],
            image_width=640,
            image_height=480,
            model_version=model,
        )
    assert any(item["type"] == "input_image" for item in content)


def test_manifest_exposes_optional_detection_instructions():
    manifest = BlockManifest(
        type="roboflow_core/open_ai@v6",
        name="detect",
        images="$inputs.image",
        task_type="object-detection",
        classes="$inputs.classes",
        detection_instructions="$inputs.detection_instructions",
    )
    assert manifest.detection_instructions == "$inputs.detection_instructions"


def test_existing_detection_manifest_does_not_require_instructions():
    manifest = BlockManifest(
        type="roboflow_core/open_ai@v6",
        name="detect",
        images="$inputs.image",
        task_type="object-detection",
        classes=["person"],
    )
    assert manifest.detection_instructions is None
