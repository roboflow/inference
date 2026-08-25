"""Tests for the Anthropic Claude v5 block (v4 + vlm-exam detection contract).

The v1-v3 behavior suite lives in ``test_anthropic_claude.py`` and the
token-usage delta in ``test_anthropic_claude_v4.py``; this file covers the
v5 delta: the absolute-pixel object-detection prompt on a PNG image
pre-resized to Claude's native upload dimensions.
"""

import base64
from typing import List
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5 import (
    BlockManifest,
    encode_image_for_task,
    prepare_object_detection_prompt,
    run_claude_prompting,
)


def test_manifest_parsing_for_object_detection_task() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/anthropic_claude@v5",
        "name": "claude",
        "images": "$inputs.image",
        "task_type": "object-detection",
        "classes": ["cat", "dog"],
        "api_key": "$inputs.api_key",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.task_type == "object-detection"
    assert result.classes == ["cat", "dog"]


def test_manifest_parsing_fails_for_object_detection_without_classes() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/anthropic_claude@v5",
        "name": "claude",
        "images": "$inputs.image",
        "task_type": "object-detection",
        "api_key": "$inputs.api_key",
    }

    # when / then
    with pytest.raises(ValueError):
        BlockManifest.model_validate(raw_manifest)


def test_prepare_object_detection_prompt_matches_vlm_exam_contract() -> None:
    # when
    system_prompt, messages = prepare_object_detection_prompt(
        base64_image="base64-image",
        classes=["cat", "dog"],
        image_width=2212,
        image_height=1659,
    )

    # then - no system prompt, PNG image placed before the text
    assert system_prompt is None
    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == "base64-image"
    assert content[1]["type"] == "text"
    prompt_text = content[1]["text"]
    assert "[x_min, y_min, x_max, y_max]" in prompt_text
    assert "absolute pixel coordinates" in prompt_text
    assert "2212x1659 pixel image" in prompt_text
    assert "Only use these labels: cat, dog" in prompt_text


def test_encode_image_for_task_resizes_detection_images_to_upload_dimensions() -> None:
    # given
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)

    # when
    base64_image, width, height = encode_image_for_task(
        image, task_type="object-detection", max_image_size=1024
    )

    # then - 4000x3000 uploads at 2212x1659; max_image_size is not applied
    assert (width, height) == (2212, 1659)
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(base64_image), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape[:2] == (1659, 2212)


def test_encode_image_for_task_keeps_small_detection_images_unchanged() -> None:
    # given
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # when
    _, width, height = encode_image_for_task(
        image, task_type="object-detection", max_image_size=1024
    )

    # then
    assert (width, height) == (640, 480)


def test_encode_image_for_task_downscales_other_tasks_to_max_image_size() -> None:
    # given
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)

    # when
    base64_image, width, height = encode_image_for_task(
        image, task_type="unconstrained", max_image_size=1024
    )

    # then - JPEG payload downscaled to the max_image_size limit
    assert (width, height) == (1024, 768)
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(base64_image), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape[:2] == (768, 1024)


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.execute_claude_requests"
)
def test_run_claude_prompting_states_uploaded_dimensions_in_detection_prompt(
    mock_execute: Mock,
) -> None:
    # given
    mock_execute.return_value = [("[]", 10, 2)]
    images = [{"type": "numpy_object", "value": np.zeros((3000, 4000, 3), np.uint8)}]

    # when
    result = run_claude_prompting(
        roboflow_api_key="rf-key",
        images=images,
        task_type="object-detection",
        prompt=None,
        output_structure=None,
        classes=["cat", "dog"],
        anthropic_api_key="sk-ant-test",
        model_version="claude-sonnet-4-5",
        max_tokens=None,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
        max_image_size=1024,
        max_concurrent_requests=None,
    )

    # then
    assert result == [("[]", 10, 2)]
    prompts: List = mock_execute.call_args.kwargs["prompts"]
    assert len(prompts) == 1
    system_prompt, messages = prompts[0]
    assert system_prompt is None
    assert "2212x1659 pixel image" in messages[0]["content"][1]["text"]
