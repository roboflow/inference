from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.openrouter import OpenRouterResult
from inference.core.workflows.core_steps.models.foundation.zai_vlm.v1 import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_REASONING_EFFORT,
    BlockManifest,
    ZaiVlmBlockV1,
    build_zai_openrouter_prompts,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

# Copied from vlm-exam `_NORMALIZED_XYXY_PROMPT_TEMPLATE` so an edit to the
# block constant fails this test.
EXPECTED_DETECTION_PROMPT = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners as integers between 0 and 1000, "
    "normalized to the image width (x) and height (y). "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: cat, dog"
)


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def _base_run_kwargs(**overrides):
    kwargs = dict(
        images=[_stub_image()],
        model_version=DEFAULT_MODEL_VERSION,
        task_type="caption",
        prompt=None,
        output_structure=None,
        classes=None,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        api_key="rf_key:account",
        privacy_level="deny",
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=None,
        max_concurrent_requests=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_manifest_defaults():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/zai_vlm@v1",
            "name": "glm",
            "images": "$inputs.image",
            "task_type": "caption",
        }
    )
    assert manifest.model_version == "GLM 5V Turbo"
    assert manifest.max_tokens == 2048
    assert manifest.temperature is None
    assert manifest.reasoning_effort == "none"
    assert manifest.privacy_level == "deny"


def test_manifest_rejects_unknown_model_version():
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/zai_vlm@v1",
                "name": "glm",
                "images": "$inputs.image",
                "task_type": "caption",
                "model_version": "GLM 4V",
            }
        )


def test_detection_prompt_is_vlm_exam_normalized_xyxy_template():
    messages = build_zai_openrouter_prompts(
        images=[np.zeros((8, 8, 3), dtype=np.uint8)],
        task_type="object-detection",
        prompt=None,
        output_structure=None,
        classes=["cat", "dog"],
    )
    assert [message["role"] for message in messages[0]] == ["user"]
    content = messages[0][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == EXPECTED_DETECTION_PROMPT


@patch(
    "inference.core.workflows.core_steps.models.foundation.zai_vlm.v1."
    "OpenRouterWorkflowBlockBase.execute_openrouter_batch_with_usage"
)
def test_run_passes_slug_and_disables_reasoning_by_default(mock_or):
    mock_or.return_value = [
        OpenRouterResult(
            content="boxes", reasoning_trace="", input_tokens=20, output_tokens=8
        )
    ]
    block = ZaiVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(task_type="object-detection", classes=["cat"])
    )

    assert mock_or.call_args.kwargs["model"] == "z-ai/glm-5v-turbo"
    assert mock_or.call_args.kwargs["reasoning"] == {"enabled": False}
    assert mock_or.call_args.kwargs["max_tokens"] == 2048
    assert mock_or.call_args.kwargs["temperature"] is None
    assert result == [
        {
            "output": "boxes",
            "classes": ["cat"],
            "thinking": "",
            "input_tokens": 20,
            "output_tokens": 8,
        }
    ]


@patch(
    "inference.core.workflows.core_steps.models.foundation.zai_vlm.v1."
    "OpenRouterWorkflowBlockBase.execute_openrouter_batch_with_usage"
)
def test_run_maps_reasoning_effort_to_openrouter_config(mock_or):
    mock_or.return_value = [
        OpenRouterResult(
            content="answer", reasoning_trace="trace", input_tokens=5, output_tokens=2
        )
    ]
    block = ZaiVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    block.run(**_base_run_kwargs(reasoning_effort="high"))

    assert mock_or.call_args.kwargs["reasoning"] == {"effort": "high"}
