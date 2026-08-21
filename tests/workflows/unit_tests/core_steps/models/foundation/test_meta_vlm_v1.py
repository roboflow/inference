from unittest.mock import MagicMock, patch

import numpy as np

from inference.core.workflows.core_steps.models.foundation.meta_vlm.v1 import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_REASONING_EFFORT,
    MODEL_VARIANTS,
    BlockManifest,
    MetaVlmBlockV1,
    build_muse_openrouter_prompts,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

# Copied from vlm-exam `_META_FLAT_NORMALIZED_PROMPT_TEMPLATE` so an edit
# to the block constant fails this test.
EXPECTED_DETECTION_PROMPT = (
    "You are an object grounding expert. Detect all objects in this image "
    "matching these labels: cat, dog. Ensure the objects accurately "
    "match the request and do not miss any objects. For each object, answer "
    'in the format {"label": "<name>", "x_min": <int>, "y_min": <int>, '
    '"x_max": <int>, "y_max": <int>}. The coordinates should be in the '
    "0-1000 range. Return a JSON array of results. If you cannot find an "
    "object, omit it from the results."
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
            "type": "roboflow_core/meta_vlm@v1",
            "name": "muse",
            "images": "$inputs.image",
            "task_type": "caption",
        }
    )
    assert manifest.model_version == DEFAULT_MODEL_VERSION
    assert manifest.max_tokens == 2048
    assert manifest.temperature is None
    assert manifest.reasoning_effort == "low"
    assert manifest.privacy_level == "deny"


def test_detection_prompt_is_vlm_exam_meta_flat_template():
    messages = build_muse_openrouter_prompts(
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
    "inference.core.workflows.core_steps.models.foundation.meta_vlm.v1."
    "OpenRouterWorkflowBlockBase.execute_openrouter_batch"
)
def test_run_openrouter_passes_slug_and_low_reasoning(mock_or):
    mock_or.return_value = [("boxes", "trace")]
    block = MetaVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")
    result = block.run(
        **_base_run_kwargs(
            task_type="object-detection",
            classes=["cat"],
            model_version="Muse Glimmer",
        )
    )
    assert mock_or.call_args.kwargs["model"] == MODEL_VARIANTS["Muse Glimmer"]
    assert mock_or.call_args.kwargs["reasoning"] == {"effort": "low"}
    assert mock_or.call_args.kwargs["max_tokens"] == 2048
    assert mock_or.call_args.kwargs["temperature"] is None
    assert result == [{"output": "boxes", "classes": ["cat"], "thinking": "trace"}]
