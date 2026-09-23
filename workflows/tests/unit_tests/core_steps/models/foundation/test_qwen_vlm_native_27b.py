from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.foundation.qwen_vlm import v2, v3, v4
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData
from roboflow_workflows.prototypes.block import (
    roboflow_platform_model,
    third_party_model,
)


@pytest.fixture(params=[v2, v3, v4], ids=["v2", "v3", "v4"])
def qwen(request):
    return request.param


def _manifest(qwen, **overrides):
    version = qwen.__name__.rsplit(".", 1)[-1]
    return qwen.BlockManifest.model_validate(
        {
            "type": f"roboflow_core/qwen_vlm@{version}",
            "name": "qwen",
            "images": "$inputs.image",
            "backend": "native",
            "model_version": "Qwen 3.8 VL 27B",
            "task_type": "unconstrained",
            "prompt": "Describe the image.",
            **overrides,
        }
    )


def test_native_27b_is_discoverable_in_schema_and_resources(qwen):
    properties = qwen.BlockManifest.model_json_schema()["properties"]
    variants = properties["model_version"]["anyOf"]
    assert any("Qwen 3.8 VL 27B" in variant.get("enum", []) for variant in variants)
    assert (
        "Qwen 3.8 VL 27B"
        in properties["enable_thinking"]["relevant_for"]["model_version"]["values"]
    )
    assert _manifest(qwen).discover_dependent_resources() == [
        roboflow_platform_model(model_id="qwen3_8-27b")
    ]
    resource = _manifest(
        qwen, model_version="$inputs.model_version"
    ).discover_dependent_resources()[0]
    assert resource.metadata.model_id_resolver("Qwen 3.8 VL 27B") == "qwen3_8-27b"


def test_existing_openrouter_27b_keeps_its_backend(qwen):
    manifest = _manifest(
        qwen, backend="openrouter", openrouter_model_version="Qwen 3.8 27B"
    )
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openrouter", model_id="qwen/qwen3.8-27b")
    ]


@pytest.mark.parametrize("mode", [StepExecutionMode.LOCAL, StepExecutionMode.REMOTE])
@pytest.mark.parametrize("enable_thinking", [False, True])
def test_native_27b_routes_to_roboflow_and_preserves_thinking(
    qwen, mode, enable_thinking
):
    manager = MagicMock()
    response = {"response": {"answer": "A cat.", "thinking": "Look at the ears."}}
    manager.run_lmm.return_value = response
    version = qwen.__name__.rsplit(".", 1)[-1]
    block = getattr(qwen, f"QwenVlmBlock{version.upper()}")(
        model_manager=manager, api_key="test-key", step_execution_mode=mode
    )
    image = WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((100, 200, 3), dtype=np.uint8),
    )
    kwargs = _manifest(qwen, enable_thinking=enable_thinking).model_dump(
        exclude={"type", "name"}
    )
    kwargs["images"] = [image]

    with patch.object(qwen, "InferenceHTTPClient") as client_class:
        client = client_class.return_value
        client.infer_lmm.return_value = response
        results = block.run(**kwargs)

    infer = manager.run_lmm if mode == StepExecutionMode.LOCAL else client.infer_lmm
    infer.assert_called_once()
    assert infer.call_args.kwargs["model_id"] == "qwen3_8-27b"
    assert infer.call_args.kwargs["enable_thinking"] is enable_thinking
    assert results[0]["output"] == "A cat."
    assert results[0]["thinking"] == "Look at the ears."
