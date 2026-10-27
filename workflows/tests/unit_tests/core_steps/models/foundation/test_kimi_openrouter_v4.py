"""Regression coverage for Kimi v4 coordinate and reasoning contracts."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from roboflow_workflows.core_steps.models.foundation.kimi_openrouter.v4 import (
    BlockManifest,
    KimiOpenrouterBlockV4,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

from tests.unit_tests.core_steps._vlm_prediction_readers import (
    detection_boxes,
    detection_class_ids,
    detection_confidences,
)


def _run(answer, *, model="Kimi K3.1 Pro - OpenRouter", enabled=True, effort="high"):
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((400, 800, 3), dtype=np.uint8),
    )
    with patch.object(
        KimiOpenrouterBlockV4, "_execute_kimi_batch", return_value=[answer]
    ) as execute:
        block = KimiOpenrouterBlockV4(model_manager=MagicMock(), api_key="test")
        result = block.run(
            images=[image],
            task_type="object-detection",
            prompt=None,
            output_structure=None,
            classes=["cat"],
            api_key="test",
            privacy_level="deny",
            model_version=model,
            max_tokens=8192,
            temperature=None,
            max_concurrent_requests=1,
            reasoning_enabled=enabled,
            reasoning_effort=effort,
        )[0]
    return result, execute.call_args.kwargs


def test_xyxy_axis_order_scaling_and_prompt():
    result, request = _run(
        json.dumps(
            [{"label": "cat", "box_2d": [100, 250, 500, 750], "confidence": 0.7}]
        )
    )
    assert not result["error_status"]
    np.testing.assert_array_equal(
        detection_boxes(result["predictions"]), [[80, 100, 400, 300]]
    )
    prompt = request["prompts"][0][0]["content"][0]["text"]
    assert "[x_min, y_min, x_max, y_max]" in prompt
    assert "0 and 1000" in prompt
    assert "confidence" in prompt
    np.testing.assert_allclose(detection_confidences(result["predictions"]), [0.7])
    assert result["inference_id"]


@pytest.mark.parametrize(
    "answer, error", [("[]", False), ("not json", True), ('[{"label":"cat"}]', True)]
)
def test_empty_and_malformed_detections(answer, error):
    result, _ = _run(answer)
    assert result["error_status"] is error
    if not error:
        assert len(detection_boxes(result["predictions"])) == 0


def test_unknown_labels_follow_shared_decoder_contract():
    result, _ = _run(
        '[{"label":"unknown","box_2d":[100,250,500,750],"confidence":0.7}]'
    )
    assert not result["error_status"]
    np.testing.assert_array_equal(detection_class_ids(result["predictions"]), [-1])


@pytest.mark.parametrize("enabled", [True, False])
def test_k26_uses_toggle_not_effort(enabled):
    _, request = _run("[]", model="Kimi K2.6 - OpenRouter", enabled=enabled)
    assert request["reasoning"] == {"enabled": enabled}


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
def test_k31_uses_supported_effort(effort):
    _, request = _run("[]", effort=effort)
    assert request["reasoning"] == {"effort": effort}


def test_v4_excludes_k25_without_changing_v3():
    base = dict(name="kimi", images="$inputs.image", task_type="caption")
    with pytest.raises(ValueError):
        BlockManifest.model_validate(
            dict(
                **base,
                type="roboflow_core/kimi_openrouter@v4",
                model_version="Kimi K2.5 - OpenRouter"
            )
        )
    from roboflow_workflows.core_steps.models.foundation.kimi_openrouter.v3 import (
        BlockManifest as LegacyManifest,
    )

    assert LegacyManifest.model_validate(
        dict(
            **base,
            type="roboflow_core/kimi_openrouter@v3",
            model_version="Kimi K2.5 - OpenRouter"
        )
    )


@pytest.mark.parametrize("route", ["direct", "proxy"])
def test_reasoning_rejection_is_not_retried_without_setting(route):
    platform = MagicMock()
    block = KimiOpenrouterBlockV4(
        model_manager=None, api_key="rf-test", platform_client=platform
    )
    with patch(
        "roboflow_workflows.core_steps.models.foundation.kimi_openrouter.v4.OpenAI"
    ) as factory:
        create = factory.return_value.__enter__.return_value.chat.completions.create
        target = platform.post if route == "proxy" else create
        target.side_effect = ValueError("reasoning unsupported")
        with pytest.raises(ValueError, match="reasoning unsupported"):
            block._execute_kimi_request(
                openrouter_api_key="rf_key:account" if route == "proxy" else "test",
                model="moonshotai/kimi-k2.6",
                messages=[],
                max_tokens=8192,
                temperature=None,
                privacy_level="deny",
                reasoning={"enabled": False},
            )
        assert target.call_count == 1
        payload = (
            target.call_args.kwargs["payload"]
            if route == "proxy"
            else target.call_args.kwargs["extra_body"]
        )
        assert payload["reasoning"] == {"enabled": False}
        assert payload["provider"]["only"] == ["moonshotai"]
        assert payload["provider"]["data_collection"] == "deny"
        assert payload["provider"]["allow_fallbacks"] is False


@pytest.mark.parametrize("answer", ["[null]", "[1]", '[{"box_2d":[1,2,3]}]'])
def test_malformed_entries_do_not_crash_workflow(answer):
    result, _ = _run(answer)
    assert result["error_status"]
