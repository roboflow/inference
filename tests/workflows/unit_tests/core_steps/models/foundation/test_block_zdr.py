from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.models.foundation.google_gemini import (
    v5 as gemini,
)
from inference.core.workflows.core_steps.models.foundation.openai import v6 as openai
from inference.core.workflows.core_steps.models.foundation.openrouter import (
    v2 as openrouter,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def image():
    return WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def manifest(module, block_type, **kwargs):
    return module.BlockManifest.model_validate(
        {
            "type": block_type,
            "name": "model",
            "images": "$inputs.image",
            "task_type": "caption",
            **kwargs,
        }
    )


def run(block, config):
    return block.run(
        images=[image()], **config.model_dump(exclude={"type", "name", "images"})
    )


@pytest.mark.parametrize("zdr", [False, True])
@pytest.mark.parametrize("key", ["sk-test", "rf_key:account", "rf_key:user:test"])
def test_openai_block_controls_response_storage_for_direct_and_proxy(zdr, key):
    config = manifest(
        openai, "roboflow_core/open_ai@v6", api_key=key, zero_data_retention=zdr
    )
    response = MagicMock(status="completed", output_text="ok", usage=None)
    with patch.object(openai, "_get_openai_client") as client, patch.object(
        openai, "post_to_roboflow_api"
    ) as post:
        client.return_value.responses.create.return_value = response
        post.return_value = {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
            ],
        }
        result = run(
            openai.OpenAIBlockV6(model_manager=MagicMock(), api_key="workspace-key"),
            config,
        )
        assert result[0]["output"] == "ok"
        if key.startswith("rf_key:"):
            params = post.call_args.kwargs["payload"]
            client.assert_not_called()
        else:
            params = client.return_value.responses.create.call_args.kwargs
            post.assert_not_called()
        if zdr:
            assert params["store"] is False
        else:
            assert "store" not in params


@pytest.mark.parametrize("zdr", [False, True])
@pytest.mark.parametrize("key", ["google-key", "rf_key:account", "rf_key:user:test"])
def test_gemini_zdr_uses_configured_key_without_unsupported_api_fields(zdr, key):
    config = manifest(
        gemini, "roboflow_core/google_gemini@v5", api_key=key, zero_data_retention=zdr
    )
    response = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]
    }
    with patch.object(gemini.requests, "post") as direct, patch.object(
        gemini, "post_to_roboflow_api"
    ) as proxy, patch.object(
        gemini, "validate_zdr_prompt", wraps=gemini.validate_zdr_prompt
    ) as validate:
        direct.return_value = MagicMock(status_code=200)
        direct.return_value.json.return_value = response
        proxy.return_value = response
        result = run(
            gemini.GoogleGeminiBlockV5(
                model_manager=MagicMock(), api_key="workspace-key"
            ),
            config,
        )
        assert result[0]["output"] == "ok"
        assert validate.call_count == int(zdr)
        if key.startswith("rf_key:"):
            payload = proxy.call_args.kwargs["payload"]
            assert payload["google_api_key"] == key
            direct.assert_not_called()
        else:
            payload = direct.call_args.kwargs["json"]
            assert direct.call_args.kwargs["headers"]["x-goog-api-key"] == key
            proxy.assert_not_called()
        assert "zero_data_retention" not in payload
        assert "store" not in payload


@pytest.mark.parametrize("key", ["google-key", "rf_key:account"])
@pytest.mark.parametrize(
    "extra",
    [
        {"cachedContent": "cachedContents/test"},
        {"cached_content": "cachedContents/test"},
        *[
            {"tools": [{name: {}}]}
            for name in (
                "google_search",
                "googleSearch",
                "google_search_retrieval",
                "googleSearchRetrieval",
                "google_maps",
                "googleMaps",
            )
        ],
    ],
)
def test_gemini_zdr_rejects_retaining_features_before_network(key, extra):
    with patch.object(gemini.requests, "post") as direct, patch.object(
        gemini, "post_to_roboflow_api"
    ) as proxy:
        with pytest.raises(ValueError, match="incompatible with zero data retention"):
            gemini.execute_gemini_request(
                roboflow_api_key="workspace-key",
                google_api_key=key,
                prompt={"contents": {"parts": [{"text": "hello"}]}, **extra},
                model_version="gemini-2.5-pro",
                zero_data_retention=True,
            )
        direct.assert_not_called()
        proxy.assert_not_called()


def test_native_blocks_keep_existing_default_and_expose_zdr_in_schema():
    for module, block_type in [
        (openai, "roboflow_core/open_ai@v6"),
        (gemini, "roboflow_core/google_gemini@v5"),
    ]:
        assert manifest(module, block_type).zero_data_retention is False
        field = module.BlockManifest.model_json_schema()["properties"][
            "zero_data_retention"
        ]
        assert field["type"] == "boolean"
        assert field["title"] == "Zero Data Retention"


def test_openrouter_existing_zdr_option_reaches_shared_executor():
    config = manifest(
        openrouter,
        "roboflow_core/openrouter@v2",
        model_id="openai/gpt-4o",
        privacy_level="zdr",
    )
    block = openrouter.OpenRouterBlockV2(
        model_manager=MagicMock(), api_key="workspace-key"
    )
    with patch.object(
        block, "execute_openrouter_batch_with_usage", return_value=[]
    ) as execute:
        run(block, config)
        assert execute.call_args.kwargs["privacy_level"] == "zdr"
