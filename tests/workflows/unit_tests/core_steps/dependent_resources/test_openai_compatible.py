"""
Dependent-resources discovery tests for the OpenAI-compatible LLM block
(``roboflow_core/openai_compatible@v1``).

The serving endpoint is user-defined via ``base_url``, so the provider is
the generic ``openai_compatible`` and ``model_name`` is returned verbatim —
for both literal values and selectors.
"""

from inference.core.workflows.core_steps.models.foundation.openai_compatible.v1 import (
    BlockManifest as OpenAICompatibleV1Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def test_openai_compatible_v1_declares_model_name_verbatim() -> None:
    manifest = OpenAICompatibleV1Manifest.model_validate(
        {
            "type": "roboflow_core/openai_compatible@v1",
            "name": "llm",
            "base_url": "http://localhost:8000/v1",
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "prompt": "Describe what you see in the image.",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="openai_compatible", model_id="Qwen/Qwen2.5-VL-7B-Instruct"
        ),
    ]


def test_openai_compatible_v1_returns_selector_fed_model_name_verbatim() -> None:
    manifest = OpenAICompatibleV1Manifest.model_validate(
        {
            "type": "roboflow_core/openai_compatible@v1",
            "name": "llm",
            "base_url": "$inputs.base_url",
            "model_name": "$inputs.model_name",
            "prompt": "Describe what you see in the image.",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai_compatible", model_id="$inputs.model_name"),
    ]
