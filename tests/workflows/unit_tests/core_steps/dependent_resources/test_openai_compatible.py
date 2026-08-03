"""
Dependent-resources discovery tests for the OpenAI-compatible LLM block
(``roboflow_core/openai_compatible@v1``).

The serving provider IS the user-defined endpoint: ``base_url`` (normalized —
trailing slash stripped) is carried as the resource ``provider``, so the same
model name served from two different servers stays distinguishable, and a
selector-fed url makes the declaration report runtime resolution as required.
"""

from inference.core.workflows.core_steps.models.foundation.openai_compatible.v1 import (
    BlockManifest as OpenAICompatibleV1Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def _build_manifest(base_url: str, model_name: str) -> OpenAICompatibleV1Manifest:
    return OpenAICompatibleV1Manifest.model_validate(
        {
            "type": "roboflow_core/openai_compatible@v1",
            "name": "llm",
            "base_url": base_url,
            "model_name": model_name,
            "prompt": "Describe what you see in the image.",
        }
    )


def test_openai_compatible_v1_carries_endpoint_as_provider() -> None:
    manifest = _build_manifest(
        base_url="http://localhost:8000/v1",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="http://localhost:8000/v1",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        ),
    ]


def test_openai_compatible_v1_normalizes_trailing_slash_in_endpoint() -> None:
    manifest = _build_manifest(
        base_url="http://localhost:8000/v1/",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    (resource,) = manifest.discover_dependent_resources()

    assert resource.metadata.provider == "http://localhost:8000/v1"


def test_openai_compatible_v1_distinguishes_same_model_on_different_servers() -> None:
    first = _build_manifest(
        base_url="http://server-a:8000/v1",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    second = _build_manifest(
        base_url="http://server-b:8000/v1",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    (first_resource,) = first.discover_dependent_resources()
    (second_resource,) = second.discover_dependent_resources()

    assert first_resource != second_resource


def test_openai_compatible_v1_selector_fed_url_requires_runtime_resolution() -> None:
    manifest = _build_manifest(
        base_url="$inputs.base_url",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    (resource,) = manifest.discover_dependent_resources()

    assert resource.metadata.provider == "$inputs.base_url"
    assert resource.metadata.requires_runtime_resolution() is True


def test_openai_compatible_v1_returns_selector_fed_model_name_verbatim() -> None:
    manifest = _build_manifest(
        base_url="$inputs.base_url",
        model_name="$inputs.model_name",
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="$inputs.base_url", model_id="$inputs.model_name"),
    ]
