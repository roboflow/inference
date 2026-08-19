"""
Dependent-resources discovery tests for the OrcaRouter block
(``roboflow_core/orcarouter@v1``).

The block is a named OrcaRouter integration: ``discover_dependent_resources()``
reports every request as a third-party model with provider ``orcarouter`` and
the route as the model id — both for literal routes like ``orcarouter/auto``
and for selector-fed model names.
"""

from inference.core.workflows.core_steps.models.foundation.orcarouter.v1 import (
    BlockManifest as OrcaRouterV1Manifest,
)
from inference.core.workflows.prototypes.block import third_party_model


def test_orcarouter_v1_declares_named_provider_and_route() -> None:
    manifest = OrcaRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/orcarouter@v1",
            "name": "llm",
            "model_name": "orcarouter/auto",
            "prompt": "Summarize the detections.",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="orcarouter",
            model_id="orcarouter/auto",
        ),
    ]


def test_orcarouter_v1_defaults_to_orcarouter_endpoint() -> None:
    manifest = OrcaRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/orcarouter@v1",
            "name": "llm",
            "model_name": "openai/gpt-5.5",
            "prompt": "Hello.",
        }
    )

    assert manifest.base_url == "https://api.orcarouter.ai/v1"


def test_orcarouter_v1_returns_selector_fed_model_name_verbatim() -> None:
    manifest = OrcaRouterV1Manifest.model_validate(
        {
            "type": "roboflow_core/orcarouter@v1",
            "name": "llm",
            "model_name": "$inputs.model_name",
            "prompt": "Hello.",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(
            provider="orcarouter", model_id="$inputs.model_name"
        ),
    ]
