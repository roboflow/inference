"""
Dependent-resources discovery tests for the Cosmos 3 Edge block
(``roboflow_core/cosmos3_edge@v1``) — the model id is held directly in the
``model_version`` field (literal restricted to ``nvidia/cosmos-3-edge``).
"""

from inference.core.workflows.core_steps.models.foundation.cosmos3.v1 import (
    BlockManifest as Cosmos3V1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_cosmos3_v1_declares_default_model_version() -> None:
    manifest = Cosmos3V1Manifest.model_validate(
        {
            "type": "roboflow_core/cosmos3_edge@v1",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="nvidia/cosmos-3-edge"),
    ]


def test_cosmos3_v1_declares_explicit_model_version() -> None:
    manifest = Cosmos3V1Manifest.model_validate(
        {
            "type": "roboflow_core/cosmos3_edge@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "nvidia/cosmos-3-edge",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="nvidia/cosmos-3-edge"),
    ]


def test_cosmos3_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = Cosmos3V1Manifest.model_validate(
        {
            "type": "roboflow_core/cosmos3_edge@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
