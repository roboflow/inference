"""
Dependent-resources discovery tests for the Moondream2 block
(``roboflow_core/moondream2@v1``) — the model id is held directly in the
``model_version`` field.
"""

from inference.core.workflows.core_steps.models.foundation.moondream2.v1 import (
    BlockManifest as Moondream2V1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_moondream2_v1_declares_default_model_version() -> None:
    manifest = Moondream2V1Manifest.model_validate(
        {
            "type": "roboflow_core/moondream2@v1",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="moondream2/moondream2_2b_jul24"),
    ]


def test_moondream2_v1_declares_explicit_model_version() -> None:
    manifest = Moondream2V1Manifest.model_validate(
        {
            "type": "roboflow_core/moondream2@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "moondream2/moondream2-2b",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="moondream2/moondream2-2b"),
    ]


def test_moondream2_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = Moondream2V1Manifest.model_validate(
        {
            "type": "roboflow_core/moondream2@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
