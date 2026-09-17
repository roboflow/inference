"""
Dependent-resources discovery tests for the SmolVLM2 block
(``roboflow_core/smolvlm2@v1``) — the model id is held directly in the
``model_version`` field.
"""

from inference.core.workflows.core_steps.models.foundation.smolvlm.v1 import (
    BlockManifest as SmolVLM2V1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_smolvlm2_v1_declares_default_model_version() -> None:
    manifest = SmolVLM2V1Manifest.model_validate(
        {
            "type": "roboflow_core/smolvlm2@v1",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="smolvlm2/smolvlm-2.2b-instruct"),
    ]


def test_smolvlm2_v1_declares_explicit_model_version() -> None:
    manifest = SmolVLM2V1Manifest.model_validate(
        {
            "type": "roboflow_core/smolvlm2@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "my_workspace/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_workspace/3"),
    ]


def test_smolvlm2_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = SmolVLM2V1Manifest.model_validate(
        {
            "type": "roboflow_core/smolvlm2@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_version": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
