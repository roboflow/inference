"""
Dependent-resources discovery tests for the Florence-2 blocks:

- ``roboflow_core/florence_2@v1`` holds the model id in ``model_version``
  (literal ids like ``florence-2-base``),
- ``roboflow_core/florence_2@v2`` holds it in ``model_id`` instead.

The default ``task_type`` (``open-vocabulary-object-detection``) requires
``classes``, so payloads pin ``task_type`` to ``ocr`` which needs no extras.
"""

from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
    BlockManifest as Florence2V1Manifest,
)
from inference.core.workflows.core_steps.models.foundation.florence2.v2 import (
    V2BlockManifest as Florence2V2Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_florence2_v1_declares_default_model_version() -> None:
    manifest = Florence2V1Manifest.model_validate(
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "ocr",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="florence-2-base"),
    ]


def test_florence2_v1_declares_explicit_model_version() -> None:
    manifest = Florence2V1Manifest.model_validate(
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "ocr",
            "model_version": "florence-2-large",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="florence-2-large"),
    ]


def test_florence2_v1_returns_selector_fed_model_version_verbatim() -> None:
    manifest = Florence2V1Manifest.model_validate(
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "ocr",
            "model_version": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]


def test_florence2_v2_declares_default_model_id() -> None:
    manifest = Florence2V2Manifest.model_validate(
        {
            "type": "roboflow_core/florence_2@v2",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "ocr",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="florence-2-base"),
    ]


def test_florence2_v2_declares_explicit_model_id() -> None:
    manifest = Florence2V2Manifest.model_validate(
        {
            "type": "roboflow_core/florence_2@v2",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "ocr",
            "model_id": "florence-pretrains/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="florence-pretrains/3"),
    ]


def test_florence2_v2_returns_selector_fed_model_id_verbatim() -> None:
    manifest = Florence2V2Manifest.model_validate(
        {
            "type": "roboflow_core/florence_2@v2",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "ocr",
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
