"""
Dependent-resources discovery tests for the Roboflow object detection
model blocks (``roboflow_core/roboflow_object_detection_model@v1..v3``).

Every version declares the model it loads; versions with the
``active_learning_target_dataset`` field additionally declare the target
project when that field is set.
"""

from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    BlockManifest as ObjectDetectionV1Manifest,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v2 import (
    BlockManifest as ObjectDetectionV2Manifest,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v3 import (
    BlockManifest as ObjectDetectionV3Manifest,
)
from inference.core.workflows.prototypes.block import (
    roboflow_platform_model,
    roboflow_platform_project,
)

# ---------------------------------------------------------------------------
# v1
# ---------------------------------------------------------------------------


def test_object_detection_v1_declares_model_only_when_active_learning_dataset_unset() -> (
    None
):
    manifest = ObjectDetectionV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "my_project/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_object_detection_v1_declares_model_and_active_learning_project() -> None:
    manifest = ObjectDetectionV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "my_project/3",
            "disable_active_learning": False,
            "active_learning_target_dataset": "my_dataset",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
        roboflow_platform_project(project_url="my_dataset"),
    ]


def test_object_detection_v1_ignores_target_dataset_when_active_learning_left_disabled() -> (
    None
):
    manifest = ObjectDetectionV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "my_project/3",
            "active_learning_target_dataset": "my_dataset",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_object_detection_v1_returns_selector_fed_model_id_verbatim() -> None:
    manifest = ObjectDetectionV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]


# ---------------------------------------------------------------------------
# v2
# ---------------------------------------------------------------------------


def test_object_detection_v2_declares_model_for_literal_model_id() -> None:
    manifest = ObjectDetectionV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "my_project/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_object_detection_v2_returns_selector_fed_model_id_verbatim() -> None:
    manifest = ObjectDetectionV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]


# ---------------------------------------------------------------------------
# v3
# ---------------------------------------------------------------------------


def test_object_detection_v3_declares_model_for_literal_model_id() -> None:
    manifest = ObjectDetectionV3Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "my_project/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_object_detection_v3_returns_selector_fed_model_id_verbatim() -> None:
    manifest = ObjectDetectionV3Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
