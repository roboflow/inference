"""
Dependent-resources discovery tests for the Roboflow multi-class
classification model blocks
(``roboflow_core/roboflow_classification_model@v1..v3``).

Every version declares the model it loads; versions with the
``active_learning_target_dataset`` field additionally declare the target
project when that field is set.
"""

from inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v1 import (
    BlockManifest as MultiClassClassificationV1Manifest,
)
from inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v2 import (
    BlockManifest as MultiClassClassificationV2Manifest,
)
from inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v3 import (
    BlockManifest as MultiClassClassificationV3Manifest,
)
from inference.core.workflows.prototypes.block import (
    roboflow_platform_model,
    roboflow_platform_project,
)

# ---------------------------------------------------------------------------
# v1
# ---------------------------------------------------------------------------


def test_multi_class_classification_v1_declares_model_for_literal_model_id() -> None:
    manifest = MultiClassClassificationV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_classification_model@v1",
            "name": "classifier",
            "images": "$inputs.image",
            "model_id": "my_project/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_multi_class_classification_v1_returns_selector_fed_model_id_verbatim() -> None:
    manifest = MultiClassClassificationV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_classification_model@v1",
            "name": "classifier",
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


def test_multi_class_classification_v2_declares_model_for_literal_model_id() -> None:
    manifest = MultiClassClassificationV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_classification_model@v2",
            "name": "classifier",
            "images": "$inputs.image",
            "model_id": "my_project/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_multi_class_classification_v2_returns_selector_fed_model_id_verbatim() -> None:
    manifest = MultiClassClassificationV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_classification_model@v2",
            "name": "classifier",
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


def test_multi_class_classification_v3_declares_model_only_when_active_learning_dataset_unset() -> (
    None
):
    manifest = MultiClassClassificationV3Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_classification_model@v3",
            "name": "classifier",
            "images": "$inputs.image",
            "model_id": "my_project/3",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
    ]


def test_multi_class_classification_v3_declares_model_and_active_learning_project() -> (
    None
):
    manifest = MultiClassClassificationV3Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_classification_model@v3",
            "name": "classifier",
            "images": "$inputs.image",
            "model_id": "my_project/3",
            "active_learning_target_dataset": "my_dataset",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="my_project/3"),
        roboflow_platform_project(project_url="my_dataset"),
    ]


def test_multi_class_classification_v3_returns_selector_fed_model_id_verbatim() -> None:
    manifest = MultiClassClassificationV3Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_classification_model@v3",
            "name": "classifier",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="$inputs.model"),
    ]
