"""
Dependent-resources discovery tests for the Action Recognition Model block
(``roboflow_core/roboflow_action_recognition_model@v1``).

The block loads its model in-process through
``model_manager.load_action_recognition_model()``, whatever the generic step
execution mode is, so it declares that model as LOCAL EXECUTION and keeps it
away from the generic ``add_model()`` preloader (``preloadable=False``). The
tensor manifest inherits the same model-resource discovery.
"""

import pytest
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    BlockManifest as ActionRecognitionV1Manifest,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1_tensor import (
    BlockManifest as ActionRecognitionV1TensorManifest,
)
from roboflow_workflows.prototypes.block import (
    DependentResourceType,
    ModelExecutionLocation,
    ModelRequiredAction,
)


@pytest.mark.parametrize(
    "model_id", ["my-project/3", "$inputs.action_model"], ids=["literal", "selector"]
)
@pytest.mark.parametrize(
    "manifest_type", [ActionRecognitionV1Manifest, ActionRecognitionV1TensorManifest]
)
def test_action_recognition_declares_its_model_as_local_non_preloadable(
    model_id: str,
    manifest_type,
) -> None:
    # given
    manifest = manifest_type.model_validate(
        {
            "type": "roboflow_core/roboflow_action_recognition_model@v1",
            "name": "actions",
            "images": "$inputs.image",
            "model_id": model_id,
        }
    )

    # when
    result = manifest.discover_dependent_resources()

    # then
    assert len(result) == 1
    resource = result[0]
    assert resource.resource_type is DependentResourceType.ROBOFLOW_PLATFORM_MODEL
    assert resource.metadata.model_id == model_id
    assert resource.metadata.required_action is ModelRequiredAction.EXECUTION
    assert resource.metadata.execution_location is ModelExecutionLocation.LOCAL
    assert resource.metadata.preloadable is False
