import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow
from inference.core.workflows.execution_engine.v1.introspection.inputs_discovery import (
    describe_workflow_inputs,
)

# Regression: the `images` field of dataset upload v1 (and its tensor sibling) declared
# `AliasChoices("image", "images")`. Pydantic names the JSON-schema property after the
# first alias, so the schema said `image` while the manifest attribute is `images` -
# compilation raised BlockInterfaceError and /describe_interface raised AttributeError.


def _build_definition(step_type: str, image_property: str) -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": step_type,
                "name": "data_collection",
                image_property: "$inputs.image",
                "target_project": "my_project",
                "usage_quota_name": "my_quota",
                "data_percentage": 100.0,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "registration_message",
                "selector": "$steps.data_collection.message",
            },
        ],
    }


@pytest.mark.parametrize(
    "step_type",
    ["roboflow_core/roboflow_dataset_upload@v1", "RoboflowDatasetUpload"],
)
@pytest.mark.parametrize("image_property", ["image", "images"])
def test_compilation_of_workflow_with_dataset_upload_v1(
    model_manager: ModelManager,
    step_type: str,
    image_property: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    compiled_workflow = compile_workflow(
        workflow_definition=_build_definition(
            step_type=step_type, image_property=image_property
        ),
        init_parameters=workflow_init_parameters,
    )

    # then
    manifest = compiled_workflow.steps["data_collection"].manifest
    assert manifest.images == "$inputs.image"


@pytest.mark.parametrize(
    "step_type",
    ["roboflow_core/roboflow_dataset_upload@v1", "RoboflowDatasetUpload"],
)
@pytest.mark.parametrize("image_property", ["image", "images"])
def test_describe_inputs_of_workflow_with_dataset_upload_v1(
    step_type: str,
    image_property: str,
) -> None:
    # when
    result = describe_workflow_inputs(
        definition=_build_definition(
            step_type=step_type, image_property=image_property
        ),
    )

    # then
    assert result == {"image": ["image"]}
