from collections import defaultdict

from inference.core.workflows.execution_engine.entities.base import (
    JsonField,
    WorkflowImage,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.v1.compiler.core import (
    collect_input_substitutions,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    ParsedWorkflowDefinition,
)
from tests.workflows.unit_tests.execution_engine.compiler.plugin_with_test_blocks.blocks import (
    ExampleModelBlockManifest,
)


def test_collect_input_substitutions() -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image_1"),
            WorkflowImage(type="WorkflowImage", name="image_2"),
            WorkflowParameter(type="WorkflowParameter", name="model_1"),
            WorkflowParameter(type="WorkflowParameter", name="model_2"),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image_1",
                model_id="$inputs.model_1",
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_2",
                image="$inputs.image_2",
                model_id="$inputs.model_2",
            ),
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions_1",
                selector="$steps.model_1.predictions",
            ),
            JsonField(
                type="JsonField",
                name="predictions_2",
                selector="$steps.model_2.predictions",
            ),
        ],
    )

    # when
    result = collect_input_substitutions(workflow_definition=workflow_definition)
    aggregated_results = defaultdict(dict)
    for element in result:
        aggregated_results[id(element.step_manifest)][
            element.manifest_property
        ] = element.input_parameter_name

    # then
    assert aggregated_results == {
        id(workflow_definition.steps[0]): {
            "model_id": "model_1",
        },
        id(workflow_definition.steps[1]): {
            "model_id": "model_2",
        },
    }
