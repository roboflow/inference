from typing import List

from inference.enterprise.workflows.complier.utils import (
    get_input_parameters_selectors,
    get_output_names,
    get_output_selectors,
    get_steps_input_selectors,
    get_steps_output_selectors,
    get_steps_selectors,
)
from inference.enterprise.workflows.entities.outputs import JsonField
from inference.enterprise.workflows.entities.workflows_specification import (
    BlockType,
    InputType,
    WorkflowSpecificationV1,
)
from inference.enterprise.workflows.errors import (
    DuplicatedSymbolError,
    InvalidReferenceError,
)
from inference.enterprise.workflows.execution_engine.compiler.entities import (
    ParsedWorkflowDefinition,
)


def validate_workflow_specification(
    workflow_definition: ParsedWorkflowDefinition,
) -> None:
    validate_inputs_names_are_unique(inputs=workflow_definition.inputs)
    validate_steps_names_are_unique(steps=workflow_definition.steps)
    validate_outputs_names_are_unique(outputs=workflow_definition.outputs)


def validate_inputs_names_are_unique(inputs: List[InputType]) -> None:
    input_parameters_selectors = get_input_parameters_selectors(inputs=inputs)
    if len(input_parameters_selectors) != len(inputs):
        raise DuplicatedSymbolError("Found duplicated input parameter names")


def validate_steps_names_are_unique(steps: List[BlockType]) -> None:
    steps_selectors = get_steps_selectors(steps=steps)
    if len(steps_selectors) != len(steps):
        raise DuplicatedSymbolError("Found duplicated steps names")


def validate_outputs_names_are_unique(outputs: List[JsonField]) -> None:
    output_names = get_output_names(outputs=outputs)
    if len(output_names) != len(outputs):
        raise DuplicatedSymbolError("Found duplicated outputs names")
