from typing import List, Optional

from inference.core.workflows.errors import DuplicatedNameError
from inference.core.workflows.execution_engine.entities.base import InputType, JsonField
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    get_input_parameters_selectors,
    get_output_selectors,
    get_steps_selectors,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


@execution_phase(
    name="workflow_definition_validation",
    categories=["execution_engine_operation"],
)
def validate_workflow_specification(
    workflow_definition: ParsedWorkflowDefinition,
    profiler: Optional[WorkflowsProfiler] = None,
) -> None:
    validate_inputs_names_are_unique(inputs=workflow_definition.inputs)
    validate_steps_names_are_unique(steps=workflow_definition.steps)
    validate_outputs_names_are_unique(outputs=workflow_definition.outputs)


def validate_inputs_names_are_unique(inputs: List[InputType]) -> None:
    input_parameters_selectors = get_input_parameters_selectors(inputs=inputs)
    if len(input_parameters_selectors) != len(inputs):
        raise DuplicatedNameError(
            public_message="Found duplicated input parameter names",
            context="workflow_compilation | specification_validation",
        )


def validate_steps_names_are_unique(steps: List[WorkflowBlockManifest]) -> None:
    steps_selectors = get_steps_selectors(steps=steps)
    if len(steps_selectors) != len(steps):
        raise DuplicatedNameError(
            public_message="Found duplicated input steps names",
            context="workflow_compilation | specification_validation",
        )


def validate_outputs_names_are_unique(outputs: List[JsonField]) -> None:
    output_names = get_output_selectors(outputs=outputs)
    if len(output_names) != len(outputs):
        raise DuplicatedNameError(
            public_message="Found duplicated input outputs names",
            context="workflow_compilation | specification_validation",
        )
