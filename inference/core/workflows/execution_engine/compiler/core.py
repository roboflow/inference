from typing import Any, Callable, Dict, List, Union

from inference.core.workflows.entities.base import WorkflowParameter
from inference.core.workflows.execution_engine.compiler.entities import (
    CompiledWorkflow,
    InputSubstitution,
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.compiler.graph_constructor import (
    prepare_execution_graph,
)
from inference.core.workflows.execution_engine.compiler.steps_initialiser import (
    initialise_steps,
)
from inference.core.workflows.execution_engine.compiler.syntactic_parser import (
    parse_workflow_definition,
)
from inference.core.workflows.execution_engine.compiler.utils import (
    construct_input_selector,
)
from inference.core.workflows.execution_engine.compiler.validator import (
    validate_workflow_specification,
)
from inference.core.workflows.execution_engine.debugger.core import dump_execution_graph
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_initializers,
    load_workflow_blocks,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


def compile_workflow(
    workflow_definition: dict,
    init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
) -> CompiledWorkflow:
    available_blocks = load_workflow_blocks()
    initializers = load_initializers()
    parsed_workflow_definition = parse_workflow_definition(
        raw_workflow_definition=workflow_definition,
    )
    validate_workflow_specification(workflow_definition=parsed_workflow_definition)
    execution_graph = prepare_execution_graph(
        workflow_definition=parsed_workflow_definition,
    )
    steps = initialise_steps(
        steps_manifest=parsed_workflow_definition.steps,
        available_bocks=available_blocks,
        explicit_init_parameters=init_parameters,
        initializers=initializers,
    )
    input_substitutions = collect_input_substitutions(
        workflow_definition=parsed_workflow_definition,
    )
    steps_by_name = {step.manifest.name: step for step in steps}
    dump_execution_graph(execution_graph=execution_graph)
    return CompiledWorkflow(
        workflow_definition=parsed_workflow_definition,
        execution_graph=execution_graph,
        steps=steps_by_name,
        input_substitutions=input_substitutions,
    )


def collect_input_substitutions(
    workflow_definition: ParsedWorkflowDefinition,
) -> List[InputSubstitution]:
    result = []
    for declared_input in workflow_definition.inputs:
        if not isinstance(declared_input, WorkflowParameter):
            continue
        input_substitutions = collect_substitutions_for_selected_input(
            input_name=declared_input.name,
            steps=workflow_definition.steps,
        )
        result.extend(input_substitutions)
    return result


def collect_substitutions_for_selected_input(
    input_name: str,
    steps: List[WorkflowBlockManifest],
) -> List[InputSubstitution]:
    input_selector = construct_input_selector(input_name=input_name)
    substitutions = []
    for step in steps:
        for field in step.model_fields:
            if getattr(step, field) != input_selector:
                continue
            substitution = InputSubstitution(
                input_parameter_name=input_name,
                step_manifest=step,
                manifest_property=field,
            )
            substitutions.append(substitution)
    return substitutions
