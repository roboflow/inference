from typing import Any, Dict

from inference.enterprise.workflows.execution_engine.compiler.blocks_loader import (
    load_initializers,
    load_workflow_blocks,
)
from inference.enterprise.workflows.execution_engine.compiler.entities import (
    CompiledWorkflow,
)
from inference.enterprise.workflows.execution_engine.compiler.graph_constructor import (
    prepare_execution_graph,
)
from inference.enterprise.workflows.execution_engine.compiler.steps_initialiser import (
    initialise_steps,
)
from inference.enterprise.workflows.execution_engine.compiler.syntactic_parser import (
    parse_workflow_definition,
)
from inference.enterprise.workflows.execution_engine.compiler.validator import (
    validate_workflow_specification,
)
from inference.enterprise.workflows.execution_engine.debugger.core import (
    dump_execution_graph,
)


def compile_workflow(
    raw_workflow_definition: dict,
    init_parameters: Dict[str, Any],
) -> CompiledWorkflow:
    available_blocks = load_workflow_blocks()
    initializers = load_initializers()
    parsed_workflow_definition = parse_workflow_definition(
        raw_workflow_definition=raw_workflow_definition,
    )
    validate_workflow_specification(workflow_definition=parsed_workflow_definition)
    execution_graph = prepare_execution_graph(
        workflow_definition=parsed_workflow_definition,
        available_blocks=available_blocks,
    )
    steps = initialise_steps(
        steps_manifest=parsed_workflow_definition.steps,
        available_bocks=available_blocks,
        explicit_init_parameters=init_parameters,
        initializers=initializers,
    )
    dump_execution_graph(execution_graph=execution_graph)
    return CompiledWorkflow(
        execution_graph=execution_graph,
        steps=steps,
    )
