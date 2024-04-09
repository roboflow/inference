from typing import Any, Dict

from inference.enterprise.workflows.execution_engine.compiler.blocks_loader import (
    load_initializers,
    load_workflow_blocks,
)
from inference.enterprise.workflows.execution_engine.compiler.steps_initialiser import (
    initialise_steps,
)
from inference.enterprise.workflows.execution_engine.compiler.syntactic_parser import (
    parse_workflow_definition,
)


def compile_workflow(
    raw_workflow_definition: dict,
    init_parameters: Dict[str, Any],
) -> None:
    available_blocks = load_workflow_blocks()
    initializers = load_initializers()
    parsed_workflow_definition = parse_workflow_definition(
        raw_workflow_definition=raw_workflow_definition,
    )
    steps = initialise_steps(
        steps_manifest=parsed_workflow_definition.steps,
        available_bocks=available_blocks,
        explicit_init_parameters=init_parameters,
        initializers=initializers,
    )
