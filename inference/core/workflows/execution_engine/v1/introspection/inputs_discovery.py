from typing import Dict, List

from inference.core.workflows.execution_engine.introspection.blocks_loader import describe_available_blocks
from inference.core.workflows.execution_engine.introspection.connections_discovery import \
    parse_all_schemas, get_all_inputs_kind_major
from inference.core.workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler import compile_dynamic_blocks


def describe_workflow_inputs(definition: dict) -> Dict[str, List[str]]:
    dynamic_blocks = compile_dynamic_blocks(
        dynamic_blocks_definitions=definition.get("dynamic_blocks_definitions", [])
    )
    blocks_description = describe_available_blocks(
        dynamic_blocks=dynamic_blocks,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )
    all_schemas = parse_all_schemas(blocks_description=blocks_description)
    detailed_input_kind2schemas = get_all_inputs_kind_major(
        blocks_description=blocks_description,
        all_schemas=all_schemas,
    )
