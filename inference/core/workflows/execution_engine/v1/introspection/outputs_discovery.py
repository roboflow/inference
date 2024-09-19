from typing import Dict, List, Tuple, Union

from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    BlocksDescription,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
    is_step_output_selector,
)
from inference.core.workflows.execution_engine.v1.core import (
    EXECUTION_ENGINE_V1_VERSION,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    compile_dynamic_blocks,
)


def describe_workflow_outputs(
    definition: dict,
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    dynamic_blocks = compile_dynamic_blocks(
        dynamic_blocks_definitions=definition.get("dynamic_blocks_definitions", [])
    )
    blocks_description = describe_available_blocks(
        dynamic_blocks=dynamic_blocks,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )
    block_output_map = get_blocks_output_property_kinds(
        blocks_description=blocks_description
    )
    try:
        step_name_to_block_type = map_step_name_to_block_type(
            workflow_steps=definition["steps"]
        )
        return determine_workflow_outputs_kinds(
            outputs_definitions=definition["outputs"],
            step_name_to_block_type=step_name_to_block_type,
            block_output_map=block_output_map,
        )
    except KeyError as error:
        raise WorkflowDefinitionError(
            public_message=f"Workflow definition invalid - missing property `{error}`.",
            inner_error=error,
            context="describing_workflow_outputs",
        )


def map_step_name_to_block_type(workflow_steps: List[dict]) -> Dict[str, str]:
    result = {}
    for step in workflow_steps:
        if "name" not in step or "type" not in step:
            raise WorkflowDefinitionError(
                public_message="Workflow definition invalid - step without `name` or `type` defined found.",
                context="describing_workflow_outputs",
            )
        result[step["name"]] = step["type"]
    return result


def extract_step_name_and_selected_property(selector: str) -> Tuple[str, str]:
    if not is_step_output_selector(selector_or_value=selector):
        raise WorkflowDefinitionError(
            public_message="Workflow definition invalid - output does not contain step selector.",
            context="describing_workflow_outputs",
        )
    step_selector = get_step_selector_from_its_output(step_output_selector=selector)
    step_name = get_last_chunk_of_selector(selector=step_selector)
    selected_property = get_last_chunk_of_selector(selector=selector)
    return step_name, selected_property


def get_blocks_output_property_kinds(
    blocks_description: BlocksDescription,
) -> Dict[str, Dict[str, List[str]]]:
    block_output_map = {}
    for block in blocks_description.blocks:
        key = block.manifest_type_identifier
        output_property_kinds = get_output_property_kinds(block.outputs_manifest)
        block_output_map[key] = output_property_kinds
        for alias in block.manifest_type_identifier_aliases:
            block_output_map[alias] = output_property_kinds
    return block_output_map


def get_output_property_kinds(
    outputs_manifest: List[OutputDefinition],
) -> Dict[str, List[str]]:
    result = {}
    for output in outputs_manifest:
        result[output.name] = [kind.name for kind in output.kind]
    return result


def determine_workflow_outputs_kinds(
    outputs_definitions: List[dict],
    step_name_to_block_type: Dict[str, str],
    block_output_map: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    workflow_response_definition = {}
    for output in outputs_definitions:
        output_name = output["name"]
        selector = output["selector"]
        step_name, selected_property = extract_step_name_and_selected_property(
            selector=selector,
        )
        if step_name not in step_name_to_block_type:
            raise WorkflowDefinitionError(
                public_message=f"Could not find step referred in outputs (`{step_name}`) within Workflow steps.",
                context="describing_workflow_outputs",
            )
        step_type = step_name_to_block_type[step_name]
        output_properties = block_output_map[step_type]
        if selected_property == "*":
            property_kind = output_properties
        else:
            if selected_property not in output_properties:
                raise WorkflowDefinitionError(
                    public_message=f"Step `{step_name}` does not declare {selected_property} in its outputs.",
                    context="describing_workflow_outputs",
                )
            property_kind = output_properties[selected_property]
        workflow_response_definition[output_name] = property_kind
    return workflow_response_definition
