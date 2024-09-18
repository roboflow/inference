from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Type

from pydantic import ValidationError

from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.entities.types import WILDCARD_KIND
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
)
from inference.core.workflows.execution_engine.introspection.connections_discovery import (
    parse_all_schemas,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockDescription,
    BlockManifestMetadata,
    SelectorDefinition,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    construct_input_selector,
    is_input_selector,
)
from inference.core.workflows.execution_engine.v1.core import (
    EXECUTION_ENGINE_V1_VERSION,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    compile_dynamic_blocks,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

SELECTED_ELEMENT_TO_INPUT_TYPE = {
    "workflow_video_metadata": {"WorkflowVideoMetadata"},
    "workflow_image": {"WorkflowImage", "InferenceImage"},
    "workflow_parameter": {"WorkflowParameter", "InferenceParameter"},
}
INPUT_TYPE_TO_SELECTED_ELEMENT = {
    input_type: selected_element
    for selected_element, input_types in SELECTED_ELEMENT_TO_INPUT_TYPE.items()
    for input_type in input_types
}


@dataclass(frozen=True)
class InputMetadata:
    name: str
    selector: str
    type: str
    declared_kind: Optional[Set[str]]


@dataclass(frozen=True)
class SelectorSearchResult:
    selector: str
    step_type: str
    step_name: str
    property_name: str
    compatible_kinds: Set[str]


def describe_workflow_inputs(definition: dict) -> Dict[str, List[str]]:
    try:
        block_type_to_metadata, block_type_to_manifest = parse_blocks(
            dynamic_blocks_definitions=definition.get("dynamic_blocks_definitions", []),
        )
        input_selectors_details = retrieve_input_selectors_details(
            inputs=definition.get("inputs", [])
        )
        search_results = search_input_selectors_in_steps(
            steps=definition.get("steps", []),
            block_type_to_metadata=block_type_to_metadata,
            block_type_to_manifest=block_type_to_manifest,
            input_selectors_details=input_selectors_details,
        )
        return summarise_input_kinds(
            input_selectors_details=input_selectors_details,
            search_results=search_results,
        )
    except KeyError as error:
        raise WorkflowDefinitionError(
            public_message=f"Workflow definition invalid - missing property `{error}`.",
            inner_error=error,
            context="describing_workflow_inputs",
        )


def parse_blocks(
    dynamic_blocks_definitions: List[dict],
) -> Tuple[Dict[str, BlockManifestMetadata], Dict[str, Type[WorkflowBlockManifest]]]:
    dynamic_blocks = compile_dynamic_blocks(
        dynamic_blocks_definitions=dynamic_blocks_definitions
    )
    blocks_description = describe_available_blocks(
        dynamic_blocks=dynamic_blocks,
        execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
    )
    block_class2all_type_names = map_block_class2all_aliases(
        block_descriptions=blocks_description.blocks,
    )
    block_class2input_manifest_class = {
        block.block_class: block.block_class.get_manifest()
        for block in blocks_description.blocks
    }
    all_schemas = parse_all_schemas(blocks_description=blocks_description)
    results_metadata, results_manifest = {}, {}
    for workflow_block, block_manifest_metadata in all_schemas.items():
        block_type_names = block_class2all_type_names[workflow_block]
        for name in block_type_names:
            results_metadata[name] = block_manifest_metadata
            results_manifest[name] = block_class2input_manifest_class[workflow_block]
    return results_metadata, results_manifest


def map_block_class2all_aliases(
    block_descriptions: List[BlockDescription],
) -> Dict[Type[WorkflowBlock], Set[str]]:
    block_class2all_type_names = {}
    for description in block_descriptions:
        block_class2all_type_names[description.block_class] = set(
            [description.manifest_type_identifier]
            + description.manifest_type_identifier_aliases
        )
    return block_class2all_type_names


def retrieve_input_selectors_details(
    inputs: List[dict],
) -> Dict[str, InputMetadata]:
    unique_names = {input_element["name"] for input_element in inputs}
    if len(unique_names) != len(inputs):
        raise WorkflowDefinitionError(
            public_message=f"Workflow definition invalid - non unique inputs names provided",
            context="describing_workflow_inputs",
        )
    result = {}
    for input_element in inputs:
        input_selector = construct_input_selector(input_name=input_element["name"])
        declared_kind = input_element.get("kind")
        if declared_kind:
            declared_kind = set(declared_kind)
        result[input_selector] = InputMetadata(
            name=input_element["name"],
            selector=input_selector,
            type=input_element["type"],
            declared_kind=declared_kind,
        )
    return result


def search_input_selectors_in_steps(
    steps: List[dict],
    input_selectors_details: Dict[str, InputMetadata],
    block_type_to_manifest: Dict[str, Type[WorkflowBlockManifest]],
    block_type_to_metadata: Dict[str, BlockManifestMetadata],
) -> List[SelectorSearchResult]:
    result = []
    for step in steps:
        step_type = step["type"]
        if step_type not in block_type_to_metadata:
            raise WorkflowDefinitionError(
                public_message=f"Workflow definition invalid - used step of type `{step_type}` which is not available "
                f"in this installation of Workflows Execution Engine.",
                context="describing_workflow_inputs",
            )
        result.extend(
            search_input_selectors_in_step(
                step_definition=step,
                input_selectors_details=input_selectors_details,
                block_type_to_manifest=block_type_to_manifest,
                block_type_to_metadata=block_type_to_metadata,
            )
        )
    return result


def search_input_selectors_in_step(
    step_definition: dict,
    input_selectors_details: Dict[str, InputMetadata],
    block_type_to_manifest: Dict[str, Type[WorkflowBlockManifest]],
    block_type_to_metadata: Dict[str, BlockManifestMetadata],
) -> List[SelectorSearchResult]:
    step_type = step_definition["type"]
    step_name = step_definition["name"]
    result = []
    block_metadata = block_type_to_metadata[step_type]
    block_manifest_class = block_type_to_manifest[step_type]
    for property_name, selector_definition in block_metadata.selectors.items():
        try:
            manifest = block_manifest_class.model_validate(step_definition)
        except ValidationError as error:
            raise WorkflowDefinitionError(
                public_message=f"Workflow definition invalid - step `{step_name}` misconfigured. See details in inner error.",
                inner_error=error,
                context="describing_workflow_inputs",
            )
        matching_references_kinds = grab_input_compatible_references_kinds(
            selector_definition=selector_definition
        )
        detected_input_selectors = grab_input_selectors_defined_for_step(
            block_manifest=manifest,
            property_name=property_name,
            selector_definition=selector_definition,
        )
        result.extend(
            prepare_search_results_for_detected_selectors(
                step_name=step_name,
                step_type=step_type,
                property_name=property_name,
                detected_input_selectors=detected_input_selectors,
                input_selectors_details=input_selectors_details,
                matching_references_kinds=matching_references_kinds,
            )
        )
    return result


def grab_input_compatible_references_kinds(
    selector_definition: SelectorDefinition,
) -> Dict[str, Set[str]]:
    matching_references = defaultdict(set)
    for reference in selector_definition.allowed_references:
        if reference.selected_element not in SELECTED_ELEMENT_TO_INPUT_TYPE:
            continue
        matching_references[reference.selected_element].update(
            k.name for k in reference.kind
        )
    return matching_references


def grab_input_selectors_defined_for_step(
    block_manifest: WorkflowBlockManifest,
    property_name: str,
    selector_definition: SelectorDefinition,
) -> List[str]:
    list_allowed = selector_definition.is_list_element
    dict_allowed = selector_definition.is_dict_element
    detected_input_selectors = []
    value = getattr(block_manifest, property_name)
    if list_allowed and isinstance(value, list):
        for selector in value:
            if is_input_selector(selector_or_value=selector):
                detected_input_selectors.append(selector)
    if dict_allowed and isinstance(value, dict):
        for selector in value.values():
            if is_input_selector(selector_or_value=selector):
                detected_input_selectors.append(selector)
    if is_input_selector(selector_or_value=value):
        detected_input_selectors.append(value)
    return detected_input_selectors


def prepare_search_results_for_detected_selectors(
    step_name: str,
    step_type: str,
    property_name: str,
    detected_input_selectors: List[str],
    input_selectors_details: Dict[str, InputMetadata],
    matching_references_kinds: Dict[str, Set[str]],
) -> List[SelectorSearchResult]:
    result = []
    for detected_input_selector in detected_input_selectors:
        if detected_input_selector not in input_selectors_details:
            raise WorkflowDefinitionError(
                public_message=f"Workflow definition invalid - step `{step_name}` declares input selector "
                f"{detected_input_selector} which is not specified in inputs.",
                context="describing_workflow_inputs",
            )
        selector_details = input_selectors_details[detected_input_selector]
        if selector_details.type not in INPUT_TYPE_TO_SELECTED_ELEMENT:
            raise WorkflowDefinitionError(
                public_message=f"Workflow definition invalid - declared input of type: {selector_details.type} "
                f"which is not supported in this installation of Workflow Execution Engine.",
                context="describing_workflow_inputs",
            )
        selected_element = INPUT_TYPE_TO_SELECTED_ELEMENT[selector_details.type]
        kinds_for_element = matching_references_kinds[selected_element]
        if not kinds_for_element:
            raise WorkflowDefinitionError(
                public_message=f"Workflow definition invalid - selector `{detected_input_selector}` declared for "
                f"step `{step_name}` property `{property_name}` is invalid.",
                context="describing_workflow_inputs",
            )
        result.append(
            SelectorSearchResult(
                selector=detected_input_selector,
                step_type=step_type,
                step_name=step_name,
                property_name=property_name,
                compatible_kinds=kinds_for_element,
            )
        )
    return result


def summarise_input_kinds(
    input_selectors_details: Dict[str, InputMetadata],
    search_results: List[SelectorSearchResult],
) -> Dict[str, List[str]]:
    search_results_for_selectors = defaultdict(list)
    for search_result in search_results:
        search_results_for_selectors[search_result.selector].append(search_result)
    result = {}
    for (
        input_selector,
        connected_search_results,
    ) in search_results_for_selectors.items():
        actual_kind_for_selector = generate_kinds_union(
            search_results=connected_search_results
        )
        selector_details = input_selectors_details[input_selector]
        declared_kind_for_selector = selector_details.declared_kind or {
            WILDCARD_KIND.name
        }
        if not kinds_are_matching(
            x=actual_kind_for_selector, y=declared_kind_for_selector
        ):
            raise WorkflowDefinitionError(
                public_message=f"Workflow definition contains whe following declaration of kind "
                f"for input `{selector_details.name}`: {declared_kind_for_selector}, whereas "
                f"context of input selectors usage indicate conflicting `{actual_kind_for_selector}`.",
                context="describing_workflow_inputs",
            )
        result[selector_details.name] = list(actual_kind_for_selector)
    unbounded_inputs = set(v.name for v in input_selectors_details.values()).difference(
        result.keys()
    )
    for input_name in unbounded_inputs:
        result[input_name] = [WILDCARD_KIND.name]
    return result


def generate_kinds_union(search_results: List[SelectorSearchResult]) -> Set[str]:
    if not search_results:
        return set()
    reference_result = search_results[0]
    kinds_union = set()
    for result in search_results:
        if not kinds_are_matching(
            x=result.compatible_kinds, y=reference_result.compatible_kinds
        ):
            raise WorkflowDefinitionError(
                public_message=f"Workflow definition invalid - input element: `{reference_result.selector}` "
                f"is used in two steps: `{reference_result.step_name}` and `{result.step_name}`. "
                f"Those steps expect incompatible type of data requiring `{reference_result.selector}` "
                f"to be of different type at the same time. "
                f"`{reference_result.step_name}.{reference_result.property_name}` requires: "
                f"{reference_result.compatible_kinds} and `{result.step_name}.{result.property_name}` "
                f"requires: `{result.compatible_kinds}`",
                context="describing_workflow_inputs",
            )
        kinds_union.update(result.compatible_kinds)
    return kinds_union


def kinds_are_matching(x: Set[str], y: Set[str]) -> bool:
    if WILDCARD_KIND.name in x or WILDCARD_KIND.name in y:
        return True
    return len(x.intersection(y)) > 0
