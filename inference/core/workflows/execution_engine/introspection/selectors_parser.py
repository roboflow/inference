from typing import Any, List, Optional

from inference.core.workflows.errors import BlockInterfaceError
from inference.core.workflows.execution_engine.introspection.entities import (
    ParsedSelector,
    SelectorDefinition,
)
from inference.core.workflows.execution_engine.introspection.schema_parser import (
    parse_block_manifest,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import is_selector
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


def get_step_selectors(
    step_manifest: WorkflowBlockManifest,
) -> List[ParsedSelector]:
    parsed_schema = parse_block_manifest(manifest_type=type(step_manifest))
    result = []
    for selector_definition in parsed_schema.selectors.values():
        property_name = selector_definition.property_name
        property_value = retrieve_property_from_manifest(
            step_manifest=step_manifest,
            property_name=property_name,
        )
        if selector_definition.is_list_element:
            selectors = retrieve_selectors_from_array(
                step_name=step_manifest.name,
                property_value=property_value,
                selector_definition=selector_definition,
            )
            result.extend(selectors)
        elif selector_definition.is_dict_element:
            selectors = retrieve_selectors_from_dictionary(
                step_name=step_manifest.name,
                property_value=property_value,
                selector_definition=selector_definition,
            )
            result.extend(selectors)
        else:
            selector = retrieve_selector_from_simple_property(
                step_name=step_manifest.name,
                property_value=property_value,
                selector_definition=selector_definition,
            )
            result.append(selector)
    return [r for r in result if r is not None]


def retrieve_property_from_manifest(
    step_manifest: WorkflowBlockManifest, property_name: str
) -> Any:
    if not hasattr(step_manifest, property_name):
        raise BlockInterfaceError(
            public_message=f"Attempted to retrieve property {property_name} from "
            f"manifest of step {step_manifest.name} based od manifest schema, but property "
            f"is not defined for object instance. That may be due to aliasing of manifest property "
            f"name in pydantic class, which is not allowed.",
            context="workflow_compilation | execution_graph_construction",
        )
    return getattr(step_manifest, property_name)


def retrieve_selectors_from_array(
    step_name: str,
    property_value: Any,
    selector_definition: SelectorDefinition,
) -> List[ParsedSelector]:
    if not isinstance(property_value, list):
        return []
    result = []
    for index, element in enumerate(property_value):
        selector = retrieve_selector_from_simple_property(
            step_name=step_name,
            property_value=element,
            selector_definition=selector_definition,
            index=index,
        )
        if selector is not None:
            result.append(selector)
    return result


def retrieve_selectors_from_dictionary(
    step_name: str,
    property_value: Any,
    selector_definition: SelectorDefinition,
) -> List[ParsedSelector]:
    if not isinstance(property_value, dict):
        return []
    result = []
    for key, element in property_value.items():
        selector = retrieve_selector_from_simple_property(
            step_name=step_name,
            property_value=element,
            selector_definition=selector_definition,
            key=key,
        )
        if selector is not None:
            result.append(selector)
    return result


def retrieve_selector_from_simple_property(
    step_name: str,
    property_value: Any,
    selector_definition: SelectorDefinition,
    index: Optional[int] = None,
    key: Optional[str] = None,
) -> Optional[ParsedSelector]:
    if not is_selector(property_value):
        return None
    return ParsedSelector(
        definition=selector_definition,
        step_name=step_name,
        value=property_value,
        index=index,
        key=key,
    )
