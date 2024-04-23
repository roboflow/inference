from typing import Any, List, Optional

from inference.enterprise.workflows.entities.types import Kind
from inference.enterprise.workflows.entities.validators import is_selector
from inference.enterprise.workflows.errors import (
    BlockInterfaceError,
    PluginInterfaceError,
)
from inference.enterprise.workflows.execution_engine.introspection.entities import (
    ReferenceDefinition,
    SelectorDefinition,
)
from inference.enterprise.workflows.prototypes.block import WorkflowBlockManifest


def get_step_selectors(
    step_manifest: WorkflowBlockManifest,
) -> List[SelectorDefinition]:
    openapi_schema = step_manifest.schema()
    result = []
    for property_name, property_definition in openapi_schema["properties"].items():
        property_value = retrieve_property_from_manifest(
            step_manifest=step_manifest,
            property_name=property_name,
        )
        if "items" in property_definition:
            selectors = retrieve_selectors_from_array(
                step_name=step_manifest.name,
                property_name=property_name,
                property_value=property_value,
                property_definition=property_definition["items"],
            )
            result.extend(selectors)
        else:
            selector = retrieve_selector_from_simple_property(
                step_name=step_manifest.name,
                property_name=property_name,
                property_value=property_value,
                property_definition=property_definition,
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
    property_name: str,
    property_value: List[Any],
    property_definition: dict,
) -> List[SelectorDefinition]:
    result = []
    for index, element in enumerate(property_value):
        selector = retrieve_selector_from_simple_property(
            step_name=step_name,
            property_name=property_name,
            property_value=element,
            property_definition=property_definition,
            index=index,
        )
        if selector is not None:
            result.append(selector)
    return result


def retrieve_selector_from_simple_property(
    step_name: str,
    property_name: str,
    property_value: Any,
    property_definition: dict,
    index: Optional[int] = None,
) -> Optional[SelectorDefinition]:
    if not is_selector(property_value):
        return None
    if "reference" in property_definition:
        return retrieve_selector_from_property_with_specific_type(
            step_name=step_name,
            property_name=property_name,
            property_value=property_value,
            property_definition=property_definition,
            index=index,
        )
    error_message = (
        f"While retrieving selectors for step {step_name} based on block manifest and values "
        f"provided in definition, compiler detected selector value provided for property "
        f"{property_name} that does not correspond to type of property declared in manifest. "
        f"Compiler expected type that is reference to other block element or defines reference "
        f"as at least one of possible value for the property."
    )
    if "anyOf" not in property_definition and "oneOf" not in property_definition:
        raise BlockInterfaceError(
            public_message=error_message,
            context="workflow_compilation | execution_graph_construction",
        )
    allowed_references = retrieve_allowed_reference_definitions_from_types_union(
        property_definition=property_definition,
    )
    if len(allowed_references) == 0:
        raise BlockInterfaceError(
            public_message=error_message,
            context="workflow_compilation | execution_graph_construction",
        )
    return SelectorDefinition(
        step_name=step_name,
        property_name=property_name,
        index=index,
        selector=property_value,
        allowed_references=allowed_references,
    )


def retrieve_selector_from_property_with_specific_type(
    step_name: str,
    property_name: str,
    property_value: Any,
    property_definition: dict,
    index: Optional[int] = None,
) -> SelectorDefinition:
    allowed_references = [
        ReferenceDefinition(
            selected_element=property_definition["selected_element"],
            kind=[Kind.model_validate(k) for k in property_definition.get("kind", [])],
        )
    ]
    return SelectorDefinition(
        step_name=step_name,
        property_name=property_name,
        index=index,
        selector=property_value,
        allowed_references=allowed_references,
    )


def retrieve_allowed_reference_definitions_from_types_union(
    property_definition: dict,
) -> List[ReferenceDefinition]:
    allowed_references_definitions = property_definition.get(
        "anyOf", []
    ) + property_definition.get("oneOf", [])
    allowed_references = []
    for definition in allowed_references_definitions:
        if "reference" not in definition:
            continue
        reference_definition = ReferenceDefinition(
            selected_element=definition["selected_element"],
            kind=[Kind.model_validate(k) for k in definition.get("kind", [])],
        )
        allowed_references.append(reference_definition)
    return allowed_references
