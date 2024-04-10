from typing import Any, List, Optional

from inference.enterprise.workflows.entities.types import Kind
from inference.enterprise.workflows.entities.validators import is_selector
from inference.enterprise.workflows.execution_engine.compiler.entities import (
    ReferenceDefinition,
    SelectorDefinition,
)
from inference.enterprise.workflows.prototypes.block import WorkflowBlockManifest


def get_step_selectors(step: WorkflowBlockManifest) -> List[SelectorDefinition]:
    openapi_schema = step.schema()
    result = []
    for property_name, property_definition in openapi_schema["properties"].items():
        property_value = getattr(step, property_name)
        print(f"Step: {step.name}, property: {property_name}, value: {property_value}")
        if "items" in property_definition:
            print("Found items")
            selectors = retrieve_selectors_from_property(
                step_name=step.name,
                property_name=property_name,
                property_value=property_value,
                property_definition=property_definition,
            )
            result.extend(selectors)
        else:
            selector = retrieve_selector_from_property(
                step_name=step.name,
                property_name=property_name,
                property_value=property_value,
                property_definition=property_definition,
            )
            result.append(selector)
    return [r for r in result if r is not None]


def retrieve_selectors_from_property(
    step_name: str,
    property_name: str,
    property_value: List[Any],
    property_definition: dict,
) -> List[SelectorDefinition]:
    result = []
    for index, element in enumerate(property_value):
        selector = retrieve_selector_from_property(
            step_name=step_name,
            property_name=property_name,
            property_value=element,
            property_definition=property_definition["items"],
            index=index,
        )
        print(f"index: {index}, element: {element}, selector: {selector}")
        if selector is not None:
            result.append(selector)
    return result


def retrieve_selector_from_property(
    step_name: str,
    property_name: str,
    property_value: Any,
    property_definition: dict,
    index: Optional[int] = None,
) -> Optional[SelectorDefinition]:
    if not is_selector(property_value):
        print("Not a selector!")
        return None
    if "reference" in property_definition:
        print("reference in property_definition")
        allowed_references = [
            ReferenceDefinition(
                selected_element=property_definition["selected_element"],
                kind=[
                    Kind.model_validate(k) for k in property_definition.get("kind", [])
                ],
            )
        ]
        return SelectorDefinition(
            step_name=step_name,
            property=property_name,
            index=index,
            selector=property_value,
            allowed_references=allowed_references,
        )
    if "anyOf" not in property_definition and "oneOf" not in property_definition:
        raise ValueError(
            f"Detected reference provided for step {step_name} and property: {property_name} - "
            f"{property_value}, when step schema does not define reference as allowed value for"
            f"this field."
        )
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
    if len(allowed_references) == 0:
        raise ValueError(
            f"Detected reference provided for step {step_name} and property: {property_name} - "
            f"{property_value}, when step schema does not define reference as allowed value for"
            f"this field."
        )
    return SelectorDefinition(
        step_name=step_name,
        property=property_name,
        index=index,
        selector=property_value,
        allowed_references=allowed_references,
    )
