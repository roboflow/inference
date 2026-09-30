"""Deterministic translation of the V1 reference definitions into V2 JSON.

The V2 definition keeps the V1 inputs, outputs, step names and selectors.
Only these parts change:

| V1 | V2 |
| --- | --- |
| ``"version": "1.0"`` | ``"version": "2.0"`` |
| ``reference/<name>@v1`` fixture step | ``fixture/<name>@v1``, same parameters |
| ContinueIf with one ``(Number) >`` or ``<`` comparison | ``fixture/threshold_gate@v1`` with ``value``, ``threshold``, ``comparator`` |
| SwitchCase | ``fixture/switch@v1`` with ``value`` and ``cases`` |
| DimensionCollapse | ``fixture/collapse@v1`` with ``data`` |
| inner workflow | same type; the child definition is translated too |

Anything else raises ``TranslationError`` instead of being guessed.
"""

import copy
from typing import Any, Dict, Optional

V1_FIXTURE_PREFIX = "reference/"
V2_FIXTURE_PREFIX = "fixture/"
CONTINUE_IF = "roboflow_core/continue_if@v1"
SWITCH_CASE = "roboflow_core/switch_case@v1"
DIMENSION_COLLAPSE = "roboflow_core/dimension_collapse@v1"
INNER_WORKFLOW = "roboflow_core/inner_workflow@v1"
COMPARATORS = {"(Number) >": ">", "(Number) <": "<"}


class TranslationError(ValueError):
    """A V1 construct has no defined V2 translation."""


def translate_workflow(definition: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one V1 workflow definition, including nested children.

    Args:
        definition: V1 definition from the reference catalogue.

    Returns:
        A new V2 definition; the input is not modified.

    Raises:
        TranslationError: For a step form without a defined translation.
    """
    translated = copy.deepcopy(definition)
    translated["version"] = "2.0"
    translated["steps"] = [translate_step(step) for step in definition["steps"]]

    return translated


def translate_saved_workflows(
    saved_workflows: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Translate every definition a saved-workflow resolver can return.

    Args:
        saved_workflows: Workflow id mapped to its V1 definition, or ``None``.

    Returns:
        The same mapping with V2 definitions, or ``None``.
    """
    if saved_workflows is None:
        return None

    translated = {
        workflow_id: translate_workflow(definition)
        for workflow_id, definition in saved_workflows.items()
    }

    return translated


def translate_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one V1 step.

    Args:
        step: V1 step definition.

    Returns:
        The V2 step definition.

    Raises:
        TranslationError: For a step form without a defined translation.
    """
    step_type = step["type"]
    if step_type.startswith(V1_FIXTURE_PREFIX):
        fixture_step = dict(step)
        fixture_step["type"] = V2_FIXTURE_PREFIX + step_type[len(V1_FIXTURE_PREFIX) :]
        return fixture_step
    if step_type == CONTINUE_IF:
        return _translate_continue_if(step)
    if step_type == SWITCH_CASE:
        return _translate_switch_case(step)
    if step_type == DIMENSION_COLLAPSE:
        return {
            "type": "fixture/collapse@v1",
            "name": step["name"],
            "data": step["data"],
        }
    if step_type == INNER_WORKFLOW:
        nested_step = dict(step)
        if "workflow_definition" in step:
            nested_step["workflow_definition"] = translate_workflow(
                step["workflow_definition"]
            )
        return nested_step

    # Dynamic block types are defined by the workflow itself and stay as-is.
    return dict(step)


def _translate_continue_if(step: Dict[str, Any]) -> Dict[str, Any]:
    group = step["condition_statement"]
    statements = group.get("statements", [])
    if group.get("type") != "StatementGroup" or len(statements) != 1:
        raise TranslationError(f"{step['name']}: only one comparison is translated")

    statement = statements[0]
    comparator = COMPARATORS.get(statement["comparator"]["type"])
    operand = statement["left_operand"]
    parameters = step["evaluation_parameters"]
    if (
        comparator is None
        or operand.get("operations")
        or operand.get("operand_name") not in parameters
        or len(parameters) != 1
    ):
        raise TranslationError(f"{step['name']}: unsupported ContinueIf condition")

    gate = {
        "type": "fixture/threshold_gate@v1",
        "name": step["name"],
        "value": parameters[operand["operand_name"]],
        "threshold": statement["right_operand"]["value"],
        "comparator": comparator,
        "next_steps": list(step["next_steps"]),
    }

    return gate


def _translate_switch_case(step: Dict[str, Any]) -> Dict[str, Any]:
    if step.get("case_insensitive") or step.get("default_next_steps"):
        raise TranslationError(
            f"{step['name']}: only exact-match SwitchCase is translated"
        )

    switch = {
        "type": "fixture/switch@v1",
        "name": step["name"],
        "value": step["value"],
        "cases": dict(step["cases"]),
    }

    return switch
