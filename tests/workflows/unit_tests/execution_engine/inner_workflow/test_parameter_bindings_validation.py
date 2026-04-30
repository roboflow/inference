import pytest

from inference.core.workflows.execution_engine.entities.base import (
    WorkflowImage,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge import (
    validate_parameter_bindings_against_child,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowParameterBindingsMissingRequiredError,
    InnerWorkflowParameterBindingsUnknownInputError,
)


def _parsed(
    *,
    inputs: list,
) -> ParsedWorkflowDefinition:
    return ParsedWorkflowDefinition(
        version="1.0",
        inputs=inputs,
        steps=[],
        outputs=[],
    )


def test_accepts_full_bindings_matching_all_inputs() -> None:
    child = _parsed(
        inputs=[
            WorkflowParameter(
                type="WorkflowParameter",
                name="a",
                kind=[],
                default_value=None,
            ),
            WorkflowParameter(
                type="WorkflowParameter",
                name="b",
                kind=[],
                default_value="fallback",
            ),
        ],
    )
    validate_parameter_bindings_against_child(
        bindings={"a": "$inputs.x", "b": "$inputs.y"},
        child_parsed=child,
        step_name="inner",
    )


def test_allows_omitting_workflow_parameter_with_non_null_default() -> None:
    child = _parsed(
        inputs=[
            WorkflowParameter(
                type="WorkflowParameter",
                name="required",
                kind=[],
                default_value=None,
            ),
            WorkflowParameter(
                type="WorkflowParameter",
                name="optional_with_default",
                kind=[],
                default_value="from-child-default",
            ),
        ],
    )
    validate_parameter_bindings_against_child(
        bindings={"required": "$inputs.parent_field"},
        child_parsed=child,
        step_name="inner",
    )


def test_rejects_unknown_binding_keys() -> None:
    child = _parsed(
        inputs=[
            WorkflowParameter(
                type="WorkflowParameter",
                name="only",
                kind=[],
                default_value=None,
            ),
        ],
    )
    with pytest.raises(
        InnerWorkflowParameterBindingsUnknownInputError, match="unknown"
    ):
        validate_parameter_bindings_against_child(
            bindings={"only": "$inputs.a", "extra": "$inputs.b"},
            child_parsed=child,
            step_name="inner",
        )


def test_rejects_missing_required_workflow_parameter() -> None:
    child = _parsed(
        inputs=[
            WorkflowParameter(
                type="WorkflowParameter",
                name="a",
                kind=[],
                default_value=None,
            ),
            WorkflowParameter(
                type="WorkflowParameter",
                name="b",
                kind=[],
                default_value=None,
            ),
        ],
    )
    with pytest.raises(
        InnerWorkflowParameterBindingsMissingRequiredError,
        match="missing parameter_bindings",
    ):
        validate_parameter_bindings_against_child(
            bindings={"a": "$inputs.x"},
            child_parsed=child,
            step_name="inner",
        )


def test_workflow_image_input_always_requires_binding() -> None:
    child = _parsed(
        inputs=[
            WorkflowImage(
                type="WorkflowImage",
                name="image",
                kind=[],
            ),
        ],
    )
    with pytest.raises(
        InnerWorkflowParameterBindingsMissingRequiredError,
        match="missing parameter_bindings",
    ):
        validate_parameter_bindings_against_child(
            bindings={},
            child_parsed=child,
            step_name="inner",
        )


def test_inference_parameter_type_alias_same_as_workflow_parameter() -> None:
    child = _parsed(
        inputs=[
            WorkflowParameter(
                type="InferenceParameter",
                name="p",
                kind=[],
                default_value=42,
            ),
        ],
    )
    validate_parameter_bindings_against_child(
        bindings={},
        child_parsed=child,
        step_name="inner",
    )
