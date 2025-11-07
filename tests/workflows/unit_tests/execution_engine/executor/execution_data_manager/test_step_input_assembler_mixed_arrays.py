from typing import Any, Dict

import pytest

from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompoundStepInputDefinition,
    DynamicStepInputDefinition,
    ListOfStepInputDefinitions,
    NodeCategory,
    NodeInputCategory,
    ParameterSpecification,
    StaticStepInputDefinition,
    StepNode,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.execution_cache import (
    ExecutionCache,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.step_input_assembler import (
    construct_non_simd_step_compound_list_input,
)


def create_test_step_node() -> StepNode:
    """Create a minimal StepNode for testing."""
    return StepNode(
        node_category=NodeCategory.STEP_NODE,
        name="test_step",
        selector="$steps.test_step",
        data_lineage=["<workflow_input>"],
        step_manifest=None,  # Not needed for these tests
        input_data={},
        step_execution_dimensionality=0,
        batch_oriented_parameters=set(),
        auto_batch_casting_lineage_supports={},
    )


def create_string_input_definition(selector: str) -> DynamicStepInputDefinition:
    """Create a DynamicStepInputDefinition for a string selector."""
    return DynamicStepInputDefinition(
        parameter_specification=ParameterSpecification(
            parameter_name=selector.split(".")[-1],  # Extract parameter name
            nested_element_key=None,
            nested_element_index=None,
        ),
        category=NodeInputCategory.NON_BATCH_INPUT_PARAMETER,
        data_lineage=[],
        selector=selector,
    )


def create_list_input_definition(selector: str) -> DynamicStepInputDefinition:
    """Create a DynamicStepInputDefinition for a list selector (LIST_OF_VALUES_KIND)."""
    return DynamicStepInputDefinition(
        parameter_specification=ParameterSpecification(
            parameter_name=selector.split(".")[-1],  # Extract parameter name
            nested_element_key=None,
            nested_element_index=None,
        ),
        category=NodeInputCategory.BATCH_INPUT_PARAMETER,  # LIST_OF_VALUES_KIND typically maps to BATCH_INPUT_PARAMETER
        data_lineage=["batch"],  # Indicates batch orientation
        selector=selector,
    )


def create_static_input_definition(value: Any) -> StaticStepInputDefinition:
    """Create a StaticStepInputDefinition for a literal value."""
    return StaticStepInputDefinition(
        parameter_specification=ParameterSpecification(
            parameter_name="static",
            nested_element_key=None,
            nested_element_index=None,
        ),
        category=NodeInputCategory.STATIC_VALUE,
        value=value,
    )


def test_mixed_array_with_string_selectors() -> None:
    """
    Test successful resolution of mixed arrays with string selectors.

    Given: ["literal", "$inputs.string_tag"] where string_tag = "value"
    Expected: ["literal", "value"]
    """
    # given
    step_node = create_test_step_node()

    # Create compound input definition representing ["literal", "$inputs.string_tag"]
    nested_definitions = [
        create_static_input_definition("literal"),
        create_string_input_definition("$inputs.string_tag"),
    ]
    parameter_spec = ListOfStepInputDefinitions(
        name="mixed_array",
        nested_definitions=nested_definitions,
    )

    runtime_parameters = {"string_tag": "value"}
    execution_cache = ExecutionCache(
        cache_content={},
        batches_compatibility={},
        step_outputs_registered=set(),
    )

    # when
    result, contains_empty_selector = construct_non_simd_step_compound_list_input(
        step_node=step_node,
        parameter_spec=parameter_spec,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
    )

    # then
    assert result == ["literal", "value"], "Expected mixed array to resolve correctly"
    assert contains_empty_selector is False, "Expected no empty selectors"


def test_mixed_array_with_list_selector_raises_error() -> None:
    """
    Test that LIST_OF_VALUES_KIND selectors in mixed arrays raise clear errors.

    Given: ["literal", "$inputs.list_tags"] where list_tags = ["a", "b"]
    Expected: ExecutionEngineRuntimeError with specific message
    """
    # given
    step_node = create_test_step_node()

    # Create compound input definition representing ["literal", "$inputs.list_tags"]
    nested_definitions = [
        create_static_input_definition("literal"),
        create_list_input_definition("$inputs.list_tags"),  # This should cause an error
    ]
    parameter_spec = ListOfStepInputDefinitions(
        name="mixed_array",
        nested_definitions=nested_definitions,
    )

    runtime_parameters = {"list_tags": ["a", "b"]}
    execution_cache = ExecutionCache(
        cache_content={},
        batches_compatibility={},
        step_outputs_registered=set(),
    )

    # when / then
    with pytest.raises(ExecutionEngineRuntimeError) as exc_info:
        construct_non_simd_step_compound_list_input(
            step_node=step_node,
            parameter_spec=parameter_spec,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
        )

    # Verify the error message is descriptive and mentions the specific validation issue
    error_message = str(exc_info.value)
    assert "Invalid mixed array" in error_message, (
        "Expected error message to mention invalid mixed array"
    )
    assert "LIST_OF_VALUES_KIND selector" in error_message, (
        "Expected error message to mention LIST_OF_VALUES_KIND selector issue"
    )


def test_pure_string_selector_still_works() -> None:
    """
    Test that pure string selectors continue to work.

    Given: "$inputs.string_tag" where string_tag = "value"
    Expected: "value"
    """
    # given
    step_node = create_test_step_node()

    # Create compound input definition representing a single string selector (not really compound, but testing the path)
    nested_definitions = [
        create_string_input_definition("$inputs.string_tag"),
    ]
    parameter_spec = ListOfStepInputDefinitions(
        name="string_array",
        nested_definitions=nested_definitions,
    )

    runtime_parameters = {"string_tag": "value"}
    execution_cache = ExecutionCache(
        cache_content={},
        batches_compatibility={},
        step_outputs_registered=set(),
    )

    # when
    result, contains_empty_selector = construct_non_simd_step_compound_list_input(
        step_node=step_node,
        parameter_spec=parameter_spec,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
    )

    # then
    assert result == ["value"], "Expected single string selector to work"
    assert contains_empty_selector is False, "Expected no empty selectors"


def test_pure_list_selector_still_works() -> None:
    """
    Test that pure list selectors continue to work when not in mixed context.

    Note: This test demonstrates the current behavior. In the actual implementation,
    pure list selectors should work, but the current step_input_assembler doesn't
    handle this case properly due to batch orientation restrictions.

    Given: "$inputs.list_tags" where list_tags = ["a", "b", "c"]
    Expected: Currently raises error due to batch orientation, but should work after fix
    """
    # given
    step_node = create_test_step_node()

    # Create compound input definition representing a single list selector
    nested_definitions = [
        create_list_input_definition("$inputs.list_tags"),
    ]
    parameter_spec = ListOfStepInputDefinitions(
        name="list_array",
        nested_definitions=nested_definitions,
    )

    runtime_parameters = {"list_tags": ["a", "b", "c"]}
    execution_cache = ExecutionCache(
        cache_content={},
        batches_compatibility={},
        step_outputs_registered=set(),
    )

    # when / then
    # NOTE: This currently raises an error due to batch orientation restrictions,
    # but after the validation fix, pure list selectors should work
    with pytest.raises(ExecutionEngineRuntimeError) as exc_info:
        construct_non_simd_step_compound_list_input(
            step_node=step_node,
            parameter_spec=parameter_spec,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
        )

    error_message = str(exc_info.value)
    assert "batch-oriented input for non-SIMD step" in error_message, (
        "Expected current implementation to reject batch-oriented inputs"
    )

    # TODO: After the validation fix, this test should be updated to:
    # assert result == [["a", "b", "c"]], "Expected pure list selector to work"
    # assert contains_empty_selector is False, "Expected no empty selectors"


def test_multiple_string_selectors_in_array() -> None:
    """
    Test multiple string selectors in an array work correctly.

    Given: ["$inputs.first", "$inputs.second"] where first = "A", second = "B"
    Expected: ["A", "B"]
    """
    # given
    step_node = create_test_step_node()

    nested_definitions = [
        create_string_input_definition("$inputs.first"),
        create_string_input_definition("$inputs.second"),
    ]
    parameter_spec = ListOfStepInputDefinitions(
        name="multi_string_array",
        nested_definitions=nested_definitions,
    )

    runtime_parameters = {"first": "A", "second": "B"}
    execution_cache = ExecutionCache(
        cache_content={},
        batches_compatibility={},
        step_outputs_registered=set(),
    )

    # when
    result, contains_empty_selector = construct_non_simd_step_compound_list_input(
        step_node=step_node,
        parameter_spec=parameter_spec,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
    )

    # then
    assert result == ["A", "B"], "Expected multiple string selectors to resolve correctly"
    assert contains_empty_selector is False, "Expected no empty selectors"


def test_mixed_array_with_multiple_literals_and_string_selector() -> None:
    """
    Test mixed array with multiple literal values and a string selector.

    Given: ["literal1", "literal2", "$inputs.string_tag"] where string_tag = "value"
    Expected: ["literal1", "literal2", "value"]
    """
    # given
    step_node = create_test_step_node()

    nested_definitions = [
        create_static_input_definition("literal1"),
        create_static_input_definition("literal2"),
        create_string_input_definition("$inputs.string_tag"),
    ]
    parameter_spec = ListOfStepInputDefinitions(
        name="complex_mixed_array",
        nested_definitions=nested_definitions,
    )

    runtime_parameters = {"string_tag": "value"}
    execution_cache = ExecutionCache(
        cache_content={},
        batches_compatibility={},
        step_outputs_registered=set(),
    )

    # when
    result, contains_empty_selector = construct_non_simd_step_compound_list_input(
        step_node=step_node,
        parameter_spec=parameter_spec,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
    )

    # then
    assert result == ["literal1", "literal2", "value"], (
        "Expected complex mixed array to resolve correctly"
    )
    assert contains_empty_selector is False, "Expected no empty selectors"


def test_mixed_array_should_reject_list_selectors_FIXED() -> None:
    """
    FIXED TEST: Test that validates and rejects LIST_OF_VALUES_KIND selectors in mixed arrays.

    This test demonstrates the validation logic that now works correctly.

    Given: ["literal", "$inputs.list_tags"] where list_tags = ["a", "b"]
    Expected: ExecutionEngineRuntimeError with specific message about mixed array restrictions
    """
    # given
    step_node = create_test_step_node()

    # Create compound input definition representing ["literal", "$inputs.list_tags"]
    nested_definitions = [
        create_static_input_definition("literal"),
        create_list_input_definition("$inputs.list_tags"),  # LIST_OF_VALUES_KIND selector in mixed array
    ]
    parameter_spec = ListOfStepInputDefinitions(
        name="mixed_array",
        nested_definitions=nested_definitions,
    )

    runtime_parameters = {"list_tags": ["a", "b"]}
    execution_cache = ExecutionCache(
        cache_content={},
        batches_compatibility={},
        step_outputs_registered=set(),
    )

    # when / then
    # This now raises an ExecutionEngineRuntimeError with a clear message
    with pytest.raises(ExecutionEngineRuntimeError) as exc_info:
        construct_non_simd_step_compound_list_input(
            step_node=step_node,
            parameter_spec=parameter_spec,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
        )

    # The error message should specifically mention mixed array restrictions
    error_message = str(exc_info.value)
    assert "Invalid mixed array in step 'test_step'" in error_message, (
        f"Expected error message to mention invalid mixed array, got: {error_message}"
    )
    assert "Array elements can only contain string literals or STRING_KIND selectors" in error_message, (
        f"Expected error message to explain valid array elements, got: {error_message}"
    )
    assert "$inputs.list_tags" in error_message, (
        f"Expected error message to identify the problematic selector, got: {error_message}"
    )


def test_mixed_array_validation_should_be_enforced_FIXED() -> None:
    """
    FIXED TEST: Test that demonstrates the validation logic that now works correctly.

    This test shows how the validation works - detecting mixed arrays and
    enforcing restrictions on LIST_OF_VALUES_KIND selectors.

    Given: [42, "$inputs.list_values", "text"] where list_values = [1, 2, 3]
    Expected: ExecutionEngineRuntimeError about mixed array validation
    """
    # given
    step_node = create_test_step_node()

    # Create a mixed array with literal values and a LIST_OF_VALUES_KIND selector
    nested_definitions = [
        create_static_input_definition(42),  # Literal integer
        create_list_input_definition("$inputs.list_values"),  # LIST_OF_VALUES_KIND selector
        create_static_input_definition("text"),  # Literal string
    ]
    parameter_spec = ListOfStepInputDefinitions(
        name="mixed_array_with_list",
        nested_definitions=nested_definitions,
    )

    runtime_parameters = {"list_values": [1, 2, 3]}
    execution_cache = ExecutionCache(
        cache_content={},
        batches_compatibility={},
        step_outputs_registered=set(),
    )

    # when / then
    # This now raises ExecutionEngineRuntimeError with clear validation message
    with pytest.raises(ExecutionEngineRuntimeError) as exc_info:
        construct_non_simd_step_compound_list_input(
            step_node=step_node,
            parameter_spec=parameter_spec,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
        )

    error_message = str(exc_info.value)
    # The validation detects this as a mixed array and rejects the LIST_OF_VALUES_KIND selector
    assert "Invalid mixed array in step 'test_step'" in error_message, (
        f"Expected validation error about mixed arrays, got: {error_message}"
    )
    assert "Array elements can only contain string literals or STRING_KIND selectors" in error_message, (
        f"Expected clear guidance about valid elements, got: {error_message}"
    )
    assert "$inputs.list_values" in error_message, (
        f"Expected error message to identify the problematic selector, got: {error_message}"
    )


def test_array_selector_validation_should_work_FIXED() -> None:
    """
    FIXED TEST: Test that shows how the validation now works correctly.

    This test demonstrates that the validation:
    1. Allows STRING_KIND selectors in mixed arrays
    2. Rejects LIST_OF_VALUES_KIND selectors in mixed arrays with clear error messages
    3. Provides helpful context about what went wrong

    Expected behavior that now works correctly.
    """
    # given
    step_node = create_test_step_node()

    # Test case 1: Mixed array with STRING_KIND selector - should work
    valid_nested_definitions = [
        create_static_input_definition("prefix"),
        create_string_input_definition("$inputs.string_value"),  # STRING_KIND - should be allowed
        create_static_input_definition("suffix"),
    ]
    valid_parameter_spec = ListOfStepInputDefinitions(
        name="valid_mixed_array",
        nested_definitions=valid_nested_definitions,
    )

    runtime_parameters = {"string_value": "middle", "list_value": ["x", "y"]}
    execution_cache = ExecutionCache(
        cache_content={},
        batches_compatibility={},
        step_outputs_registered=set(),
    )

    # This now works (STRING_KIND in mixed array is allowed)
    result, _ = construct_non_simd_step_compound_list_input(
        step_node=step_node,
        parameter_spec=valid_parameter_spec,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
    )
    assert result == ["prefix", "middle", "suffix"], "STRING_KIND selectors should work in mixed arrays"

    # Test case 2: Mixed array with LIST_OF_VALUES_KIND selector - should fail with specific error
    invalid_nested_definitions = [
        create_static_input_definition("prefix"),
        create_list_input_definition("$inputs.list_value"),  # LIST_OF_VALUES_KIND - should be rejected
        create_static_input_definition("suffix"),
    ]
    invalid_parameter_spec = ListOfStepInputDefinitions(
        name="invalid_mixed_array",
        nested_definitions=invalid_nested_definitions,
    )

    # This now raises ExecutionEngineRuntimeError with helpful message
    with pytest.raises(ExecutionEngineRuntimeError) as exc_info:
        construct_non_simd_step_compound_list_input(
            step_node=step_node,
            parameter_spec=invalid_parameter_spec,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
        )

    error_message = str(exc_info.value)
    # Validation provides clear, helpful error message
    assert "Invalid mixed array in step 'test_step'" in error_message, (
        f"Expected clear validation error message, got: {error_message}"
    )
    assert "$inputs.list_value" in error_message, (
        "Expected error message to identify the problematic selector"
    )
    assert "Array elements can only contain string literals or STRING_KIND selectors" in error_message, (
        "Expected error message to explain valid array elements"
    )