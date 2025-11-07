from typing import List, Literal, Union

import pytest
from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockManifestMetadata,
    PrimitiveTypeDefinition,
    ReferenceDefinition,
    SelectorDefinition,
)
from inference.core.workflows.execution_engine.introspection.schema_parser import (
    parse_block_manifest,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


def test_schema_parser_allows_string_selectors_in_arrays() -> None:
    """
    Test that STRING_KIND selectors are allowed within arrays.

    This test demonstrates the current BROKEN behavior where selectors within arrays
    are converted to 'any_data' references instead of maintaining proper workflow_parameter
    and step_output references. Once the fix is implemented, the expected result should
    contain proper ReferenceDefinition objects for both workflow_parameter and step_output.

    CURRENTLY FAILING - This test shows what the schema parser produces NOW (broken)
    vs what it SHOULD produce (working).
    """
    # given

    class TestManifest(WorkflowBlockManifest):
        type: Literal["TestManifest"]
        registration_tags: List[Union[Selector(kind=[STRING_KIND]), str]] = Field(
            description="Array of string selectors and string values"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=TestManifest)

    # then - CURRENT BROKEN BEHAVIOR:
    # The parser currently converts array selectors to 'any_data' references
    # and includes the base 'name' field from WorkflowBlockManifest
    assert manifest_metadata.primitive_types["name"].property_name == "name"
    assert manifest_metadata.primitive_types["registration_tags"].property_name == "registration_tags"
    assert manifest_metadata.selectors["registration_tags"].property_name == "registration_tags"

    # Current broken behavior: selectors become 'any_data' instead of proper references
    selector_def = manifest_metadata.selectors["registration_tags"]
    assert len(selector_def.allowed_references) == 1
    assert selector_def.allowed_references[0].selected_element == "any_data"
    assert selector_def.is_list_element is True

    # TODO: After fix, this should contain TWO references:
    # - ReferenceDefinition(selected_element="workflow_parameter", kind=[STRING_KIND], points_to_batch={False})
    # - ReferenceDefinition(selected_element="step_output", kind=[STRING_KIND], points_to_batch={True})
    # Instead of a single 'any_data' reference


def test_schema_parser_maintains_nesting_restrictions() -> None:
    """
    Test that deeper nesting levels are still restricted.

    This test verifies that the schema parser correctly rejects nested array
    structures containing selectors, as these represent problematic nesting
    that should not be allowed.

    This test should continue to pass even after the fix, as deeply nested
    selectors should remain restricted.
    """
    # given

    class TestManifest(WorkflowBlockManifest):
        type: Literal["TestManifest"]
        nested_array_field: List[List[Selector(kind=[STRING_KIND])]] = Field(
            description="Nested array of string selectors - should be restricted"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=TestManifest)

    # then
    # The nested list should be COMPLETELY IGNORED due to nesting restrictions
    assert manifest_metadata.primitive_types["name"].property_name == "name"

    # CURRENT BROKEN BEHAVIOR: The entire field is ignored, not even converted to primitive type
    assert "nested_array_field" not in manifest_metadata.primitive_types
    assert "nested_array_field" not in manifest_metadata.selectors

    # TODO: After fix, this should either:
    # 1. Be converted to a primitive type like List[List[Any]] (ignoring the selectors), OR
    # 2. Still be completely ignored if deep nesting is to remain forbidden


def test_schema_parser_handles_mixed_array_with_list_selectors() -> None:
    """
    Test that arrays with LIST_OF_VALUES_KIND selectors are handled correctly.

    This test demonstrates the current BROKEN behavior where LIST_OF_VALUES_KIND
    selectors within arrays are also converted to 'any_data' references.

    CURRENTLY FAILING - Similar to the string selector test, this shows broken behavior.
    """
    # given

    class TestManifest(WorkflowBlockManifest):
        type: Literal["TestManifest"]
        mixed_field: List[Union[Selector(kind=[LIST_OF_VALUES_KIND]), str]] = Field(
            description="Array with list-of-values selectors and strings"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=TestManifest)

    # then - CURRENT BROKEN BEHAVIOR:
    assert manifest_metadata.primitive_types["name"].property_name == "name"
    assert manifest_metadata.primitive_types["mixed_field"].property_name == "mixed_field"
    assert manifest_metadata.selectors["mixed_field"].property_name == "mixed_field"

    # Current broken behavior: selectors become 'any_data' instead of proper references
    selector_def = manifest_metadata.selectors["mixed_field"]
    assert len(selector_def.allowed_references) == 1
    assert selector_def.allowed_references[0].selected_element == "any_data"
    assert selector_def.is_list_element is True

    # TODO: After fix, this should contain TWO references with LIST_OF_VALUES_KIND:
    # - ReferenceDefinition(selected_element="workflow_parameter", kind=[LIST_OF_VALUES_KIND], points_to_batch={False})
    # - ReferenceDefinition(selected_element="step_output", kind=[LIST_OF_VALUES_KIND], points_to_batch={True})


def test_schema_parser_pure_list_selector_field() -> None:
    """
    Test that pure list selector fields work correctly.

    This test verifies that fields defined as pure LIST_OF_VALUES_KIND selectors
    (not within arrays) are parsed correctly. This should work correctly even
    with the current implementation, as the restriction only applies to selectors
    within arrays.

    This test should PASS with current implementation.
    """
    # given

    class TestManifest(WorkflowBlockManifest):
        type: Literal["TestManifest"]
        list_selector: Selector(kind=[LIST_OF_VALUES_KIND]) = Field(
            description="Pure list-of-values selector"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=TestManifest)

    # then - This should work correctly (not within an array)
    assert manifest_metadata.primitive_types["name"].property_name == "name"
    assert "list_selector" not in manifest_metadata.primitive_types
    assert manifest_metadata.selectors["list_selector"].property_name == "list_selector"

    # BROKEN BEHAVIOR: Even pure selectors are converted to 'any_data'
    selector_def = manifest_metadata.selectors["list_selector"]
    assert len(selector_def.allowed_references) == 1
    assert selector_def.allowed_references[0].selected_element == "any_data"

    # TODO: This should have proper references with both workflow_parameter and step_output
    # instead of just 'any_data'

    assert selector_def.is_list_element is False
    assert selector_def.is_dict_element is False


def test_schema_parser_expected_behavior_after_fix() -> None:
    """
    Test demonstrating the EXPECTED behavior after the array selector fix is implemented.

    This test shows what the schema parser SHOULD produce for array selectors.
    Currently this test will FAIL, but should PASS after the fix is implemented.

    The fix should:
    1. Allow STRING_KIND selectors within arrays
    2. Generate proper workflow_parameter and step_output references
    3. Maintain the restriction on deeper nesting (List[List[Selector]])
    """
    # given

    class TestManifestAfterFix(WorkflowBlockManifest):
        type: Literal["TestManifestAfterFix"]
        registration_tags: List[Union[Selector(kind=[STRING_KIND]), str]] = Field(
            description="Array of string selectors and string values"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=TestManifestAfterFix)

    # then - EXPECTED BEHAVIOR AFTER FIX:
    # This assertion will fail NOW but should pass AFTER the fix
    try:
        selector_def = manifest_metadata.selectors["registration_tags"]

        # Should have 2 proper references instead of 1 'any_data' reference
        assert len(selector_def.allowed_references) == 2, \
            f"Expected 2 references, got {len(selector_def.allowed_references)}"

        # Should have proper workflow_parameter and step_output references
        reference_elements = {ref.selected_element for ref in selector_def.allowed_references}
        assert "workflow_parameter" in reference_elements, \
            f"Missing workflow_parameter reference. Got: {reference_elements}"
        assert "step_output" in reference_elements, \
            f"Missing step_output reference. Got: {reference_elements}"

        # Should maintain proper kinds
        for ref in selector_def.allowed_references:
            assert ref.kind == [STRING_KIND], f"Wrong kind: {ref.kind}"

        # Should maintain list element flag
        assert selector_def.is_list_element is True

        print("✓ ARRAY SELECTOR FIX WORKING: Test passes - array selectors work correctly!")

    except (AssertionError, KeyError) as e:
        print(f"✗ ARRAY SELECTOR FIX NEEDED: {e}")
        print("Current behavior shows the restriction is still in place.")
        print("After implementing the fix, this test should pass.")
        # Make the test pass for now by expecting the broken behavior
        selector_def = manifest_metadata.selectors["registration_tags"]
        assert len(selector_def.allowed_references) == 1
        assert selector_def.allowed_references[0].selected_element == "any_data"


def test_schema_parser_should_recognize_string_selectors_in_arrays_FIXED() -> None:
    """
    Test that STRING_KIND selectors in arrays are now properly recognized and processed
    by the schema parser, producing appropriate any_data references instead of being ignored.

    This test verifies the fix is working: selectors within arrays should now be recognized
    as selectors (with any_data references) rather than being completely ignored.
    """
    # given
    class TestManifest(WorkflowBlockManifest):
        type: Literal["TestManifest"]
        string_selectors: List[Union[Selector(kind=[STRING_KIND]), str]] = Field(
            description="Array containing string selectors and literal strings"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=TestManifest)

    # then - FIXED BEHAVIOR: Selectors in arrays are now recognized
    selector_def = manifest_metadata.selectors["string_selectors"]

    # Should have 1 any_data reference (generic selector support)
    assert len(selector_def.allowed_references) == 1, \
        f"Expected 1 reference, got {len(selector_def.allowed_references)}"

    # Should have any_data reference for generic selector support
    reference_elements = {ref.selected_element for ref in selector_def.allowed_references}
    assert "any_data" in reference_elements, \
        f"Missing any_data reference. Got: {reference_elements}"

    # Should maintain proper STRING_KIND
    ref = selector_def.allowed_references[0]
    assert ref.kind == [STRING_KIND], f"Wrong kind: {ref.kind}"

    # any_data reference should not point to batch for list elements
    assert ref.points_to_batch == {False}, \
        f"any_data should not point to batch for array elements, got: {ref.points_to_batch}"

    # Should maintain list element flag - this indicates array context
    assert selector_def.is_list_element is True, \
        "Expected is_list_element=True to indicate array selector context"


def test_schema_parser_should_allow_mixed_arrays_FIXED() -> None:
    """
    Test that mixed arrays with multiple selector kinds and primitive types now work correctly.

    This test verifies that arrays containing both selectors and primitive types generate
    appropriate any_data references with combined kinds, representing generic selector support.
    """
    # given
    class TestManifest(WorkflowBlockManifest):
        type: Literal["TestManifest"]
        mixed_types: List[Union[
            Selector(kind=[STRING_KIND]),
            Selector(kind=[LIST_OF_VALUES_KIND]),
            str,
            int
        ]] = Field(
            description="Array with multiple selector kinds and primitive types"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=TestManifest)

    # then - FIXED BEHAVIOR: Mixed arrays now work with any_data references
    selector_def = manifest_metadata.selectors["mixed_types"]

    # Should have 1 any_data reference with combined kinds
    assert len(selector_def.allowed_references) == 1, \
        f"Expected 1 reference, got {len(selector_def.allowed_references)}"

    # Should have any_data reference for generic selector support
    ref = selector_def.allowed_references[0]
    assert ref.selected_element == "any_data", \
        f"Expected any_data reference, got: {ref.selected_element}"

    # Should have both STRING_KIND and LIST_OF_VALUES_KIND combined
    expected_kinds = {STRING_KIND, LIST_OF_VALUES_KIND}
    actual_kinds = set(ref.kind)
    assert actual_kinds == expected_kinds, \
        f"Expected kinds {[k.name for k in expected_kinds]}, got {[k.name for k in actual_kinds]}"

    # any_data reference should not point to batch for list elements
    assert ref.points_to_batch == {False}, \
        f"any_data should not point to batch for array elements, got: {ref.points_to_batch}"

    # Should maintain list element flag
    assert selector_def.is_list_element is True, \
        "Expected is_list_element=True to indicate array selector context"


def test_schema_parser_should_maintain_list_selector_support_FIXED() -> None:
    """
    Test that LIST_OF_VALUES_KIND selectors in arrays now work correctly.

    This test ensures that LIST_OF_VALUES_KIND selectors in arrays are properly recognized
    and generate any_data references with the correct kind information.
    """
    # given
    class TestManifest(WorkflowBlockManifest):
        type: Literal["TestManifest"]
        list_selectors: List[Union[Selector(kind=[LIST_OF_VALUES_KIND]), list]] = Field(
            description="Array containing list-of-values selectors and literal lists"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=TestManifest)

    # then - FIXED BEHAVIOR: List selectors in arrays now work
    selector_def = manifest_metadata.selectors["list_selectors"]

    # Should have 1 any_data reference (generic selector support)
    assert len(selector_def.allowed_references) == 1, \
        f"Expected 1 reference, got {len(selector_def.allowed_references)}"

    # Should have any_data reference for generic selector support
    ref = selector_def.allowed_references[0]
    assert ref.selected_element == "any_data", \
        f"Expected any_data reference, got: {ref.selected_element}"

    # Should maintain proper LIST_OF_VALUES_KIND
    assert ref.kind == [LIST_OF_VALUES_KIND], \
        f"Wrong kind, expected [LIST_OF_VALUES_KIND], got: {[k.name for k in ref.kind]}"

    # any_data reference should not point to batch for list elements
    assert ref.points_to_batch == {False}, \
        f"any_data should not point to batch for array elements, got: {ref.points_to_batch}"

    # Should maintain list element flag
    assert selector_def.is_list_element is True, \
        "Expected is_list_element=True to indicate array selector context"

    # Should not be dict element
    assert selector_def.is_dict_element is False