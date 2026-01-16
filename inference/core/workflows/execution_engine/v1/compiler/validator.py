import os
from typing import Dict, List, Optional, Type

from inference.core.workflows.errors import DuplicatedNameError, WorkflowDefinitionError
from inference.core.workflows.execution_engine.entities.base import InputType, JsonField
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    get_input_parameters_selectors,
    get_output_selectors,
    get_steps_selectors,
)
from inference.core.workflows.prototypes.block import WorkflowBlock, WorkflowBlockManifest


@execution_phase(
    name="workflow_definition_validation",
    categories=["execution_engine_operation"],
)
def validate_workflow_specification(
    workflow_definition: ParsedWorkflowDefinition,
    available_blocks: Optional[List[BlockSpecification]] = None,
    profiler: Optional[WorkflowsProfiler] = None,
) -> None:
    validate_inputs_names_are_unique(inputs=workflow_definition.inputs)
    validate_steps_names_are_unique(steps=workflow_definition.steps)
    validate_outputs_names_are_unique(outputs=workflow_definition.outputs)
    if available_blocks is not None:
        validate_disabled_blocks(
            workflow_definition=workflow_definition,
            available_blocks=available_blocks
        )


def validate_inputs_names_are_unique(inputs: List[InputType]) -> None:
    input_parameters_selectors = get_input_parameters_selectors(inputs=inputs)
    if len(input_parameters_selectors) != len(inputs):
        raise DuplicatedNameError(
            public_message="Found duplicated input parameter names",
            context="workflow_compilation | specification_validation",
        )


def validate_steps_names_are_unique(steps: List[WorkflowBlockManifest]) -> None:
    steps_selectors = get_steps_selectors(steps=steps)
    if len(steps_selectors) != len(steps):
        raise DuplicatedNameError(
            public_message="Found duplicated input steps names",
            context="workflow_compilation | specification_validation",
        )


def validate_outputs_names_are_unique(outputs: List[JsonField]) -> None:
    output_names = get_output_selectors(outputs=outputs)
    if len(output_names) != len(outputs):
        raise DuplicatedNameError(
            public_message="Found duplicated input outputs names",
            context="workflow_compilation | specification_validation",
        )


def validate_disabled_blocks(
    workflow_definition: ParsedWorkflowDefinition,
    available_blocks: List[BlockSpecification],
) -> None:
    """
    Validates that workflow doesn't contain disabled blocks based on configuration.
    
    This function checks if the workflow contains blocks that have been disabled
    via environment configuration. This is useful for:
    - Preventing duplicate side effects in mirroring/testing scenarios
    - Enforcing security policies (e.g., no external API calls)
    - Cost control (e.g., disabling expensive foundation models)
    - Environment-specific restrictions (e.g., no sinks in development)
    
    Raises:
        WorkflowDefinitionError: If disabled blocks are found
    """
    # Check if selective block disabling is enabled
    if not _is_selective_blocks_disabled():
        return
    
    # Get disabled block types and patterns from environment
    disabled_types = _get_disabled_block_types()
    disabled_patterns = _get_disabled_block_patterns()
    disable_reason = _get_disable_reason()
    
    # Build a map of block identifiers to their specifications
    blocks_by_identifier = {
        block.identifier: block for block in available_blocks
    }
    
    # Check each step in the workflow
    for step in workflow_definition.steps:
        step_type = step.type
        
        # Check against specific patterns (e.g., specific block identifiers)
        for pattern in disabled_patterns:
            if pattern.lower() in step_type.lower():
                raise WorkflowDefinitionError(
                    public_message=(
                        f"Block type '{step_type}' is not allowed. "
                        f"Matched disabled pattern '{pattern}'. "
                        f"{disable_reason}"
                    ),
                    context="workflow_compilation | block_validation",
                )
        
        # Check block category from manifest
        if step_type in blocks_by_identifier:
            block_spec = blocks_by_identifier[step_type]
            block_type = _get_block_type_from_specification(block_spec)
            
            if block_type and block_type.lower() in disabled_types:
                raise WorkflowDefinitionError(
                    public_message=(
                        f"Block type '{step_type}' (category: {block_type}) is not allowed. "
                        f"Blocks of type '{block_type}' are disabled. "
                        f"{disable_reason}"
                    ),
                    context="workflow_compilation | block_validation",
                )


def _is_selective_blocks_disabled() -> bool:
    """Check if selective block disabling is enabled."""
    # Support both old and new environment variable names for backward compatibility
    return (
        os.environ.get("WORKFLOW_SELECTIVE_BLOCKS_DISABLE", "false").lower() == "true" or
        os.environ.get("WORKFLOW_MIRRORING_MODE", "false").lower() == "true"
    )


def _get_disabled_block_types() -> List[str]:
    """Get list of disabled block type categories from environment."""
    # Support both old and new environment variable names
    disabled_types_str = os.environ.get(
        "WORKFLOW_DISABLED_BLOCK_TYPES",
        os.environ.get("WORKFLOW_BLOCKED_BLOCK_TYPES", "")
    )
    
    # If no explicit configuration, don't disable any types by default
    # This allows users to be explicit about what they want to disable
    if not disabled_types_str:
        return []
    
    return [t.strip().lower() for t in disabled_types_str.split(",") if t.strip()]


def _get_disabled_block_patterns() -> List[str]:
    """Get list of disabled block identifier patterns from environment."""
    # Support both old and new environment variable names
    disabled_patterns_str = os.environ.get(
        "WORKFLOW_DISABLED_BLOCK_PATTERNS",
        os.environ.get("WORKFLOW_BLOCKED_BLOCK_PATTERNS", "")
    )
    
    # No default patterns - users should explicitly specify what to disable
    if not disabled_patterns_str:
        return []
    
    return [p.strip().lower() for p in disabled_patterns_str.split(",") if p.strip()]


def _get_disable_reason() -> str:
    """Get the reason for disabling blocks from environment."""
    reason = os.environ.get(
        "WORKFLOW_DISABLE_REASON",
        "These blocks are disabled by system configuration."
    )
    return reason


def _get_block_type_from_specification(block_spec: BlockSpecification) -> Optional[str]:
    """
    Extract the block type (e.g., 'sink', 'model') from a BlockSpecification.
    
    The block type is stored in the manifest class's json_schema_extra field.
    """
    try:
        # Get the JSON schema from the manifest class
        schema = block_spec.manifest_class.model_json_schema()
        
        # Extract block_type from json_schema_extra
        json_schema_extra = schema.get("json_schema_extra", {})
        block_type = json_schema_extra.get("block_type")
        
        return block_type
    except Exception:
        # If we can't determine the block type, return None
        # This allows the validation to continue with pattern matching
        return None
