from typing import TYPE_CHECKING, List, Optional

from inference.core.workflows.errors import (
    DuplicatedNameError,
    EnterpriseFeatureNotAvailableError,
)
from inference.core.workflows.execution_engine.entities.base import InputType, JsonField
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    get_input_parameters_selectors,
    get_output_selectors,
    get_steps_selectors,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest

if TYPE_CHECKING:
    from inference.core.workflows.execution_engine.v1.compiler.entities import (
        BlockSpecification,
    )


@execution_phase(
    name="workflow_definition_validation",
    categories=["execution_engine_operation"],
)
def validate_workflow_specification(
    workflow_definition: ParsedWorkflowDefinition,
    available_blocks: Optional[List["BlockSpecification"]] = None,
    api_key: Optional[str] = None,
    profiler: Optional[WorkflowsProfiler] = None,
) -> None:
    validate_inputs_names_are_unique(inputs=workflow_definition.inputs)
    validate_steps_names_are_unique(steps=workflow_definition.steps)
    validate_outputs_names_are_unique(outputs=workflow_definition.outputs)
    if available_blocks is not None:
        validate_enterprise_blocks_access(
            steps=workflow_definition.steps,
            available_blocks=available_blocks,
            api_key=api_key,
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


def validate_enterprise_blocks_access(
    steps: List[WorkflowBlockManifest],
    available_blocks: List["BlockSpecification"],
    api_key: Optional[str] = None,
) -> None:
    """
    Validate that enterprise-only blocks are only used by accounts with enterprise access.

    Args:
        steps: List of workflow steps to validate
        available_blocks: List of all available block specifications
        api_key: API key for plan validation (optional)

    Raises:
        EnterpriseFeatureNotAvailableError: If a non-enterprise account attempts
            to use an enterprise-only block
    """
    # Create a mapping of block type -> BlockSpecification for fast lookup
    blocks_by_type = {}
    for block_spec in available_blocks:
        # Get the manifest type identifier
        block_schema = block_spec.manifest_class.model_json_schema()
        type_property = block_schema.get("properties", {}).get("type", {})
        block_type = type_property.get("const") or (
            type_property.get("enum", [None])[0]
            if type_property.get("enum")
            else None
        )
        if block_type:
            blocks_by_type[block_type] = block_spec

    # Check each step for enterprise-only blocks
    for step in steps:
        step_type = step.type

        # Find the corresponding block specification
        block_spec = blocks_by_type.get(step_type)
        if not block_spec:
            # Block not found - will be caught by other validation
            continue

        # Check if this block is enterprise-only
        is_enterprise_only = _is_block_enterprise_only(block_spec)

        if not is_enterprise_only:
            # Not an enterprise block, no validation needed
            continue

        # This is an enterprise block - validate access
        if not _has_enterprise_access(api_key):
            block_name = _get_block_name(block_spec)
            raise EnterpriseFeatureNotAvailableError(
                public_message=f"The workflow step '{step.name}' uses '{block_name}' which is an "
                f"enterprise-only feature. This block requires an enterprise subscription. "
                f"Please upgrade your account or remove this block from your workflow.",
                context="workflow_compilation | enterprise_access_validation",
            )


def _is_block_enterprise_only(block_spec: "BlockSpecification") -> bool:
    """Check if a block is marked as enterprise-only."""
    try:
        # Get the model_config from the manifest class
        config = getattr(block_spec.manifest_class, "model_config", {})
        if isinstance(config, dict):
            json_schema_extra = config.get("json_schema_extra", {})
        else:
            # ConfigDict object - access as attribute
            json_schema_extra = getattr(config, "json_schema_extra", {})

        ui_manifest = json_schema_extra.get("ui_manifest", {})
        return ui_manifest.get("enterprise_only", False)
    except (AttributeError, KeyError, TypeError):
        # If we can't determine, assume it's not enterprise-only
        return False


def _get_block_name(block_spec: "BlockSpecification") -> str:
    """Get the human-friendly name of a block."""
    try:
        config = getattr(block_spec.manifest_class, "model_config", {})
        if isinstance(config, dict):
            json_schema_extra = config.get("json_schema_extra", {})
        else:
            # ConfigDict object - access as attribute
            json_schema_extra = getattr(config, "json_schema_extra", {})
        return json_schema_extra.get("name", block_spec.identifier)
    except (AttributeError, KeyError, TypeError):
        return block_spec.identifier


def _has_enterprise_access(api_key: Optional[str]) -> bool:
    """
    Check if the provided API key has enterprise access.

    Args:
        api_key: The API key to validate

    Returns:
        True if the account has enterprise access, False otherwise
    """
    from inference.core.env import API_BASE_URL
    from inference.core.logger import logger
    from inference.usage_tracking.plan_details import PlanDetails

    # If no API key provided, deny access
    if not api_key:
        return False

    try:
        # Initialize PlanDetails to check subscription tier
        plan_details = PlanDetails(
            api_plan_endpoint_url=f"{API_BASE_URL}/plan",
            sqlite_cache_enabled=False,  # Don't cache during compilation
        )

        # Get the plan for this API key
        plan = plan_details.get_api_key_plan(api_key=api_key)

        # Check if the account has enterprise access
        is_enterprise = plan.get("is_enterprise", False)
        return is_enterprise

    except Exception as e:
        # If we can't validate (network issues, etc.), fail open to avoid breaking workflows
        # Log the error but allow access - this matches the fail-open behavior requirement
        logger.warning(
            f"Could not validate enterprise access for API key: {e}. "
            f"Allowing access (fail-open behavior)."
        )
        return True  # Fail open
