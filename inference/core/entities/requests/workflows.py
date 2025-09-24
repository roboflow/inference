from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    DynamicBlockDefinition,
)


class WorkflowInferenceRequest(BaseModel):
    api_key: Optional[str] = Field(
        default=None,
        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
    )
    inputs: Dict[str, Any] = Field(
        description="Dictionary that contains each parameter defined as an input for chosen workflow"
    )
    excluded_fields: Optional[List[str]] = Field(
        default=None,
        description="List of field that shall be excluded from the response (among those defined in workflow specification)",
    )
    enable_profiling: bool = Field(
        default=False,
        description="Flag to request Workflow run profiling. Enables Workflow profiler only when server settings "
        "allow profiling traces to be exported to clients. Only applies for Workflows definitions saved "
        "on Roboflow platform.",
    )
    workflow_id: Optional[str] = Field(
        default=None, description="Optional identifier of workflow"
    )


class PredefinedWorkflowInferenceRequest(WorkflowInferenceRequest):
    use_cache: bool = Field(
        default=True,
        description="Controls usage of cache for workflow definitions. Set this to False when you frequently modify "
        "definition saved in Roboflow app and want to fetch the newest version for the request.",
    )


class WorkflowSpecificationInferenceRequest(WorkflowInferenceRequest):
    specification: dict
    is_preview: bool = Field(
        default=False,
        description="Reserved, used internally by Roboflow to distinguish between preview and non-preview runs",
    )


class DescribeBlocksRequest(BaseModel):
    dynamic_blocks_definitions: List[DynamicBlockDefinition] = Field(
        default_factory=list, description="Dynamic blocks to be used."
    )
    execution_engine_version: Optional[str] = Field(
        default=None,
        description="Requested Execution Engine compatibility. If given, result will only "
        "contain blocks suitable for requested EE version, otherwise - descriptions for "
        "all available blocks will be delivered.",
    )


class DescribeInterfaceRequest(BaseModel):
    api_key: str = Field(
        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
    )


class PredefinedWorkflowDescribeInterfaceRequest(DescribeInterfaceRequest):
    use_cache: bool = Field(
        default=True,
        description="Controls usage of cache for workflow definitions. Set this to False when you frequently modify "
        "definition saved in Roboflow app and want to fetch the newest version for the request. "
        "Only applies for Workflows definitions saved on Roboflow platform.",
    )


class WorkflowSpecificationDescribeInterfaceRequest(DescribeInterfaceRequest):
    specification: dict
