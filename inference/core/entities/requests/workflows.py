from typing import Optional

from pydantic import BaseModel, Field
from roboflow_workflows.http_contract.entities import (  # noqa: F401
    DescribeBlocksRequest,
    DescribeInterfaceRequest,
    PredefinedWorkflowDescribeInterfaceRequest,
    PredefinedWorkflowInferenceRequest,
    WorkflowInferenceRequest,
    WorkflowSpecificationDescribeInterfaceRequest,
    WorkflowSpecificationInferenceRequest,
)


class DescribeWorkloadRequest(BaseModel):
    api_key: Optional[str] = Field(
        default=None,
        description="Roboflow API Key used to resolve the workflow definition and, when "
        "model metadata enrichment is enabled, to look up model metadata. "
        "May alternatively be sent in the `Authorization: Bearer <api_key>` header - the route "
        "still requires a key through one of the channels.",
    )


class PredefinedWorkflowDescribeWorkloadRequest(DescribeWorkloadRequest):
    use_cache: bool = Field(
        default=True,
        description="Controls usage of cache for workflow definitions. Set this to False when you frequently modify "
        "definition saved in Roboflow app and want to fetch the newest version for the request. "
        "Only applies for Workflows definitions saved on Roboflow platform.",
    )
    workflow_version_id: Optional[str] = Field(
        default=None,
        description="Specific version of the workflow to fetch. If not provided, the latest version is used.",
    )


class WorkflowSpecificationDescribeWorkloadRequest(DescribeWorkloadRequest):
    specification: dict
