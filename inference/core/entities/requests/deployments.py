from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from inference.enterprise.deployments.entities.deployment_specs import DeploymentSpecV1


class DeploymentsInferenceRequest(BaseModel):
    api_key: str = Field(
        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
    )
    runtime_parameters: Dict[str, Any] = Field(
        description="Dictionary that contains each parameter defined as an input for chosen deployment"
    )
    excluded_fields: Optional[List[str]] = Field(
        default=None,
        description="List of field that shall be excluded from the response (among those defined in deployment specs)",
    )


class DeploymentSpecificationInferenceRequest(DeploymentsInferenceRequest):
    specification: DeploymentSpecV1
