from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowInferenceRequest(BaseModel):
    api_key: str = Field(
        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
    )
    inputs: Dict[str, Any] = Field(
        description="Dictionary that contains each parameter defined as an input for chosen workflow"
    )
    excluded_fields: Optional[List[str]] = Field(
        default=None,
        description="List of field that shall be excluded from the response (among those defined in workflow specification)",
    )


class WorkflowSpecificationInferenceRequest(WorkflowInferenceRequest):
    specification: dict
