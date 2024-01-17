from typing import Any, Dict

from pydantic import BaseModel, Field


class DeploymentsInferenceRequest(BaseModel):
    api_key: str = Field(
        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
    )
    runtime_parameters: Dict[str, Any] = Field(
        description="Dictionary that contains each parameter defined as an input for chosen deployment"
    )
