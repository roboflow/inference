from typing import Any, Dict

from pydantic import BaseModel, Field


class DeploymentsInferenceResponse(BaseModel):
    deployment_outputs: Dict[str, Any] = Field(
        description="Dictionary with keys defined in deployment output and serialised values"
    )
