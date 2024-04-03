from typing import Any, Dict

from pydantic import BaseModel, Field


class WorkflowInferenceResponse(BaseModel):
    outputs: Dict[str, Any] = Field(
        description="Dictionary with keys defined in workflow output and serialised values"
    )


class WorkflowValidationStatus(BaseModel):
    status: str = Field(description="Represents validation status")
