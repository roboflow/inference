from pydantic import BaseModel, Field, ValidationError


class NotebookStartResponse(BaseModel):
    """Response model for notebook start request"""

    success: str = Field(..., description="Status of the request")
    message: str = Field(..., description="Message of the request", optional=True)
