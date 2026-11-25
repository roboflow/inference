"""Frozen dataset semantics supplied by Core for oriented-box models."""

from pydantic import BaseModel, Field, StrictBool


class OrientedBoxMetadata(BaseModel):
    """Preserve class front policy independently of a model's heading head."""

    classHeading: dict[str, StrictBool] = Field(
        description="Frozen generated-version class heading policy."
    )
    classHeadingRevision: int = Field(
        strict=True, ge=0, description="Frozen generated-version policy revision."
    )
