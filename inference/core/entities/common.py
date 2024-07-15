from pydantic import Field

ModelID = Field(
    description="A unique model identifier",
    json_schema_extra={
        "example": "raccoon-detector-1",
    },
)
ModelType = Field(
    default=None,
    description="The type of the model, usually referring to what task the model performs",
    json_schema_extra={
        "example": "object-detection",
    },
)
ApiKey = Field(
    default=None,
    description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
)
