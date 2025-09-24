from pydantic import Field

ModelID = Field(example="raccoon-detector-1", description="A unique model identifier")
ModelType = Field(
    default=None,
    examples=["object-detection"],
    description="The type of the model, usually referring to what task the model performs",
)
ApiKey = Field(
    default=None,
    description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
)
