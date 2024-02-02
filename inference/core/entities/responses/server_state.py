from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.core.managers.entities import ModelDescription


class ServerVersionInfo(BaseModel):
    """Server version information.

    Attributes:
        name (str): Server name.
        version (str): Server version.
        uuid (str): Server UUID.
    """

    name: str = Field(examples=["Roboflow Inference Server"])
    version: str = Field(examples=["0.0.1"])
    uuid: str = Field(examples=["9c18c6f4-2266-41fb-8a0f-c12ae28f6fbe"])


class ModelDescriptionEntity(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(
        description="Identifier of the model", examples=["some-project/3"]
    )
    task_type: str = Field(
        description="Type of the task that the model performs",
        examples=["classification"],
    )
    batch_size: Optional[Union[int, str]] = Field(
        None,
        description="Batch size accepted by the model (if registered).",
    )
    input_height: Optional[int] = Field(
        None,
        description="Image input height accepted by the model (if registered).",
    )
    input_width: Optional[int] = Field(
        None,
        description="Image input width accepted by the model (if registered).",
    )

    @classmethod
    def from_model_description(
        cls, model_description: ModelDescription
    ) -> "ModelDescriptionEntity":
        return cls(
            model_id=model_description.model_id,
            task_type=model_description.task_type,
            batch_size=model_description.batch_size,
            input_height=model_description.input_height,
            input_width=model_description.input_width,
        )


class ModelsDescriptions(BaseModel):
    models: List[ModelDescriptionEntity] = Field(
        description="List of models that are loaded by model manager.",
    )

    @classmethod
    def from_models_descriptions(
        cls, models_descriptions: List[ModelDescription]
    ) -> "ModelsDescriptions":
        return cls(
            models=[
                ModelDescriptionEntity.from_model_description(
                    model_description=model_description
                )
                for model_description in models_descriptions
            ]
        )
