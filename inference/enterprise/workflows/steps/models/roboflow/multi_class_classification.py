from typing import Literal, Union, Optional, Type, List, Dict, Any, Tuple

from pydantic import BaseModel, ConfigDict, Field

from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import InferenceImageSelector, OutputStepImageSelector, \
    InferenceParameterSelector, ROBOFLOW_MODEL_ID_KIND, BOOLEAN_KIND, ROBOFLOW_PROJECT_KIND, FloatZeroToOne, \
    FLOAT_ZERO_TO_ONE_KIND, CLASSIFICATION_PREDICTION_KIND, STRING_KIND, PARENT_ID_KIND, PREDICTION_TYPE_KIND, \
    ROBOFLOW_API_KEY_KIND, FlowControl


class BlockManifest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "This block represents inference from Roboflow multi-class classification model.",
            "docs": "https://inference.roboflow.com/workflows/classify_objects",
            "block_type": "model",
        }
    )
    type: Literal["RoboflowClassificationModel"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    model_id: Union[InferenceParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        Field(
            description="Roboflow model identifier",
            examples=["my_project/3", "$inputs.model"],
        )
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    disable_active_learning: Union[
        Optional[bool], InferenceParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Parameter to decide if Active Learning data sampling is disabled for the model",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        InferenceParameterSelector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for Active Learning data sampling - see Roboflow Active Learning "
        "docs for more information",
        examples=["my_project", "$inputs.al_target_project"],
    )
    roboflow_api_key: Union[InferenceParameterSelector(kind=[ROBOFLOW_API_KEY_KIND]), Optional[str]] = Field(
        default=None,
        description="API key for Roboflow platform. When using with inference server or hosted inference API - leave it `None`",
        examples=["my-api-key", "$inputs.roboflow_api_key"],
    )


class RoboflowClassificationBlock:

    @classmethod
    def get_input_manifest(cls) -> Type[BaseModel]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name="top", kind=[STRING_KIND]),
            OutputDefinition(name="confidence", kind=[FLOAT_ZERO_TO_ONE_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    async def run_locally(self) -> Union[Dict[str, Any], Tuple[Dict[str, Any], FlowControl]]:
        pass

    async def run_remotely(self) -> Union[Dict[str, Any], Tuple[Dict[str, Any], FlowControl]]:
        pass
