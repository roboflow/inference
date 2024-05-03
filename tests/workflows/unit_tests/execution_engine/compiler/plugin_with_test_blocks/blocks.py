from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field

from inference.enterprise.workflows.entities.base import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BATCH_OF_BOOLEAN_KIND,
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    PREDICTION_TYPE_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    FlowControl,
    StepOutputImageSelector,
    StepOutputSelector,
    StepSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class ExampleModelBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        protected_namespaces=(),
    )
    type: Literal["ExampleModel"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    model_id: Union[WorkflowParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        Field(
            description="Roboflow model identifier",
            examples=["my_project/3", "$inputs.model"],
        )
    )
    string_value: Optional[str] = Field(default=None)


class ExampleModelBlock(WorkflowBlock):
    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleModelBlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    async def run_locally(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleFlowControlBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleFlowControl"]
    steps_to_choose: List[StepSelector]


class ExampleFlowControlBlock(WorkflowBlock):
    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleFlowControlBlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    async def run_locally(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleTransformationBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleTransformation"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    predictions: StepOutputSelector(
        kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
    ) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
    )


class ExampleTransformationBlock(WorkflowBlock):
    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleTransformationBlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGES_KIND]),
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    async def run_locally(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleSinkBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleSink"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    predictions: StepOutputSelector(
        kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
    ) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
    )


class ExampleSinkBlock(WorkflowBlock):
    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleSinkBlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="status", kind=[BATCH_OF_BOOLEAN_KIND]),
        ]

    async def run_locally(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleFusionBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleFusion"]
    predictions: List[
        StepOutputSelector(kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND])
    ] = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=[["$steps.my_object_detection_model.predictions"]],
    )


class ExampleFusionBlock(WorkflowBlock):
    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleFusionBlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    async def run_locally(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleBlockWithInitManifest(WorkflowBlockManifest):
    type: Literal["ExampleBlockWithInit"]
    predictions: List[
        StepOutputSelector(kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND])
    ] = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=[["$steps.my_object_detection_model.predictions"]],
    )


class ExampleBlockWithInit(WorkflowBlock):

    def __init__(self, a: int, b: str):
        self.a = a
        self.b = b

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["a", "b"]

    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleBlockWithInitManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    async def run_locally(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleBlockWithFaultyInitManifest(WorkflowBlockManifest):
    type: Literal["ExampleBlockWithFaultyInit"]
    predictions: List[
        StepOutputSelector(kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND])
    ] = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=[["$steps.my_object_detection_model.predictions"]],
    )


class ExampleBlockWithFaultyInit(WorkflowBlock):

    def __init__(self, a: int, b: str):
        self.a = a
        self.b = b

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["a"]

    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleBlockWithFaultyInitManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    async def run_locally(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass
