from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PREDICTION_TYPE_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ImageInputField,
    RoboflowModelField,
    StepOutputImageSelector,
    StepOutputSelector,
    StepSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class ExampleModelBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        protected_namespaces=(),
    )
    type: Literal["ExampleModel"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    model_id: Union[WorkflowParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        RoboflowModelField
    )
    string_value: Optional[str] = Field(default=None)

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]


class ExampleModelBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleModelBlockManifest

    def run(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleNonBatchFlowControlBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleNonBatchFlowControl"]
    next_steps: List[StepSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ExampleFlowControlBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleFlowControl"]
    a_steps: List[StepSelector]
    b_steps: List[StepSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ExampleFlowControlBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleFlowControlBlockManifest

    def run(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleTransformationBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleTransformation"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    predictions: StepOutputSelector(kind=[OBJECT_DETECTION_PREDICTION_KIND]) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]


class ExampleTransformationBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleTransformationBlockManifest

    def run(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleSinkBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleSink"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    predictions: StepOutputSelector(kind=[OBJECT_DETECTION_PREDICTION_KIND]) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="status", kind=[BOOLEAN_KIND]),
        ]


class ExampleSinkBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleSinkBlockManifest

    def run(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleFusionBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleFusion"]
    predictions: List[StepOutputSelector(kind=[OBJECT_DETECTION_PREDICTION_KIND])] = (
        Field(
            description="Reference to predictions of detection-like model, that can be based of cropping "
            "(detection must define RoI - eg: bounding box)",
            examples=[["$steps.my_object_detection_model.predictions"]],
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]


class ExampleFusionBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleFusionBlockManifest

    def run(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleBlockWithInitManifest(WorkflowBlockManifest):
    type: Literal["ExampleBlockWithInit"]
    predictions: List[StepOutputSelector(kind=[OBJECT_DETECTION_PREDICTION_KIND])] = (
        Field(
            description="Reference to predictions of detection-like model, that can be based of cropping "
            "(detection must define RoI - eg: bounding box)",
            examples=[["$steps.my_object_detection_model.predictions"]],
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]


class ExampleBlockWithInit(WorkflowBlock):

    def __init__(self, a: int, b: str):
        self.a = a
        self.b = b

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["a", "b"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleBlockWithInitManifest

    def run(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ExampleBlockWithFaultyInitManifest(WorkflowBlockManifest):
    type: Literal["ExampleBlockWithFaultyInit"]
    predictions: List[StepOutputSelector(kind=[OBJECT_DETECTION_PREDICTION_KIND])] = (
        Field(
            description="Reference to predictions of detection-like model, that can be based of cropping "
            "(detection must define RoI - eg: bounding box)",
            examples=[["$steps.my_object_detection_model.predictions"]],
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]


class ExampleBlockWithFaultyInit(WorkflowBlock):

    def __init__(self, a: int, b: str):
        self.a = a
        self.b = b

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["a"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExampleBlockWithFaultyInitManifest

    def run(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass
