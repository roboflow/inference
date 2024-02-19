from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.inputs import (
    InferenceImage,
    InferenceParameter,
)
from inference.enterprise.workflows.entities.outputs import JsonField
from inference.enterprise.workflows.entities.steps import (
    AbsoluteStaticCrop,
    ActiveLearningDataCollector,
    ClassificationModel,
    ClipComparison,
    Condition,
    Crop,
    DetectionFilter,
    DetectionOffset,
    DetectionsConsensus,
    InstanceSegmentationModel,
    KeypointsDetectionModel,
    MultiLabelClassificationModel,
    ObjectDetectionModel,
    OCRModel,
    RelativeStaticCrop,
    YoloWorld,
)

InputType = Annotated[
    Union[InferenceImage, InferenceParameter], Field(discriminator="type")
]
StepType = Annotated[
    Union[
        ClassificationModel,
        MultiLabelClassificationModel,
        ObjectDetectionModel,
        KeypointsDetectionModel,
        InstanceSegmentationModel,
        OCRModel,
        Crop,
        Condition,
        DetectionFilter,
        DetectionOffset,
        ClipComparison,
        RelativeStaticCrop,
        AbsoluteStaticCrop,
        DetectionsConsensus,
        ActiveLearningDataCollector,
        YoloWorld,
    ],
    Field(discriminator="type"),
]


class WorkflowSpecificationV1(BaseModel):
    version: Literal["1.0"]
    inputs: List[InputType]
    steps: List[StepType]
    outputs: List[JsonField]


class WorkflowSpecification(BaseModel):
    specification: (
        WorkflowSpecificationV1  # in the future - union with discriminator can be used
    )
