from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt
from typing_extensions import Annotated


class DisabledActiveLearningConfiguration(BaseModel):
    enabled: Literal[False]


class LimitDefinition(BaseModel):
    type: Literal["minutely", "hourly", "daily"]
    value: PositiveInt


class RandomSamplingConfig(BaseModel):
    type: Literal["random"]
    name: str
    traffic_percentage: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    tags: List[str] = Field(default_factory=list)
    limits: List[LimitDefinition] = Field(default_factory=list)


class CloseToThresholdSampling(BaseModel):
    type: Literal["close_to_threshold"]
    name: str
    probability: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    threshold: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    epsilon: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    max_batch_images: Optional[int] = Field(default=None)
    only_top_classes: bool = Field(default=True)
    minimum_objects_close_to_threshold: int = Field(default=1)
    selected_class_names: Optional[List[str]] = Field(default=None)
    tags: List[str] = Field(default_factory=list)
    limits: List[LimitDefinition] = Field(default_factory=list)


class ClassesBasedSampling(BaseModel):
    type: Literal["classes_based"]
    name: str
    probability: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    selected_class_names: List[str]
    tags: List[str] = Field(default_factory=list)
    limits: List[LimitDefinition] = Field(default_factory=list)


class DetectionsBasedSampling(BaseModel):
    type: Literal["detections_number_based"]
    name: str
    probability: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    more_than: Optional[NonNegativeInt]
    less_than: Optional[NonNegativeInt]
    selected_class_names: Optional[List[str]] = Field(default=None)
    tags: List[str] = Field(default_factory=list)
    limits: List[LimitDefinition] = Field(default_factory=list)


class ActiveLearningBatchingStrategy(BaseModel):
    batches_name_prefix: str
    recreation_interval: Literal["never", "daily", "weekly", "monthly"]
    max_batch_images: Optional[int] = Field(default=None)


ActiveLearningStrategyType = Annotated[
    Union[
        RandomSamplingConfig,
        CloseToThresholdSampling,
        ClassesBasedSampling,
        DetectionsBasedSampling,
    ],
    Field(discriminator="type"),
]


class EnabledActiveLearningConfiguration(BaseModel):
    enabled: bool
    persist_predictions: bool
    sampling_strategies: List[ActiveLearningStrategyType]
    batching_strategy: ActiveLearningBatchingStrategy
    tags: List[str] = Field(default_factory=list)
    max_image_size: Optional[Tuple[PositiveInt, PositiveInt]] = Field(default=None)
    jpeg_compression_level: int = Field(default=95, gt=0, le=100)
