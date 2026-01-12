from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from inference_models.models.base.classification import (
    ClassificationModel,
    MultiLabelClassificationModel,
)
from inference_models.models.base.depth_estimation import DepthEstimationModel
from inference_models.models.base.documents_parsing import StructuredOCRModel
from inference_models.models.base.embeddings import TextImageEmbeddingModel
from inference_models.models.base.instance_segmentation import InstanceSegmentationModel
from inference_models.models.base.keypoints_detection import KeyPointsDetectionModel
from inference_models.models.base.object_detection import (
    ObjectDetectionModel,
    OpenVocabularyObjectDetectionModel,
)

ModelArchitecture = str
TaskType = Optional[str]
MODEL_CONFIG_FILE_NAME = "model_config.json"


class BackendType(str, Enum):
    TORCH = "torch"
    TORCH_SCRIPT = "torch-script"
    ONNX = "onnx"
    TRT = "trt"
    HF = "hugging-face"
    ULTRALYTICS = "ultralytics"
    MEDIAPIPE = "mediapipe"
    CUSTOM = "custom"


AnyModel = Union[
    ClassificationModel,
    MultiLabelClassificationModel,
    DepthEstimationModel,
    StructuredOCRModel,
    TextImageEmbeddingModel,
    InstanceSegmentationModel,
    KeyPointsDetectionModel,
    ObjectDetectionModel,
    OpenVocabularyObjectDetectionModel,
]


@dataclass(frozen=True)
class InferenceModelConfig:
    model_architecture: Optional[ModelArchitecture]
    task_type: TaskType
    backend_type: Optional[BackendType]
    model_module: Optional[str]
    model_class: Optional[str]

    def is_library_model(self) -> bool:
        return self.model_architecture is not None and self.backend_type is not None
