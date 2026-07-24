from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

from inference_models.models.base.classification import (
    ClassificationModel,
    MultiLabelClassificationModel,
)
from inference_models.models.base.depth_estimation import DepthEstimationModel
from inference_models.models.base.documents_parsing import (
    StructuredOCRModel,
    TextOnlyOCRModel,
)
from inference_models.models.base.embeddings import TextImageEmbeddingModel
from inference_models.models.base.instance_segmentation import InstanceSegmentationModel
from inference_models.models.base.keypoints_detection import KeyPointsDetectionModel
from inference_models.models.base.object_detection import (
    ObjectDetectionModel,
    OpenVocabularyObjectDetectionModel,
)
from inference_models.models.base.semantic_segmentation import SemanticSegmentationModel

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
    SemanticSegmentationModel,
    TextOnlyOCRModel,
]


@dataclass(frozen=True)
class InferenceModelConfig:
    model_architecture: Optional[ModelArchitecture]
    task_type: TaskType
    backend_type: Optional[BackendType]
    model_module: Optional[str]
    model_class: Optional[str]
    model_features: Optional[dict] = None
    trusted_source: Optional[bool] = None
    model_dependencies: Optional[List[dict]] = None
    recommended_parameters: Optional[dict] = None
    quantization: Optional[str] = None
    dynamic_batch_size_supported: Optional[bool] = None
    static_batch_size: Optional[int] = None
    runtime_compatibility_hash: Optional[str] = None
    offline_compatibility_hash: Optional[str] = None
    offline_manifest_version: Optional[int] = None

    def is_library_model(self) -> bool:
        return self.model_architecture is not None and self.backend_type is not None


@dataclass(frozen=True)
class PreProcessingOverrides:
    disable_contrast_enhancement: bool = field(default=False)
    disable_grayscale: bool = field(default=False)
    disable_static_crop: bool = field(default=False)
