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
    offline_manifest_version: Optional[int] = None
    model_id: Optional[str] = None
    # Provider-resolved identity that owns this cached package.  ``model_id``
    # in the on-disk manifest records the cache directory owner, which may
    # differ for explicitly discovered local packages.
    canonical_model_id: Optional[str] = None
    # Canonical identity of every package artefact. Hashed artefacts point at
    # the shared blob named after their MD5; provider artefacts without an MD5
    # are bound to the SHA-256 of the regular in-package file that was warmed.
    package_artifacts: Optional[List[dict]] = None
    # Dependency directory links are part of the package materialization too.
    # Binding them prevents a failed rewrite from silently repointing a parent
    # package at a different dependency package.
    dependency_package_paths: Optional[List[dict]] = None
    # SHA-256 of the complete raw manifest dictionary. This is intentionally
    # computed while reading the manifest so auto-resolution metadata can bind
    # every load-driving field, including custom model_module/model_class keys.
    manifest_content_hash: Optional[str] = None

    def is_library_model(self) -> bool:
        return self.model_architecture is not None and self.backend_type is not None


@dataclass(frozen=True)
class PreProcessingOverrides:
    disable_contrast_enhancement: bool = field(default=False)
    disable_grayscale: bool = field(default=False)
    disable_static_crop: bool = field(default=False)
