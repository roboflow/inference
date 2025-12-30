import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from inference_models.entities import ColorFormat
from inference_models.model_pipelines.auto_loaders.core import AutoModelPipeline
from inference_models.models.auto_loaders.core import AutoModel
from inference_models.models.base.classification import (
    ClassificationModel,
    ClassificationPrediction,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.depth_estimation import DepthEstimationModel
from inference_models.models.base.documents_parsing import (
    StructuredOCRModel,
    TextOnlyOCRModel,
)
from inference_models.models.base.embeddings import TextImageEmbeddingModel
from inference_models.models.base.instance_segmentation import (
    InstanceDetections,
    InstanceSegmentationModel,
)
from inference_models.models.base.keypoints_detection import (
    KeyPoints,
    KeyPointsDetectionModel,
)
from inference_models.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
    OpenVocabularyObjectDetectionModel,
)
from inference_models.models.base.semantic_segmentation import SemanticSegmentationModel
