import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from inference_exp.entities import ColorFormat
from inference_exp.models.auto_loaders.core import AutoModel
from inference_exp.models.base.classification import (
    ClassificationModel,
    ClassificationPrediction,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from inference_exp.models.base.depth_estimation import DepthEstimationModel
from inference_exp.models.base.documents_parsing import DocumentParsingModel
from inference_exp.models.base.embeddings import TextImageEmbeddingModel
from inference_exp.models.base.instance_segmentation import (
    InstanceDetections,
    InstanceSegmentationModel,
)
from inference_exp.models.base.keypoints_detection import (
    KeyPoints,
    KeyPointsDetectionModel,
)
from inference_exp.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
    OpenVocabularyObjectDetectionModel,
)
