import os.path
from typing import List, Optional, Union, Tuple

import torch

from inference.v1.models.auto_loaders.auto_negotiation import negotiate_model_packages
from inference.v1.models.base.classification import (ClassificationModel, MultiLabelClassificationModel)
from inference.v1.models.base.instance_segmentation import InstanceSegmentationModel
from inference.v1.models.base.keypoints_detection import KeyPointsDetectionModel
from inference.v1.models.base.object_detection import ObjectDetectionModel
from inference.v1.models.base.depth_estimation import DepthEstimationModel
from inference.v1.models.base.documents_parsing import DocumentParsingModel
from inference.v1.models.base.embeddings import TextImageEmbeddingModel
from inference.v1.weights_providers.core import get_model_from_provider
from inference.v1.weights_providers.entities import BackendType, Quantization
from inference.v1.configuration import DEFAULT_DEVICE


AnyModel = Union[
    ClassificationModel,
    MultiLabelClassificationModel,
    DepthEstimationModel,
    DocumentParsingModel,
    TextImageEmbeddingModel,
    InstanceSegmentationModel,
    KeyPointsDetectionModel,
    ObjectDetectionModel,
]


class AutoModel:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        requested_model_package_id: Optional[str] = None,
        requested_backends: Optional[Union[str, BackendType, List[Union[str, BackendType]]]] = None,
        requested_batch_size: Optional[Union[int, Tuple[int, int]]] = None,
        requested_quantization: Optional[Union[str, Quantization, List[Union[str, Quantization]]]] = None,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        device: torch.device = DEFAULT_DEVICE,
        default_onnx_trt_options: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> AnyModel:
        if not os.path.isdir(model_name_or_path):
            model_metadata = get_model_from_provider(
                provider=weights_provider,
                model_id=model_name_or_path,
                api_key=api_key,
            )
            matching_model_packages = negotiate_model_packages(
                model_packages=model_metadata.model_packages,
                requested_model_package_id=requested_model_package_id,
                requested_backends=requested_backends,
                requested_batch_size=requested_batch_size,
                requested_quantization=requested_quantization,
            )
        else:
            pass
