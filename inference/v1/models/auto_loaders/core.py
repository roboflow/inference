import os.path
from typing import List, Optional, Tuple, Union

import torch

from inference.v1.configuration import DEFAULT_DEVICE
from inference.v1.errors import ModelLoadingError
from inference.v1.logger import logger
from inference.v1.models.auto_loaders.auto_negotiation import negotiate_model_packages
from inference.v1.models.auto_loaders.models_registry import ModelArchitecture, TaskType
from inference.v1.models.base.classification import (
    ClassificationModel,
    MultiLabelClassificationModel,
)
from inference.v1.models.base.depth_estimation import DepthEstimationModel
from inference.v1.models.base.documents_parsing import DocumentParsingModel
from inference.v1.models.base.embeddings import TextImageEmbeddingModel
from inference.v1.models.base.instance_segmentation import InstanceSegmentationModel
from inference.v1.models.base.keypoints_detection import KeyPointsDetectionModel
from inference.v1.models.base.object_detection import ObjectDetectionModel
from inference.v1.weights_providers.core import get_model_from_provider
from inference.v1.weights_providers.entities import (
    BackendType,
    ModelPackageMetadata,
    Quantization,
)

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
        requested_backends: Optional[
            Union[str, BackendType, List[Union[str, BackendType]]]
        ] = None,
        requested_batch_size: Optional[Union[int, Tuple[int, int]]] = None,
        requested_quantization: Optional[
            Union[str, Quantization, List[Union[str, Quantization]]]
        ] = None,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        device: torch.device = DEFAULT_DEVICE,
        default_onnx_trt_options: bool = True,
        max_package_loading_attempts: Optional[int] = None,
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


def attempt_loading_matching_model_packages(
    model_id: str,
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
    matching_model_packages: List[ModelPackageMetadata],
    max_package_loading_attempts: Optional[int] = None,
) -> AnyModel:
    if max_package_loading_attempts is not None:
        matching_model_packages = matching_model_packages[:max_package_loading_attempts]
    if not matching_model_packages:
        raise ModelLoadingError(
            f"Cannot load model {model_id} - no matching model package candidates for given model "
            f"running in this environment."
        )
    failed_load_attempts: List[Tuple[str, Exception]] = []
    for model_package in matching_model_packages:
        try:
            return initialize_model(
                model_id=model_id,
                model_architecture=model_architecture,
                task_type=task_type,
                model_package=model_package,
            )
        except Exception as error:
            logger.warning(
                f"Model package with id {model_package.package_id} that was selected to be loaded "
                f"failed to load with error: {error} of type {error.__class__.__name__}. This may "
                f"be caused several issues. If you see this warning after manually specifying model "
                f"package to be loaded - make sure that all required dependencies are installed. If "
                f"that warning is displayed when the model package was auto-selected - there is most "
                f"likely a bug in `inference` and you should raise an issue providing full context of "
                f"the event. https://github.com/roboflow/inference/issues"
            )
            failed_load_attempts.append((model_package.package_id, error))
    summary_of_errors = "\n".join(
        f"\t* model_package_id={model_package_id} error={error} error_type={error.__class__.__name__}"
        for model_package_id, error in failed_load_attempts
    )
    raise ModelLoadingError(
        f"Could not load any of model package candidate for model {model_id}. This may "
        f"be caused several issues. If you see this warning after manually specifying model "
        f"package to be loaded - make sure that all required dependencies are installed. If "
        f"that warning is displayed when the model package was auto-selected - there is most "
        f"likely a bug in `inference` and you should raise an issue providing full context of "
        f"the event. https://github.com/roboflow/inference/issues\n\n"
        f"Here is the summary of errors for specific model packages:\n{summary_of_errors}"
    )


def initialize_model(
    model_id: str,
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
    model_package: ModelPackageMetadata,
) -> AnyModel:
    pass
