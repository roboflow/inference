from typing import List, Optional, Tuple, Union

import torch
from inference_exp import SemanticSegmentationModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.errors import EnvironmentConfigurationError, MissingDependencyError
from inference_exp.models.base.semantic_segmentation import SemanticSegmentationResult
from inference_exp.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_exp.utils.onnx_introspection import get_selected_onnx_execution_providers

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import DeepLabV3Plus model with ONNX backend - this error means that some additional dependencies "
        f"are not installed in the environment. If you run the `inference-exp` library directly in your Python "
        f"program, make sure the following extras of the package are installed: \n"
        f"\t* `onnx-cpu` - when you wish to use library with CPU support only\n"
        f"\t* `onnx-cu12` - for running on GPU with Cuda 12 installed\n"
        f"\t* `onnx-cu118` - for running on GPU with Cuda 11.8 installed\n"
        f"\t* `onnx-jp6-cu126` - for running on Jetson with Jetpack 6\n"
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error


class DeepLabV3PlusForSemanticSegmentationOnnx(
    SemanticSegmentationModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "DeepKabV3PlusForSemanticSegmentationOnnx":
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message=f"Could not initialize model - selected backend is ONNX which requires execution provider to "
                f"be specified - explicitly in `from_pretrained(...)` method or via env variable "
                f"`ONNXRUNTIME_EXECUTION_PROVIDERS`. If you run model locally - adjust your setup, otherwise "
                f"contact the platform support.",
                help_url="https://todo",
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.onnx",
            ],
        )

    @property
    def class_names(self) -> List[str]:
        return []

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
        pass

    def forward(
        self, pre_processed_images: PreprocessedInputs, **kwargs
    ) -> RawPrediction:
        pass

    def post_process(
        self,
        model_results: RawPrediction,
        pre_processing_meta: PreprocessedInputs,
        **kwargs,
    ) -> List[SemanticSegmentationResult]:
        pass
