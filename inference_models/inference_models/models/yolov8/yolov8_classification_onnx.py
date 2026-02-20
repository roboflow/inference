from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import ClassificationModel, ClassificationPrediction
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import (
    CorruptedModelPackageError,
    EnvironmentConfigurationError,
    MissingDependencyError,
)
from inference_models.models.base.types import PreprocessedInputs
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import (
    run_onnx_session_with_batch_size_limit,
    set_onnx_execution_provider_defaults,
)
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLOv8 model with ONNX backend requires pycuda installation, which is brought with "
                "`onnx-*` extras of `inference-models` library. If you see this error running locally, "
                "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
                " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
                f"model, You can also contact Roboflow to get support."
                "Additionally - if AutoModel.from_pretrained(...) "
                f"automatically selects model package which does not match your environment - that's a serious problem and "
                f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLOv8ForClassificationOnnx(ClassificationModel[torch.Tensor, torch.Tensor]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "YOLOv8ForClassificationOnnx":
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
        onnx_execution_providers = set_onnx_execution_provider_defaults(
            providers=onnx_execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_onnx_trt_options=default_onnx_trt_options,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.onnx",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        inference_config = parse_inference_config(
            config_path=model_package_content["inference_config.json"],
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
            },
            implicit_resize_mode_substitutions={
                ResizeMode.FIT_LONGER_EDGE: (
                    ResizeMode.LETTERBOX,
                    127,
                    "YOLOv8 Classification model running with ONNX backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `letterbox` "
                    "resize mode with gray edges will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        if inference_config.post_processing.type != "softmax":
            raise CorruptedModelPackageError(
                message="Expected Softmax to be the post-processing",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["weights.onnx"],
            providers=onnx_execution_providers,
        )
        input_shape = session.get_inputs()[0].shape
        input_batch_size = input_shape[0]
        if isinstance(input_batch_size, str):
            input_batch_size = None
        input_name = session.get_inputs()[0].name
        return cls(
            session=session,
            input_name=input_name,
            inference_config=inference_config,
            class_names=class_names,
            device=device,
            input_batch_size=input_batch_size,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
        input_batch_size: Optional[int],
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device
        self._input_batch_size = input_batch_size
        self._session_thread_lock = Lock()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
        )[0]

    def forward(
        self, pre_processed_images: PreprocessedInputs, **kwargs
    ) -> torch.Tensor:
        with self._session_thread_lock:
            return run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs={self._input_name: pre_processed_images},
                min_batch_size=self._input_batch_size,
                max_batch_size=self._input_batch_size,
            )[0]

    def post_process(
        self,
        model_results: torch.Tensor,
        **kwargs,
    ) -> ClassificationPrediction:
        if self._inference_config.post_processing.fused:
            confidence = model_results
        else:
            confidence = torch.nn.functional.softmax(model_results, dim=-1)
        return ClassificationPrediction(
            class_id=confidence.argmax(dim=-1),
            confidence=confidence,
        )
