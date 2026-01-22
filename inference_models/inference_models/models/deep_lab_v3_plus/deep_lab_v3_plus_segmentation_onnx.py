from threading import Lock
from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional

from inference_models import ColorFormat, SemanticSegmentationModel
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
)
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import run_onnx_session_with_batch_size_limit
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
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
        message=f"Could not import DeepLabV3Plus model with ONNX backend - this error means that some additional dependencies "
        f"are not installed in the environment. If you run the `inference-models` library directly in your Python "
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
    ) -> "DeepLabV3PlusForSemanticSegmentationOnnx":
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
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        try:
            background_class_id = [c.lower() for c in class_names].index("background")
        except ValueError:
            background_class_id = -1
        inference_config = parse_inference_config(
            config_path=model_package_content["inference_config.json"],
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
            },
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["weights.onnx"],
            providers=onnx_execution_providers,
        )
        input_batch_size = session.get_inputs()[0].shape[0]
        if isinstance(input_batch_size, str):
            input_batch_size = None
        input_name = session.get_inputs()[0].name
        return cls(
            session=session,
            input_name=input_name,
            class_names=class_names,
            inference_config=inference_config,
            background_class_id=background_class_id,
            device=device,
            input_batch_size=input_batch_size,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        inference_config: InferenceConfig,
        class_names: List[str],
        background_class_id: int,
        device: torch.device,
        input_batch_size: Optional[int],
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._class_names = class_names
        self._background_class_id = background_class_id
        self._device = device
        self._input_batch_size = input_batch_size
        self._session_thread_lock = Lock()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
        )

    def forward(
        self, pre_processed_images: PreprocessedInputs, **kwargs
    ) -> RawPrediction:
        with self._session_thread_lock:
            return run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs={self._input_name: pre_processed_images},
                min_batch_size=self._input_batch_size,
                max_batch_size=self._input_batch_size,
            )[0]

    def post_process(
        self,
        model_results: RawPrediction,
        pre_processing_meta: PreprocessedInputs,
        confidence_threshold: float = 0.5,
        **kwargs,
    ) -> List[SemanticSegmentationResult]:
        results = []
        for image_results, image_metadata in zip(model_results, pre_processing_meta):
            inference_size = image_metadata.inference_size
            mask_h_scale = model_results.shape[2] / inference_size.height
            mask_w_scale = model_results.shape[3] / inference_size.width
            mask_pad_top, mask_pad_bottom, mask_pad_left, mask_pad_right = (
                round(mask_h_scale * image_metadata.pad_top),
                round(mask_h_scale * image_metadata.pad_bottom),
                round(mask_w_scale * image_metadata.pad_left),
                round(mask_w_scale * image_metadata.pad_right),
            )
            _, mh, mw = image_results.shape
            if (
                mask_pad_top < 0
                or mask_pad_bottom < 0
                or mask_pad_left < 0
                or mask_pad_right < 0
            ):
                image_results = torch.nn.functional.pad(
                    image_results,
                    (
                        abs(min(mask_pad_left, 0)),
                        abs(min(mask_pad_right, 0)),
                        abs(min(mask_pad_top, 0)),
                        abs(min(mask_pad_bottom, 0)),
                    ),
                    "constant",
                    self._background_class_id,
                )
                padded_mask_offset_top = max(mask_pad_top, 0)
                padded_mask_offset_bottom = max(mask_pad_bottom, 0)
                padded_mask_offset_left = max(mask_pad_left, 0)
                padded_mask_offset_right = max(mask_pad_right, 0)
                image_results = image_results[
                    :,
                    padded_mask_offset_top : image_results.shape[1]
                    - padded_mask_offset_bottom,
                    padded_mask_offset_left : image_results.shape[1]
                    - padded_mask_offset_right,
                ]
            else:
                image_results = image_results[
                    :,
                    mask_pad_top : mh - mask_pad_bottom,
                    mask_pad_left : mw - mask_pad_right,
                ]
            if (
                image_results.shape[1]
                != image_metadata.size_after_pre_processing.height
                or image_results.shape[2]
                != image_metadata.size_after_pre_processing.width
            ):
                image_results = functional.resize(
                    image_results,
                    [
                        image_metadata.size_after_pre_processing.height,
                        image_metadata.size_after_pre_processing.width,
                    ],
                    interpolation=functional.InterpolationMode.BILINEAR,
                )
            image_results = torch.nn.functional.softmax(image_results, dim=0)
            image_confidence, image_class_ids = torch.max(image_results, dim=0)
            below_threshold = image_confidence < confidence_threshold
            image_confidence[below_threshold] = 0.0
            image_class_ids[below_threshold] = self._background_class_id
            if (
                image_metadata.static_crop_offset.offset_x > 0
                or image_metadata.static_crop_offset.offset_y > 0
            ):
                original_size_confidence_canvas = torch.zeros(
                    (
                        image_metadata.original_size.height,
                        image_metadata.original_size.width,
                    ),
                    device=self._device,
                    dtype=image_confidence.dtype,
                )
                original_size_confidence_canvas[
                    image_metadata.static_crop_offset.offset_y : image_metadata.static_crop_offset.offset_y
                    + image_confidence.shape[0],
                    image_metadata.static_crop_offset.offset_x : image_metadata.static_crop_offset.offset_x
                    + image_confidence.shape[1],
                ] = image_confidence
                original_size_confidence_class_id_canvas = (
                    torch.ones(
                        (
                            image_metadata.original_size.height,
                            image_metadata.original_size.width,
                        ),
                        device=self._device,
                        dtype=image_class_ids.dtype,
                    )
                    * self._background_class_id
                )
                original_size_confidence_class_id_canvas[
                    image_metadata.static_crop_offset.offset_y : image_metadata.static_crop_offset.offset_y
                    + image_class_ids.shape[0],
                    image_metadata.static_crop_offset.offset_x : image_metadata.static_crop_offset.offset_x
                    + image_class_ids.shape[1],
                ] = image_class_ids
                image_class_ids = original_size_confidence_class_id_canvas
                image_confidence = original_size_confidence_canvas
            results.append(
                SemanticSegmentationResult(
                    segmentation_map=image_class_ids,
                    confidence=image_confidence,
                )
            )
        return results
