from threading import Lock
from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional

from inference_models import ColorFormat, SemanticSegmentationModel
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import PreprocessedInputs, PreprocessingMetadata
from inference_models.models.common.cuda import (
    use_cuda_context,
    use_primary_cuda_context,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    TRTConfig,
    parse_class_names_file,
    parse_inference_config,
    parse_trt_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.common.trt import (
    get_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_model,
)

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import YOLOv8 model with TRT backend - this error means that some additional dependencies "
        f"are not installed in the environment. If you run the `inference-models` library directly in your Python "
        f"program, make sure the following extras of the package are installed: `trt10` - installation can only "
        f"succeed for Linux and Windows machines with Cuda 12 installed. Jetson devices, should have TRT 10.x "
        f"installed for all builds with Jetpack 6. "
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="TODO", help_url="https://todo"
    ) from import_error


class DeepLabV3PlusForSemanticSegmentationTRT(
    SemanticSegmentationModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        **kwargs,
    ) -> "DeepLabV3PlusForSemanticSegmentationTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=f"TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://todo",
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "trt_config.json",
                "engine.plan",
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
        trt_config = parse_trt_config(
            config_path=model_package_content["trt_config.json"]
        )
        cuda.init()
        cuda_device = cuda.Device(device.index or 0)
        with use_primary_cuda_context(cuda_device=cuda_device) as cuda_context:
            engine = load_model(
                model_path=model_package_content["engine.plan"],
                engine_host_code_allowed=engine_host_code_allowed,
            )
            execution_context = engine.create_execution_context()
        inputs, outputs = get_engine_inputs_and_outputs(engine=engine)
        if len(inputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model input, found: {len(inputs)}.",
                help_url="https://todo",
            )
        if len(outputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model output, found: {len(outputs)}.",
                help_url="https://todo",
            )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_name=outputs[0],
            class_names=class_names,
            background_class_id=background_class_id,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_name: str,
        output_name: str,
        class_names: List[str],
        background_class_id: int,
        inference_config: InferenceConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = [output_name]
        self._class_names = class_names
        self._background_class_id = background_class_id
        self._inference_config = inference_config
        self._trt_config = trt_config
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._lock = Lock()

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
    ) -> torch.Tensor:
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                return infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                )[0]

    def post_process(
        self,
        model_results: torch.Tensor,
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
