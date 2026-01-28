from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import InstanceDetections, InstanceSegmentationModel
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
)
from inference_models.entities import ColorFormat
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
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
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    crop_masks_to_boxes,
    post_process_nms_fused_model_output,
    preprocess_segmentation_masks,
    run_nms_for_instance_segmentation,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.common.trt import (
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
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
        message="TODO",
        help_url="https://todo",
    ) from import_error


class YOLOv8ForInstanceSegmentationTRT(
    InstanceSegmentationModel[
        torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
    ]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        **kwargs,
    ) -> "YOLOv8ForInstanceSegmentationTRT":
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
        inference_config = parse_inference_config(
            config_path=model_package_content["inference_config.json"],
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
            },
        )
        if inference_config.post_processing.type != "nms":
            raise CorruptedModelPackageError(
                message="Expected NMS to be the post-processing",
                help_url="https://todo",
            )
        trt_config = parse_trt_config(
            config_path=model_package_content["trt_config.json"]
        )
        cuda.init()
        cuda_device = cuda.Device(device.index or 0)
        with use_primary_cuda_context(cuda_device=cuda_device) as cuda_context:
            engine = load_trt_model(
                model_path=model_package_content["engine.plan"],
                engine_host_code_allowed=engine_host_code_allowed,
            )
            execution_context = engine.create_execution_context()
        inputs, outputs = get_trt_engine_inputs_and_outputs(engine=engine)
        if len(inputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model input, found: {len(inputs)}.",
                help_url="https://todo",
            )
        if len(outputs) != 2:
            raise CorruptedModelPackageError(
                message=f"Implementation assume 2 model outputs, found: {len(outputs)}.",
                help_url="https://todo",
            )
        if "output0" not in outputs or "output1" not in outputs:
            raise CorruptedModelPackageError(
                message=f"Expected model outputs to be named `output0` and `output1`, but found: {outputs}.",
                help_url="https://todo",
            )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=["output0", "output1"],
            class_names=class_names,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            execution_context=execution_context,
            cuda_context=cuda_context,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_name: str,
        output_names: List[str],
        class_names: List[str],
        inference_config: InferenceConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = output_names
        self._class_names = class_names
        self._inference_config = inference_config
        self._trt_config = trt_config
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._session_thread_lock = Lock()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._session_thread_lock:
            with use_cuda_context(context=self._cuda_context):
                instances, protos = infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                )
                return instances, protos

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
        iou_threshold: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
        **kwargs,
    ) -> List[InstanceDetections]:
        instances, protos = model_results
        if self._inference_config.post_processing.fused:
            nms_results = post_process_nms_fused_model_output(
                output=instances, conf_thresh=confidence
            )
        else:
            nms_results = run_nms_for_instance_segmentation(
                output=instances,
                conf_thresh=confidence,
                iou_thresh=iou_threshold,
                max_detections=max_detections,
                class_agnostic=class_agnostic_nms,
            )
        final_results = []
        for image_bboxes, image_protos, image_meta in zip(
            nms_results, protos, pre_processing_meta
        ):
            pre_processed_masks = preprocess_segmentation_masks(
                protos=image_protos,
                masks_in=image_bboxes[:, 6:],
            )
            cropped_masks = crop_masks_to_boxes(
                image_bboxes[:, :4], pre_processed_masks
            )
            padding = (
                image_meta.pad_left,
                image_meta.pad_top,
                image_meta.pad_right,
                image_meta.pad_bottom,
            )
            aligned_boxes, aligned_masks = align_instance_segmentation_results(
                image_bboxes=image_bboxes,
                masks=cropped_masks,
                padding=padding,
                scale_height=image_meta.scale_height,
                scale_width=image_meta.scale_width,
                original_size=image_meta.original_size,
                size_after_pre_processing=image_meta.size_after_pre_processing,
                inference_size=image_meta.inference_size,
                static_crop_offset=image_meta.static_crop_offset,
            )
            final_results.append(
                InstanceDetections(
                    xyxy=aligned_boxes[:, :4].round().int(),
                    class_id=aligned_boxes[:, 5].int(),
                    confidence=aligned_boxes[:, 4],
                    mask=aligned_masks,
                )
            )
        return final_results
