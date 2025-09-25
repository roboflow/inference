import threading
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import Detections, ObjectDetectionModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import EnvironmentConfigurationError, MissingDependencyError
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.onnx import (
    run_session_with_batch_size_limit,
    set_execution_provider_defaults,
)
from inference_exp.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_exp.models.common.roboflow.post_processing import (
    rescale_image_detections,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.models.rfdetr.class_remapping import prepare_class_remapping
from inference_exp.utils.onnx_introspection import get_selected_onnx_execution_providers

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import YOLOv8 model with ONNX backend - this error means that some additional dependencies "
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


class RFDetrForObjectDetectionONNX(
    (
        ObjectDetectionModel[
            torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
        ]
    )
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "RFDetrForObjectDetectionONNX":
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
        onnx_execution_providers = set_execution_provider_defaults(
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
        )
        class_id_remapping = None
        if inference_config.class_names_operations:
            class_names, class_id_remapping = prepare_class_remapping(
                class_names=class_names,
                class_names_operations=inference_config.class_names_operations,
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
            class_id_remapping=class_id_remapping,
            inference_config=inference_config,
            device=device,
            input_batch_size=input_batch_size,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        class_names: List[str],
        class_id_remapping: Optional[Dict[int, int]],
        inference_config: InferenceConfig,
        device: torch.device,
        input_batch_size: Optional[int],
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._class_names = class_names
        self._class_id_remapping = class_id_remapping
        self._device = device
        self._min_batch_size = input_batch_size
        self._max_batch_size = (
            input_batch_size
            if input_batch_size is not None
            else inference_config.forward_pass.max_dynamic_batch_size
        )
        self._session_thread_lock = threading.Lock()

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
            bboxes, logits = run_session_with_batch_size_limit(
                session=self._session,
                inputs={self._input_name: pre_processed_images},
                min_batch_size=self._min_batch_size,
                max_batch_size=self._max_batch_size,
            )
            return bboxes, logits

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        threshold: float = 0.5,
        **kwargs,
    ) -> List[Detections]:
        bboxes, logits = model_results
        logits_sigmoid = torch.nn.functional.sigmoid(logits)
        results = []
        for image_bboxes, image_logits, image_meta in zip(
            bboxes, logits_sigmoid, pre_processing_meta
        ):
            confidence, top_classes = image_logits.max(dim=1)
            confidence_mask = confidence > threshold
            confidence = confidence[confidence_mask]
            top_classes = top_classes[confidence_mask]
            selected_boxes = image_bboxes[confidence_mask]
            confidence, sorted_indices = torch.sort(confidence, descending=True)
            top_classes = top_classes[sorted_indices]
            selected_boxes = selected_boxes[sorted_indices]
            if self._class_id_remapping is not None:
                remapping_mask = [
                    label not in self._class_id_remapping for label in top_classes
                ]
                top_classes = top_classes[remapping_mask]
                top_classes = torch.tensor(
                    [
                        self._class_id_remapping[label]
                        for label in top_classes[remapping_mask].tolist()
                    ],
                    device=top_classes.device,
                )
                selected_boxes = selected_boxes[remapping_mask]
                confidence = confidence[remapping_mask]
            cxcy = selected_boxes[:, :2]
            wh = selected_boxes[:, 2:]
            xy_min = cxcy - 0.5 * wh
            xy_max = cxcy + 0.5 * wh
            selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
            inference_size_hwhw = torch.tensor(
                [
                    image_meta.inference_size.height,
                    image_meta.inference_size.width,
                    image_meta.inference_size.height,
                    image_meta.inference_size.width,
                ],
                device=self._device,
            )
            selected_boxes_xyxy = selected_boxes_xyxy_pct * inference_size_hwhw
            selected_boxes_xyxy = rescale_image_detections(
                image_detections=selected_boxes_xyxy,
                image_metadata=image_meta,
            )
            detections = Detections(
                xyxy=selected_boxes_xyxy.round().int(),
                confidence=confidence,
                class_id=top_classes.int(),
            )
            results.append(detections)
        return results
