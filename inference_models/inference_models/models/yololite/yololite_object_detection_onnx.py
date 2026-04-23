from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, ObjectDetectionModel, PreProcessingOverrides
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_YOLOLITE_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLOLITE_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLOLITE_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLOLITE_DEFAULT_MAX_DETECTIONS,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import (
    run_onnx_session_with_batch_size_limit,
    set_onnx_execution_provider_defaults,
)
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    post_process_nms_fused_model_output,
    rescale_detections,
    run_nms_for_object_detection,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLOLite model with ONNX backend requires onnxruntime installation, which is brought with "
        "`onnx-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        "model. You can also contact Roboflow to get support. "
        "Additionally - if AutoModel.from_pretrained(...) "
        "automatically selects model package which does not match your environment - that's a serious problem and "
        "we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLOLiteForObjectDetectionOnnx(
    ObjectDetectionModel[torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, ...]]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "YOLOLiteForObjectDetectionOnnx":
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message="Could not initialize model - selected backend is ONNX which requires execution provider to "
                "be specified - explicitly in `from_pretrained(...)` method or via env variable "
                "`ONNXRUNTIME_EXECUTION_PROVIDERS`. If you run model locally - adjust your setup, otherwise "
                "contact the platform support.",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#environmentconfigurationerror",
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
                    "YOLOLite Object Detection model running with ONNX backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `letterbox` "
                    "resize mode with gray edges will be used instead.",
                )
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
            device=device,
            input_batch_size=input_batch_size,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
        input_batch_size: Optional[int],
        recommended_parameters=None,
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device
        self._min_batch_size = input_batch_size
        self._max_batch_size = (
            input_batch_size
            if input_batch_size is not None
            else inference_config.forward_pass.max_dynamic_batch_size
        )
        self._session_thread_lock = Lock()
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Union[Tuple[int, int], int]] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
            pre_processing_overrides=pre_processing_overrides,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        with self._session_thread_lock:
            outputs = run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs={self._input_name: pre_processed_images},
                min_batch_size=self._min_batch_size,
                max_batch_size=self._max_batch_size,
            )
        return tuple(outputs)

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, ...],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        iou_threshold: float = INFERENCE_MODELS_YOLOLITE_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLOLITE_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLOLITE_DEFAULT_CLASS_AGNOSTIC_NMS,
        **kwargs,
    ) -> List[Detections]:
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_YOLOLITE_DEFAULT_CONFIDENCE,
        )
        confidence = confidence_filter.get_threshold(self.class_names)
        # Backward compatibility: earlier model packages have no post_processing config — always unfused 3-tensor output
        if (
            self._inference_config.post_processing
            and self._inference_config.post_processing.fused
        ):
            nms_results = self._post_process_fused(model_results, confidence)
        else:
            nms_results = self._post_process_unfused(
                model_results,
                confidence,
                iou_threshold,
                max_detections,
                class_agnostic_nms,
            )
        rescaled_results = rescale_detections(
            detections=nms_results,
            images_metadata=pre_processing_meta,
        )
        results = []
        for result in rescaled_results:
            results.append(
                Detections(
                    xyxy=result[:, :4].round().int(),
                    class_id=result[:, 5].int(),
                    confidence=result[:, 4],
                )
            )
        return results

    def _post_process_fused(
        self,
        model_results: Tuple[torch.Tensor, ...],
        confidence: Union[float, torch.Tensor],
    ) -> List[torch.Tensor]:
        # Single output tensor [B, max_det, 6]: x1, y1, x2, y2, conf, class_id
        output = model_results[0]
        return post_process_nms_fused_model_output(
            output=output, conf_thresh=confidence
        )

    def _post_process_unfused(
        self,
        model_results: Tuple[torch.Tensor, ...],
        confidence: Union[float, torch.Tensor],
        iou_threshold: float,
        max_detections: int,
        class_agnostic_nms: bool,
    ) -> List[torch.Tensor]:
        # Decoded outputs without fused NMS: boxes_xyxy [B,N,4], obj_logits [B,N,1], cls_logits [B,N,C]
        boxes_xyxy, obj_logits, cls_logits = (
            model_results[0],
            model_results[1],
            model_results[2],
        )
        obj_conf = torch.sigmoid(obj_logits)
        cls_conf = torch.sigmoid(cls_logits)
        combined_scores = obj_conf * cls_conf

        boxes_t = boxes_xyxy.permute(0, 2, 1)
        scores_t = combined_scores.permute(0, 2, 1)
        nms_input = torch.cat([boxes_t, scores_t], dim=1)

        return run_nms_for_object_detection(
            output=nms_input,
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            max_detections=max_detections,
            class_agnostic=class_agnostic_nms,
            box_format="xyxy",
        )
