from threading import Lock
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torchvision

from inference_models import (
    InstanceDetections,
    InstanceSegmentationMaskFormat,
    InstanceSegmentationModel,
    PreProcessingOverrides,
)
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_YOLACT_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLACT_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLACT_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLACT_DEFAULT_MAX_DETECTIONS,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
    ModelInputError,
    ModelRuntimeError,
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
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.yolact.common import prepare_dense_masks, prepare_rle_masks
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLA-CT model with ONNX backend requires pycuda installation, which is brought with "
        "`onnx-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLOACTForInstanceSegmentationOnnx(
    InstanceSegmentationModel[
        torch.Tensor,
        PreProcessingMetadata,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
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
    ) -> "YOLOACTForInstanceSegmentationOnnx":
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message=f"Could not initialize model - selected backend is ONNX which requires execution provider to "
                f"be specified - explicitly in `from_pretrained(...)` method or via env variable "
                f"`ONNXRUNTIME_EXECUTION_PROVIDERS`. If you run model locally - adjust your setup, otherwise "
                f"contact the platform support.",
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
                    "YOLACT model running with ONNX backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `letterbox` "
                    "resize mode with gray edges will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["weights.onnx"],
            providers=onnx_execution_providers,
        )
        input_batch_size = session.get_inputs()[0].shape[0]
        if input_batch_size != 1:
            raise ModelRuntimeError(
                message="Implementation of YOLOACTForInstanceSegmentationOnnx is adjusted to work correctly with "
                "onnx models accepting inputs with `batch_size=1`. It can be extended if needed, but we've "
                "not heard such request so far. If you find that a valuable feature - let us know via "
                "https://github.com/roboflow/inference/issues",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        input_name = session.get_inputs()[0].name
        return cls(
            session=session,
            input_name=input_name,
            class_names=class_names,
            inference_config=inference_config,
            device=device,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
        recommended_parameters=None,
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device
        self._session_thread_lock = Lock()
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def supported_mask_formats(self) -> Set[InstanceSegmentationMaskFormat]:
        return {"dense", "rle"}

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            pre_processing_overrides=pre_processing_overrides,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._session_thread_lock:
            (
                all_loc_data,
                all_conf_data,
                all_mask_data,
                all_prior_data,
                all_proto_data,
            ) = ([], [], [], [], [])
            for image in pre_processed_images:
                loc_data, conf_data, mask_data, prior_data, proto_data = (
                    run_onnx_session_with_batch_size_limit(
                        session=self._session,
                        inputs={self._input_name: image.unsqueeze(0).contiguous()},
                    )
                )
                all_loc_data.append(loc_data)
                all_conf_data.append(conf_data)
                all_mask_data.append(mask_data)
                all_prior_data.append(prior_data)
                all_proto_data.append(proto_data)
            return (
                torch.cat(all_loc_data, dim=0),
                torch.cat(all_conf_data, dim=0),
                torch.cat(all_mask_data, dim=0),
                torch.stack(all_prior_data, dim=0),
                torch.cat(all_proto_data, dim=0),
            )

    def post_process(
        self,
        model_results: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        iou_threshold: float = INFERENCE_MODELS_YOLACT_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLACT_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLACT_DEFAULT_CLASS_AGNOSTIC_NMS,
        mask_format: InstanceSegmentationMaskFormat = "dense",
        **kwargs,
    ) -> List[InstanceDetections]:
        if mask_format not in self.supported_mask_formats:
            raise ModelInputError(
                message=f"YOLA-CT Instance Segmentation models support the following mask "
                f"formats: {self.supported_mask_formats}. Requested format: {mask_format} "
                f"is not supported. If you see this error while running on Roboflow platform, "
                f"contact support or raise an issue at https://github.com/roboflow/inference/issues. "
                f"When running locally - please verify your integration to make sure that appropriate "
                f"value of `mask_format` parameter is set.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_YOLACT_DEFAULT_CONFIDENCE,
        )
        confidence = confidence_filter.get_threshold(self.class_names)
        all_loc_data, all_conf_data, all_mask_data, all_prior_data, all_proto_data = (
            model_results
        )
        batch_size = all_loc_data.shape[0]
        num_priors = all_loc_data.shape[1]
        boxes = torch.zeros((batch_size, num_priors, 4), device=self._device)
        for batch_element_id, (
            batch_element_loc_data,
            batch_element_priors,
            image_prep_meta,
        ) in enumerate(zip(all_loc_data, all_prior_data, pre_processing_meta)):
            image_boxes = decode_predicted_bboxes(
                loc_data=batch_element_loc_data,
                priors=batch_element_priors,
            )
            inference_height, inference_width = (
                image_prep_meta.inference_size.height,
                image_prep_meta.inference_size.width,
            )
            scale = torch.tensor(
                [inference_width, inference_height, inference_width, inference_height],
                device=self._device,
            )
            image_boxes = image_boxes.mul_(scale)
            boxes[batch_element_id, :, :] = image_boxes
        all_conf_data = all_conf_data[:, :, 1:]  # remove background class
        instances = torch.cat([boxes, all_conf_data, all_mask_data], dim=2)
        nms_results = run_nms_for_instance_segmentation(
            output=instances,
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            max_detections=max_detections,
            class_agnostic=class_agnostic_nms,
        )
        if mask_format == "dense":
            return prepare_dense_masks(
                nms_results=nms_results,
                all_proto_data=all_proto_data,
                pre_processing_meta=pre_processing_meta,
            )
        return prepare_rle_masks(
            nms_results=nms_results,
            all_proto_data=all_proto_data,
            pre_processing_meta=pre_processing_meta,
        )


def decode_predicted_bboxes(
    loc_data: torch.Tensor, priors: torch.Tensor
) -> torch.Tensor:
    variances = torch.tensor([0.1, 0.2], device=loc_data.device)
    boxes = torch.cat(
        [
            priors[:, :2] + loc_data[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc_data[:, 2:] * variances[1]),
        ],
        dim=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def run_nms_for_instance_segmentation(
    output: torch.Tensor,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    """
    `conf_thresh`: scalar applies to all classes; 1-D tensor of shape
    (num_classes,) indexed by class_id for per-class thresholds.
    """
    bs = output.shape[0]
    boxes = output[:, :, :4]  # (N, 19248, 4)
    scores = output[:, :, 4:-32]  # (N, 19248, num_classes)
    masks = output[:, :, -32:]
    results = []
    for b in range(bs):
        bboxes = boxes[b]  # (19248, 4)
        class_scores = scores[b]  # (19248, 80)
        box_masks = masks[b]
        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)
        if isinstance(conf_thresh, torch.Tensor):
            mask = class_conf > conf_thresh.to(output.device)[class_ids]
        else:
            mask = class_conf > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 38), device=output.device))
            continue
        bboxes = bboxes[mask]
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        box_masks = box_masks[mask]
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        keep = torchvision.ops.batched_nms(
            bboxes, class_conf, nms_class_ids, iou_thresh
        )
        keep = keep[:max_detections]
        detections = torch.cat(
            [
                bboxes[keep],
                class_conf[keep].unsqueeze(1),
                class_ids[keep].unsqueeze(1).float(),
                box_masks[keep],
            ],
            dim=1,
        )  # [x1, y1, x2, y2, conf, cls]
        results.append(detections)
    return results
