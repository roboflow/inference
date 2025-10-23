from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from inference_exp import InstanceDetections, InstanceSegmentationModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
    ModelRuntimeError,
)
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
    align_instance_segmentation_results,
    crop_masks_to_boxes,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.utils.onnx_introspection import get_selected_onnx_execution_providers

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import YOLOv5 model with ONNX backend - this error means that some additional dependencies "
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
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["weights.onnx"],
            providers=onnx_execution_providers,
        )
        input_batch_size = session.get_inputs()[0].shape[0]
        if input_batch_size != 1:
            raise ModelRuntimeError(
                message="Implementation of YOLOACTForInstanceSegmentationOnnx is adjusted to work correctly with "
                "onnx models accepting inputs with `batch_size=1`. It can be extended if needed, but we've "
                "not heard such request so far. If you find that a valueble feature - let us know via "
                "https://github.com/roboflow/inference/issues"
            )
        input_name = session.get_inputs()[0].name
        return cls(
            session=session,
            input_name=input_name,
            class_names=class_names,
            inference_config=inference_config,
            device=device,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device
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
                    run_session_with_batch_size_limit(
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
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        max_detections: int = 100,
        class_agnostic: bool = False,
        **kwargs,
    ) -> List[InstanceDetections]:
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
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_detections=max_detections,
            class_agnostic=class_agnostic,
        )
        final_results = []
        for image_bboxes, image_protos, image_meta in zip(
            nms_results, all_proto_data, pre_processing_meta
        ):
            pre_processed_masks = image_protos @ image_bboxes[:, 6:].T
            pre_processed_masks = 1 / (1 + torch.exp(-pre_processed_masks))
            pre_processed_masks = torch.permute(pre_processed_masks, (2, 0, 1))
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
                binarization_threshold=0.5,
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


def decode_masks(
    boxes: torch.Tensor, masks: torch.Tensor, proto: torch.Tensor, img_dim
):
    ret_mask = proto @ boxes[:, 6:].T
    ret_mask = 1 / (1 + torch.exp(-ret_mask))
    ret_mask = crop_masks_to_boxes(boxes=boxes[:, 4], masks=ret_mask)


def run_nms_for_instance_segmentation(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
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
