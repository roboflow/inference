import base64
import io
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional

from inference.core.entities.requests import (
    ClassificationInferenceRequest,
    InferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    InstanceSegmentationRLEPrediction,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
    SemanticSegmentationInferenceResponse,
    SemanticSegmentationPrediction,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    INFERENCE_MODELS_INSTANCE_SEGMENTATION_MEMORY_OPTIMIZED_FORMAT,
    INFERENCE_MODELS_INSTANCE_SEGMENTATION_MEMORY_OPTIMIZED_POSTPROCESS,
    RFDETR_ONNX_MAX_RESOLUTION,
    VALID_INFERENCE_MODELS_BACKENDS,
)
from inference.core.models.base import Model
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr, load_image_rgb
from inference.core.utils.postprocess import masks2poly
from inference.core.utils.visualisation import draw_detection_predictions
from inference.models.aliases import resolve_roboflow_model_alias
from inference_models import (
    AutoModel,
    ClassificationModel,
    ClassificationPrediction,
    Detections,
    InstanceDetections,
    InstanceSegmentationModel,
    KeyPoints,
    KeyPointsDetectionModel,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
    ObjectDetectionModel,
    PreProcessingOverrides,
    SemanticSegmentationModel,
)
from inference_models.configuration import (
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLO26_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_BINARIZATION_THRESHOLD,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_SMOOTHING_ENABLED,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
    INFERENCE_MODELS_YOLOV5_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLOV5_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLOV5_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLOV5_DEFAULT_MAX_DETECTIONS,
    INFERENCE_MODELS_YOLOV7_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLOV7_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLOV7_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLOV7_DEFAULT_MAX_DETECTIONS,
)
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import PreprocessingMetadata
from inference_models.models.common.roboflow.post_processing import (
    crop_masks_to_boxes,
    post_process_nms_fused_model_output,
    preprocess_segmentation_masks,
    run_nms_for_instance_segmentation,
)
from inference_models.models.yolov5.nms import run_yolov5_nms_for_instance_segmentation

DEFAULT_COLOR_PALETTE = [
    "#A351FB",
    "#FF4040",
    "#FFA1A0",
    "#FF7633",
    "#FFB633",
    "#D1D435",
    "#4CFB12",
    "#94CF1A",
    "#40DE8A",
    "#1B9640",
    "#00D6C1",
    "#2E9CAA",
    "#00C4FF",
    "#364797",
    "#6675FF",
    "#0019EF",
    "#863AFF",
    "#530087",
    "#CD3AFF",
    "#FF97CA",
    "#FF39C9",
]


def _resize_and_binarize_mask_for_serialization(
    mask: torch.Tensor,
    target_height: int,
    target_width: int,
    output_height: int,
    output_width: int,
    binarization_threshold: float,
    output_offset_x: int = 0,
    output_offset_y: int = 0,
) -> torch.Tensor:
    if output_offset_x > 0 or output_offset_y > 0:
        aligned_mask = torch.zeros(
            (output_height, output_width),
            dtype=torch.bool,
            device=mask.device,
        )
    else:
        aligned_mask = torch.empty(
            (output_height, output_width),
            dtype=torch.bool,
            device=mask.device,
        )
    resized_mask = functional.resize(
        mask[None, ...],
        [target_height, target_width],
        interpolation=functional.InterpolationMode.BILINEAR,
    )[0]
    aligned_mask[
        output_offset_y : output_offset_y + target_height,
        output_offset_x : output_offset_x + target_width,
    ] = resized_mask.gt_(binarization_threshold)
    return aligned_mask


def _iter_aligned_instance_segmentation_results(
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    padding: Tuple[int, int, int, int],
    scale_width: float,
    scale_height: float,
    original_size: Any,
    size_after_pre_processing: Any,
    inference_size: Any,
    static_crop_offset: Any,
    binarization_threshold: float = 0.0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    if image_bboxes.shape[0] == 0:
        return
    pad_left, pad_top, pad_right, pad_bottom = padding
    offsets = torch.tensor(
        [pad_left, pad_top, pad_left, pad_top],
        device=image_bboxes.device,
    )
    image_bboxes[:, :4].sub_(offsets)
    scale = torch.as_tensor(
        [scale_width, scale_height, scale_width, scale_height],
        dtype=image_bboxes.dtype,
        device=image_bboxes.device,
    )
    image_bboxes[:, :4].div_(scale)
    n, mh, mw = masks.shape
    mask_h_scale = mh / inference_size.height
    mask_w_scale = mw / inference_size.width
    mask_pad_top, mask_pad_bottom, mask_pad_left, mask_pad_right = (
        round(mask_h_scale * pad_top),
        round(mask_h_scale * pad_bottom),
        round(mask_w_scale * pad_left),
        round(mask_w_scale * pad_right),
    )
    if (
        mask_pad_top < 0
        or mask_pad_bottom < 0
        or mask_pad_left < 0
        or mask_pad_right < 0
    ):
        masks = torch.nn.functional.pad(
            masks,
            (
                abs(min(mask_pad_left, 0)),
                abs(min(mask_pad_right, 0)),
                abs(min(mask_pad_top, 0)),
                abs(min(mask_pad_bottom, 0)),
            ),
            "constant",
            0,
        )
        padded_mask_offset_top = max(mask_pad_top, 0)
        padded_mask_offset_bottom = max(mask_pad_bottom, 0)
        padded_mask_offset_left = max(mask_pad_left, 0)
        padded_mask_offset_right = max(mask_pad_right, 0)
        masks = masks[
            :,
            padded_mask_offset_top : masks.shape[1] - padded_mask_offset_bottom,
            padded_mask_offset_left : masks.shape[2] - padded_mask_offset_right,
        ]
    else:
        masks = masks[
            :, mask_pad_top : mh - mask_pad_bottom, mask_pad_left : mw - mask_pad_right
        ]

    if static_crop_offset.offset_x > 0 or static_crop_offset.offset_y > 0:
        static_crop_offsets = torch.as_tensor(
            [
                static_crop_offset.offset_x,
                static_crop_offset.offset_y,
                static_crop_offset.offset_x,
                static_crop_offset.offset_y,
            ],
            dtype=image_bboxes.dtype,
            device=image_bboxes.device,
        )
        image_bboxes[:, :4].add_(static_crop_offsets)

    for mask_id in range(n):
        if static_crop_offset.offset_x > 0 or static_crop_offset.offset_y > 0:
            aligned_mask = _resize_and_binarize_mask_for_serialization(
                mask=masks[mask_id],
                target_height=size_after_pre_processing.height,
                target_width=size_after_pre_processing.width,
                output_height=original_size.height,
                output_width=original_size.width,
                binarization_threshold=binarization_threshold,
                output_offset_x=static_crop_offset.offset_x,
                output_offset_y=static_crop_offset.offset_y,
            )
        else:
            aligned_mask = _resize_and_binarize_mask_for_serialization(
                mask=masks[mask_id],
                target_height=size_after_pre_processing.height,
                target_width=size_after_pre_processing.width,
                output_height=size_after_pre_processing.height,
                output_width=size_after_pre_processing.width,
                binarization_threshold=binarization_threshold,
            )
        yield image_bboxes[mask_id], aligned_mask


def _mask_to_largest_polygon(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy()
    return masks2poly(mask_np[None, ...])[0]


def _mask_to_coco_rle(mask: torch.Tensor) -> Dict[str, object]:
    from pycocotools import mask as mask_utils

    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(mask_np))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


class InferenceModelsObjectDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "object-detection"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: ObjectDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            rf_detr_max_input_resolution=RFDETR_ONNX_MAX_RESOLUTION,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: List[Detections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )

        responses: List[ObjectDetectionInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            class_ids = det.class_id.detach().cpu().numpy()

            predictions: List[ObjectDetectionPrediction] = []

            for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    ObjectDetectionPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        **{"class": class_name},
                        class_id=class_id_int,
                    )
                )

            responses.append(
                ObjectDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


class InferenceModelsInstanceSegmentationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "instance-segmentation"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: InstanceSegmentationModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            rf_detr_max_input_resolution=RFDETR_ONNX_MAX_RESOLUTION,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def _postprocess_memory_optimized(
        self,
        predictions: Any,
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> Optional[List[InstanceSegmentationInferenceResponse]]:
        post_process_stream = getattr(self._model, "_post_process_stream", None)
        if post_process_stream is None:
            return self._postprocess_memory_optimized_for_model(
                predictions=predictions,
                preprocess_return_metadata=preprocess_return_metadata,
                **kwargs,
            )
        with torch.cuda.stream(post_process_stream):
            self._record_predictions_to_stream(
                predictions=predictions,
                stream=post_process_stream,
            )
            optimized_response = self._postprocess_memory_optimized_for_model(
                predictions=predictions,
                preprocess_return_metadata=preprocess_return_metadata,
                **kwargs,
            )
        post_process_stream.synchronize()
        return optimized_response

    def _postprocess_memory_optimized_for_model(
        self,
        predictions: Any,
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> Optional[List[InstanceSegmentationInferenceResponse]]:
        model_class_name = self._model.__class__.__name__
        if model_class_name.startswith("YOLOv5ForInstanceSegmentation"):
            return self._postprocess_yolov5_memory_optimized(
                predictions=predictions,
                preprocess_return_metadata=preprocess_return_metadata,
                **kwargs,
            )
        if model_class_name.startswith("YOLOv7ForInstanceSegmentation"):
            return self._postprocess_yolov7_memory_optimized(
                predictions=predictions,
                preprocess_return_metadata=preprocess_return_metadata,
                **kwargs,
            )
        if model_class_name.startswith("YOLOv8ForInstanceSegmentation"):
            return self._postprocess_yolov8_memory_optimized(
                predictions=predictions,
                preprocess_return_metadata=preprocess_return_metadata,
                **kwargs,
            )
        if model_class_name.startswith("YOLO26ForInstanceSegmentation"):
            return self._postprocess_yolo26_memory_optimized(
                predictions=predictions,
                preprocess_return_metadata=preprocess_return_metadata,
                **kwargs,
            )
        if model_class_name.startswith("RFDetrForInstanceSegmentation"):
            return self._postprocess_rfdetr_memory_optimized(
                predictions=predictions,
                preprocess_return_metadata=preprocess_return_metadata,
                **kwargs,
            )
        return None

    def _record_predictions_to_stream(
        self,
        predictions: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        if isinstance(predictions, torch.Tensor):
            predictions.record_stream(stream)
            return
        if isinstance(predictions, (list, tuple)):
            for prediction in predictions:
                self._record_predictions_to_stream(
                    predictions=prediction,
                    stream=stream,
                )

    def _postprocess_yolov5_memory_optimized(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        preprocess_return_metadata: PreprocessingMetadata,
        confidence: float = INFERENCE_MODELS_YOLOV5_DEFAULT_CONFIDENCE,
        iou_threshold: float = INFERENCE_MODELS_YOLOV5_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLOV5_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLOV5_DEFAULT_CLASS_AGNOSTIC_NMS,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        instances, protos = predictions
        nms_results = run_yolov5_nms_for_instance_segmentation(
            output=instances.permute(0, 2, 1),
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            max_detections=max_detections,
            class_agnostic=class_agnostic_nms,
        )
        return self._build_memory_optimized_prototype_mask_response(
            nms_results=nms_results,
            protos=protos,
            preprocess_return_metadata=preprocess_return_metadata,
            binarization_threshold=0.0,
            class_filter=kwargs.get("class_filter"),
        )

    def _postprocess_yolov7_memory_optimized(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        preprocess_return_metadata: PreprocessingMetadata,
        confidence: float = INFERENCE_MODELS_YOLOV7_DEFAULT_CONFIDENCE,
        iou_threshold: float = INFERENCE_MODELS_YOLOV7_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLOV7_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLOV7_DEFAULT_CLASS_AGNOSTIC_NMS,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        instances, protos = predictions
        nms_results = run_nms_for_instance_segmentation(
            output=instances.permute(0, 2, 1),
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            max_detections=max_detections,
            class_agnostic=class_agnostic_nms,
        )
        return self._build_memory_optimized_prototype_mask_response(
            nms_results=nms_results,
            protos=protos,
            preprocess_return_metadata=preprocess_return_metadata,
            binarization_threshold=0.0,
            class_filter=kwargs.get("class_filter"),
        )

    def _postprocess_yolov8_memory_optimized(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        preprocess_return_metadata: PreprocessingMetadata,
        confidence: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
        iou_threshold: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = (
            INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS
        ),
        masks_smoothing_enabled: bool = (
            INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_SMOOTHING_ENABLED
        ),
        masks_binarization_threshold: float = (
            INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_BINARIZATION_THRESHOLD
        ),
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        instances, protos = predictions
        if self._model._inference_config.post_processing.fused:
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
        return self._build_memory_optimized_prototype_mask_response(
            nms_results=nms_results,
            protos=protos,
            preprocess_return_metadata=preprocess_return_metadata,
            binarization_threshold=masks_binarization_threshold,
            class_filter=kwargs.get("class_filter"),
            masks_smoothing_enabled=masks_smoothing_enabled,
        )

    def _postprocess_yolo26_memory_optimized(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        preprocess_return_metadata: PreprocessingMetadata,
        confidence: float = INFERENCE_MODELS_YOLO26_DEFAULT_CONFIDENCE,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        instances, protos = predictions
        nms_results = post_process_nms_fused_model_output(
            output=instances, conf_thresh=confidence
        )
        return self._build_memory_optimized_prototype_mask_response(
            nms_results=nms_results,
            protos=protos,
            preprocess_return_metadata=preprocess_return_metadata,
            binarization_threshold=0.0,
            class_filter=kwargs.get("class_filter"),
        )

    def _postprocess_rfdetr_memory_optimized(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        preprocess_return_metadata: PreprocessingMetadata,
        confidence: float = INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        all_bboxes, all_logits, all_masks = predictions
        all_logits = torch.nn.functional.sigmoid(all_logits)
        responses = []
        for image_bboxes, image_logits, image_masks, image_meta in zip(
            all_bboxes, all_logits, all_masks, preprocess_return_metadata
        ):
            image_confidence, top_classes = image_logits.max(dim=1)
            confidence_mask = image_confidence > confidence
            image_confidence = image_confidence[confidence_mask]
            top_classes = top_classes[confidence_mask]
            selected_boxes = image_bboxes[confidence_mask]
            selected_masks = image_masks[confidence_mask]
            image_confidence, sorted_indices = torch.sort(
                image_confidence, descending=True
            )
            top_classes = top_classes[sorted_indices]
            selected_boxes = selected_boxes[sorted_indices]
            selected_masks = selected_masks[sorted_indices]
            classes_re_mapping = getattr(self._model, "_classes_re_mapping", None)
            if classes_re_mapping is not None:
                remapping_mask = torch.isin(
                    top_classes, classes_re_mapping.remaining_class_ids
                )
                top_classes = classes_re_mapping.class_mapping[
                    top_classes[remapping_mask]
                ]
                selected_boxes = selected_boxes[remapping_mask]
                image_confidence = image_confidence[remapping_mask]
                selected_masks = selected_masks[remapping_mask]
            class_keep_mask = self._get_class_keep_mask(
                class_ids=top_classes,
                class_filter=kwargs.get("class_filter"),
            )
            top_classes = top_classes[class_keep_mask]
            selected_boxes = selected_boxes[class_keep_mask]
            image_confidence = image_confidence[class_keep_mask]
            selected_masks = selected_masks[class_keep_mask]
            cxcy = selected_boxes[:, :2]
            wh = selected_boxes[:, 2:]
            selected_boxes_xyxy_pct = torch.cat(
                [cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1
            )
            denorm_size = (
                image_meta.nonsquare_intermediate_size or image_meta.inference_size
            )
            denorm_size_whwh = torch.tensor(
                [
                    denorm_size.width,
                    denorm_size.height,
                    denorm_size.width,
                    denorm_size.height,
                ],
                device=all_bboxes.device,
            )
            selected_boxes_xyxy = selected_boxes_xyxy_pct * denorm_size_whwh
            image_bboxes_for_alignment = torch.cat(
                [
                    selected_boxes_xyxy,
                    image_confidence[:, None],
                    top_classes[:, None].float(),
                ],
                dim=1,
            )
            responses.append(
                self._build_memory_optimized_image_response(
                    image_bboxes=image_bboxes_for_alignment,
                    masks=selected_masks,
                    image_meta=image_meta,
                    inference_size=denorm_size,
                    binarization_threshold=0.0,
                )
            )
        return responses

    def _build_memory_optimized_prototype_mask_response(
        self,
        nms_results: List[torch.Tensor],
        protos: torch.Tensor,
        preprocess_return_metadata: PreprocessingMetadata,
        binarization_threshold: float,
        class_filter: Optional[List[str]],
        masks_smoothing_enabled: bool = False,
    ) -> List[InstanceSegmentationInferenceResponse]:
        responses = []
        for image_bboxes, image_protos, image_meta in zip(
            nms_results, protos, preprocess_return_metadata
        ):
            image_bboxes = self._filter_image_bboxes_by_class(
                image_bboxes=image_bboxes,
                class_filter=class_filter,
            )
            pre_processed_masks = preprocess_segmentation_masks(
                protos=image_protos,
                masks_in=image_bboxes[:, 6:],
            )
            if masks_smoothing_enabled:
                pre_processed_masks = torch.nn.functional.sigmoid(pre_processed_masks)
            cropped_masks = crop_masks_to_boxes(
                image_bboxes[:, :4], pre_processed_masks
            )
            responses.append(
                self._build_memory_optimized_image_response(
                    image_bboxes=image_bboxes,
                    masks=cropped_masks,
                    image_meta=image_meta,
                    inference_size=image_meta.inference_size,
                    binarization_threshold=binarization_threshold,
                )
            )
        return responses

    def _build_memory_optimized_image_response(
        self,
        image_bboxes: torch.Tensor,
        masks: torch.Tensor,
        image_meta: PreprocessingMetadata,
        inference_size,
        binarization_threshold: float,
    ) -> InstanceSegmentationInferenceResponse:
        response_predictions = []
        padding = (
            image_meta.pad_left,
            image_meta.pad_top,
            image_meta.pad_right,
            image_meta.pad_bottom,
        )
        for aligned_box, aligned_mask in _iter_aligned_instance_segmentation_results(
            image_bboxes=image_bboxes,
            masks=masks,
            padding=padding,
            scale_height=image_meta.scale_height,
            scale_width=image_meta.scale_width,
            original_size=image_meta.original_size,
            size_after_pre_processing=image_meta.size_after_pre_processing,
            inference_size=inference_size,
            static_crop_offset=image_meta.static_crop_offset,
            binarization_threshold=binarization_threshold,
        ):
            response_predictions.append(
                self._build_serialized_instance_prediction(
                    aligned_box=aligned_box,
                    aligned_mask=aligned_mask,
                )
            )
        return InstanceSegmentationInferenceResponse(
            predictions=response_predictions,
            image=InferenceResponseImage(
                width=image_meta.original_size.width,
                height=image_meta.original_size.height,
            ),
        )

    def _build_serialized_instance_prediction(
        self,
        aligned_box: torch.Tensor,
        aligned_mask: torch.Tensor,
    ) -> Union[InstanceSegmentationPrediction, InstanceSegmentationRLEPrediction]:
        aligned_box_cpu = aligned_box.detach().cpu()
        x1, y1, x2, y2 = aligned_box_cpu[:4].round().int().tolist()
        conf = float(aligned_box_cpu[4])
        class_id_int = int(aligned_box_cpu[5])
        class_name = self._class_name_for_id(class_id=class_id_int)
        prediction_kwargs = {
            "x": (float(x1) + float(x2)) / 2.0,
            "y": (float(y1) + float(y2)) / 2.0,
            "width": float(x2) - float(x1),
            "height": float(y2) - float(y1),
            "confidence": float(conf),
            "class": class_name,
            "class_id": class_id_int,
        }
        if INFERENCE_MODELS_INSTANCE_SEGMENTATION_MEMORY_OPTIMIZED_FORMAT == "rle":
            return InstanceSegmentationRLEPrediction(
                **{
                    **prediction_kwargs,
                    "rle": _mask_to_coco_rle(mask=aligned_mask),
                }
            )
        polygon = _mask_to_largest_polygon(mask=aligned_mask)
        return InstanceSegmentationPrediction(
            **{
                **prediction_kwargs,
                "points": [Point(x=point[0], y=point[1]) for point in polygon],
            }
        )

    def _filter_image_bboxes_by_class(
        self,
        image_bboxes: torch.Tensor,
        class_filter: Optional[List[str]],
    ) -> torch.Tensor:
        keep_mask = self._get_class_keep_mask(
            class_ids=image_bboxes[:, 5],
            class_filter=class_filter,
        )
        return image_bboxes[keep_mask]

    def _get_class_keep_mask(
        self,
        class_ids: torch.Tensor,
        class_filter: Optional[List[str]],
    ) -> torch.Tensor:
        if not class_filter or class_ids.shape[0] == 0:
            return torch.ones(
                (class_ids.shape[0],),
                dtype=torch.bool,
                device=class_ids.device,
            )
        class_ids_list = class_ids.detach().cpu().int().tolist()
        keep = [
            self._class_name_for_id(class_id=class_id) in class_filter
            for class_id in class_ids_list
        ]
        return torch.as_tensor(keep, dtype=torch.bool, device=class_ids.device)

    def _class_name_for_id(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def postprocess(
        self,
        predictions: List[InstanceDetections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if INFERENCE_MODELS_INSTANCE_SEGMENTATION_MEMORY_OPTIMIZED_POSTPROCESS:
            optimized_response = self._postprocess_memory_optimized(
                predictions=predictions,
                preprocess_return_metadata=preprocess_return_metadata,
                **mapped_kwargs,
            )
            if optimized_response is not None:
                return optimized_response
        detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )

        responses: List[InstanceSegmentationInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            masks = det.mask.detach().cpu().numpy()
            polys = masks2poly(masks)
            class_ids = det.class_id.detach().cpu().numpy()

            predictions: List[InstanceSegmentationPrediction] = []

            for (x1, y1, x2, y2), mask_as_poly, conf, class_id in zip(
                xyxy, polys, confs, class_ids
            ):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    InstanceSegmentationPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        points=[
                            Point(x=point[0], y=point[1]) for point in mask_as_poly
                        ],
                        **{"class": class_name},
                        class_id=class_id_int,
                    )
                )

            responses.append(
                InstanceSegmentationInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


class InferenceModelsKeyPointsDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "keypoint-detection"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: KeyPointsDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        if "request" in kwargs:
            keypoint_confidence_threshold = kwargs["request"].keypoint_confidence
            kwargs["key_points_threshold"] = keypoint_confidence_threshold
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Tuple[List[KeyPoints], Optional[List[Detections]]],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        keypoints_list, detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )
        if detections_list is None:
            raise RuntimeError(
                "Keypoints detection model does not provide instances detection - this is not supported for "
                "models from `inference-models` package which are adapted to work with `inference`."
            )
        key_points_classes = self._model.key_points_classes
        responses: List[KeypointsDetectionInferenceResponse] = []
        for preproc_metadata, keypoints, det in zip(
            preprocess_return_metadata, keypoints_list, detections_list
        ):

            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            class_ids = det.class_id.detach().cpu().numpy()
            keypoints_xy = keypoints.xy.detach().cpu().tolist()
            keypoints_class_id = keypoints.class_id.detach().cpu().tolist()
            keypoints_confidence = keypoints.confidence.detach().cpu().tolist()
            predictions: List[KeypointsPrediction] = []

            for (
                (x1, y1, x2, y2),
                conf,
                class_id,
                instance_keypoints_xy,
                instance_keypoints_class_id,
                instance_keypoints_confidence,
            ) in zip(
                xyxy,
                confs,
                class_ids,
                keypoints_xy,
                keypoints_class_id,
                keypoints_confidence,
            ):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    KeypointsPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        **{"class": class_name},
                        class_id=class_id_int,
                        keypoints=model_keypoints_to_response(
                            instance_keypoints_xy=instance_keypoints_xy,
                            instance_keypoints_confidence=instance_keypoints_confidence,
                            instance_keypoints_class_id=instance_keypoints_class_id,
                            key_points_classes=key_points_classes,
                        ),
                    )
                )

            responses.append(
                KeypointsDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )

        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


def model_keypoints_to_response(
    instance_keypoints_xy: List[
        List[Union[float, int]]
    ],  # (num_key_points_foc_class_of_object, 2)
    instance_keypoints_confidence: List[float],  # (instance_key_points, )
    instance_keypoints_class_id: int,
    key_points_classes: List[List[str]],
) -> List[Keypoint]:
    keypoint_classes = key_points_classes[instance_keypoints_class_id]
    results = []
    for keypoint_class_id, ((x, y), confidence, keypoint_class_name) in enumerate(
        zip(instance_keypoints_xy, instance_keypoints_confidence, keypoint_classes)
    ):
        if confidence <= 0.0:
            continue
        keypoint = Keypoint(
            x=x,
            y=y,
            confidence=confidence,
            class_id=keypoint_class_id,
            **{"class": keypoint_class_name},
        )
        results.append(keypoint)
    return results


class InferenceModelsClassificationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "classification"
        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: Union[ClassificationModel, MultiLabelClassificationModel] = (
            AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self.api_key,
                allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
                allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
                weights_provider_extra_headers=extra_weights_provider_headers,
                backend=backend,
                **kwargs,
            )
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        images_shapes = [i.shape[:2] for i in np_images]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs), images_shapes

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Tuple[List[KeyPoints], Optional[List[Detections]]],
        returned_metadata: List[Tuple[int, int]],
        **kwargs,
    ) -> Union[
        List[MultiLabelClassificationInferenceResponse],
        List[ClassificationInferenceResponse],
    ]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        post_processed_predictions = self._model.post_process(
            predictions, **mapped_kwargs
        )
        if isinstance(post_processed_predictions, list):
            # multi-label classification
            return prepare_multi_label_classification_response(
                post_processed_predictions,
                image_sizes=returned_metadata,
                class_names=self.class_names,
                confidence_threshold=kwargs.get("confidence", 0.5),
            )
        else:
            # single-label classification
            return prepare_classification_response(
                post_processed_predictions,
                image_sizes=returned_metadata,
                class_names=self.class_names,
                confidence_threshold=kwargs.get("confidence", 0.5),
            )

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def infer_from_request(
        self,
        request: ClassificationInferenceRequest,
    ) -> Union[List[InferenceResponse], InferenceResponse]:
        """
        Handle an inference request to produce an appropriate response.

        Args:
            request (ClassificationInferenceRequest): The request object encapsulating the image(s) and relevant parameters.

        Returns:
            Union[List[InferenceResponse], InferenceResponse]: The response object(s) containing the predictions, visualization, and other pertinent details. If a list of images was provided, a list of responses is returned. Otherwise, a single response is returned.

        Notes:
            - Starts a timer at the beginning to calculate inference time.
            - Processes the image(s) through the `infer` method.
            - Generates the appropriate response object(s) using `make_response`.
            - Calculates and sets the time taken for inference.
            - If visualization is requested, the predictions are drawn on the image.
        """
        t1 = perf_counter()
        responses = self.infer(**request.dict(), return_image_dims=True)
        for response in responses:
            response.time = perf_counter() - t1
            response.inference_id = getattr(request, "id", None)

        if request.visualize_predictions:
            for response in responses:
                response.visualization = draw_predictions(
                    request, response, self.class_names
                )

        if not isinstance(request.image, list):
            responses = responses[0]

        return responses


def prepare_multi_label_classification_response(
    post_processed_predictions: List[MultiLabelClassificationPrediction],
    image_sizes: List[Tuple[int, int]],
    class_names: List[str],
    confidence_threshold: float,
) -> List[MultiLabelClassificationInferenceResponse]:
    results = []
    for prediction, image_size in zip(post_processed_predictions, image_sizes):
        image_predictions_dict = dict()
        predicted_classes = []
        for class_id, confidence in enumerate(prediction.confidence.cpu().tolist()):
            cls_name = class_names[class_id]
            image_predictions_dict[cls_name] = {
                "confidence": confidence,
                "class_id": class_id,
            }
            if confidence > confidence_threshold:
                predicted_classes.append(cls_name)
        results.append(
            MultiLabelClassificationInferenceResponse(
                predictions=image_predictions_dict,
                predicted_classes=predicted_classes,
                image=InferenceResponseImage(width=image_size[1], height=image_size[0]),
                # essentially pushing a dummy values as I have no intention breaking the new API for the sake of delivering value that has no practical use
            )
        )
    return results


def prepare_classification_response(
    post_processed_predictions: ClassificationPrediction,
    image_sizes: List[Tuple[int, int]],
    class_names: List[str],
    confidence_threshold: float,
) -> List[ClassificationInferenceResponse]:
    responses = []
    for classes_confidence, image_size in zip(
        post_processed_predictions.confidence.cpu().tolist(), image_sizes
    ):
        individual_classes_predictions = []
        for i, cls_name in enumerate(class_names):
            class_score = float(classes_confidence[i])
            if class_score < confidence_threshold:
                continue
            class_prediction = {
                "class_id": i,
                "class": cls_name,
                "confidence": round(class_score, 4),
            }
            individual_classes_predictions.append(class_prediction)
        individual_classes_predictions = sorted(
            individual_classes_predictions, key=lambda x: x["confidence"], reverse=True
        )
        response = ClassificationInferenceResponse(
            image=InferenceResponseImage(width=image_size[1], height=image_size[0]),
            # essentially pushing a dummy values as I have no intention breaking the new API for the sake of delivering value that has no practical use
            predictions=individual_classes_predictions,
            top=(
                individual_classes_predictions[0]["class"]
                if individual_classes_predictions
                else ""
            ),
            confidence=(
                individual_classes_predictions[0]["confidence"]
                if individual_classes_predictions
                else 0.0
            ),
        )
        responses.append(response)
    return responses


def draw_predictions(inference_request, inference_response, class_names: List[str]):
    """Draw prediction visuals on an image.

    This method overlays the predictions on the input image, including drawing rectangles and text to visualize the predicted classes.

    Args:
        inference_request: The request object containing the image and parameters.
        inference_response: The response object containing the predictions and other details.
        class_names: List of class names corresponding to the model's classes.

    Returns:
        bytes: The bytes of the visualized image in JPEG format.
    """
    image = load_image_rgb(inference_request.image)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    class_id_2_color = {
        i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, class_name in enumerate(class_names)
    }
    if isinstance(inference_response.predictions, list):
        prediction = inference_response.predictions[0]
        color = class_id_2_color.get(prediction.class_id, "#4892EA")
        draw.rectangle(
            [0, 0, image.size[1], image.size[0]],
            outline=color,
            width=inference_request.visualization_stroke_width,
        )
        text = f"{prediction.class_id} - {prediction.class_name} {prediction.confidence:.2f}"
        text_size = font.getbbox(text)

        # set button size + 10px margins
        button_size = (text_size[2] + 20, text_size[3] + 20)
        button_img = Image.new("RGBA", button_size, color)
        # put text on button with 10px margins
        button_draw = ImageDraw.Draw(button_img)
        button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

        # put button on source image in position (0, 0)
        image.paste(button_img, (0, 0))
    else:
        if len(inference_response.predictions) > 0:
            box_color = "#4892EA"
            draw.rectangle(
                [0, 0, image.size[1], image.size[0]],
                outline=box_color,
                width=inference_request.visualization_stroke_width,
            )
        row = 0
        predictions = [
            (cls_name, pred)
            for cls_name, pred in inference_response.predictions.items()
        ]
        predictions = sorted(predictions, key=lambda x: x[1].confidence, reverse=True)
        for i, (cls_name, pred) in enumerate(predictions):
            color = class_id_2_color.get(cls_name, "#4892EA")
            text = f"{cls_name} {pred.confidence:.2f}"
            text_size = font.getbbox(text)

            # set button size + 10px margins
            button_size = (text_size[2] + 20, text_size[3] + 20)
            button_img = Image.new("RGBA", button_size, color)
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

            # put button on source image in position (0, 0)
            image.paste(button_img, (0, row))
            row += button_size[1]

    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


class InferenceModelsSemanticSegmentationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "semantic-segmentation"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: SemanticSegmentationModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    @property
    def class_map(self):
        # match segment.roboflow.com
        return {str(k): v for k, v in enumerate(self.class_names)}

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: torch.Tensor,
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[SemanticSegmentationInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        segmentation_results = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )

        responses: List[SemanticSegmentationInferenceResponse] = []
        for preproc_metadata, segmentation in zip(
            preprocess_return_metadata, segmentation_results
        ):
            height = preproc_metadata.original_size.height
            width = preproc_metadata.original_size.width
            response_image = InferenceResponseImage(width=width, height=height)
            # WARNING! This way of conversion is hazardous - first of all, if background class is not in class names,
            # for certain pre-processing, we end up with -1 values which will be wrapped to 255 - second of all,
            # we can support only 256 classes - those constraints should be fine until inference 2.0
            response_predictions = SemanticSegmentationPrediction(
                segmentation_mask=self.img_to_b64_str(
                    segmentation.segmentation_map.to(torch.uint8)
                ),
                confidence_mask=self.img_to_b64_str(
                    (segmentation.confidence * 255).to(torch.uint8)
                ),
                class_map=self.class_map,
                image=dict(response_image),
            )
            response = SemanticSegmentationInferenceResponse(
                predictions=response_predictions,
                image=response_image,
            )
            responses.append(response)
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def img_to_b64_str(self, img: torch.Tensor) -> str:
        if img.dtype != torch.uint8:
            raise ValueError(
                f"img_to_b64_str requires uint8 tensor but got dtype {img.dtype}"
            )

        img = Image.fromarray(img.cpu().numpy())
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        return img_str

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        raise NotImplementedError(
            "draw_predictions(...) is not implemented for semantic segmentation models - responses contain "
            "visualization already."
        )
