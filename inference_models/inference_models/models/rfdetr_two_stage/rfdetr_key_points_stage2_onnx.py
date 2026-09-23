"""Stage 2 of the RF-DETR two-stage keypoint model: a top-down keypoint model served with ONNX Runtime.

The model sees one object per crop. Given an image and object boxes, the crop
of each box is cut with the training geometry, run through the graph together
with its ``s_norm`` scale, and the normalized keypoints are mapped back into
the image. Called without boxes, every image is treated as one crop, which is
how a single-object dataset is served. The two-stage model supplies the boxes
from its stage-1 detector.
"""

import json
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from inference_models import Detections, KeyPoints, KeyPointsDetectionModel
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_RFDETR_STAGE2_DEFAULT_KEY_POINTS_THRESHOLD,
)
from inference_models.developer_tools import align_device_with_onnx_session
from inference_models.entities import ColorFormat
from inference_models.errors import (
    CorruptedModelPackageError,
    EnvironmentConfigurationError,
    MissingDependencyError,
    ModelInputError,
)
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
    parse_key_points_metadata,
)
from inference_models.models.rfdetr_two_stage.crop_geometry import (
    CropGeometry,
    TopDownCropConfig,
    crop_object,
    full_image_box,
    s_norm_for_box,
)
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running RF-DETR keypoint stage-2 model with ONNX backend requires onnxruntime, which is brought with "
        "`onnx-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        "model, You can also contact Roboflow to get support.",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

DEFAULT_MAX_BATCH_SIZE = 32
PIXELS_INPUT = "pixels"
S_NORM_INPUT = "s_norm"
KEYPOINTS_OUTPUT = "keypoints"
SIGMAS_OUTPUT = "sigmas"
PRESENCE_OUTPUT = "presence"
SCORES_OUTPUT = "scores"

ImagesInput = Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
BoxesInput = Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]]]


@dataclass(frozen=True)
class KeyPointSlots:
    """The contiguous output slots one object class owns in the stacked layout."""

    offset: int
    count: int


@dataclass(frozen=True)
class CropRecord:
    image_index: int
    class_id: int
    box_xyxy: np.ndarray
    geometry: CropGeometry
    detection_confidence: Optional[float]


@dataclass(frozen=True)
class TopDownBatchMetadata:
    crops: List[CropRecord]
    images_count: int


RawPosePrediction = Dict[str, torch.Tensor]


class RFDetrKeyPointsStage2ONNX(
    KeyPointsDetectionModel[
        Tuple[torch.Tensor, torch.Tensor], TopDownBatchMetadata, RawPosePrediction
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
    ) -> "RFDetrKeyPointsStage2ONNX":
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message="Could not initialize RF-DETR keypoint stage-2 model with ONNX backend - no ONNX execution providers "
                "are available in this environment.",
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
                "keypoints_metadata.json",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        inference_config = parse_inference_config(
            config_path=model_package_content["inference_config.json"],
            allowed_resize_modes=set(ResizeMode),
        )
        crop_config, key_point_slots = parse_top_down_config(
            config_path=model_package_content["inference_config.json"],
            inference_config=inference_config,
            class_names=class_names,
        )
        key_points_classes, skeletons = parse_key_points_metadata(
            key_points_metadata_path=model_package_content["keypoints_metadata.json"],
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["weights.onnx"],
            providers=onnx_execution_providers,
        )
        device = align_device_with_onnx_session(session=session, device=device)
        return cls(
            session=session,
            class_names=class_names,
            key_points_classes=key_points_classes,
            skeletons=skeletons,
            key_point_slots=key_point_slots,
            crop_config=crop_config,
            inference_config=inference_config,
            device=device,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        class_names: List[str],
        key_points_classes: List[List[str]],
        skeletons: List[List[Tuple[int, int]]],
        key_point_slots: List[KeyPointSlots],
        crop_config: TopDownCropConfig,
        inference_config: InferenceConfig,
        device: torch.device,
    ):
        self._session = session
        self._session_thread_lock = threading.Lock()
        self._class_names = class_names
        self._key_points_classes = key_points_classes
        self._skeletons = skeletons
        self._key_point_slots = key_point_slots
        self._crop_config = crop_config
        self._device = device
        self._max_key_points = max(
            (slots.count for slots in key_point_slots), default=0
        )
        network_input = inference_config.network_input
        self._scaling_factor = float(network_input.scaling_factor or 1.0)
        mean, std = network_input.normalization or ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self._mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        input_names = [model_input.name for model_input in session.get_inputs()]
        self._pixels_input, self._s_norm_input = _resolve_input_names(input_names)
        self._output_names = [
            model_output.name for model_output in session.get_outputs()
        ]
        static_batch_size = session.get_inputs()[0].shape[0]
        if isinstance(static_batch_size, int):
            self._min_batch_size = static_batch_size
            self._max_batch_size = static_batch_size
        else:
            self._min_batch_size = 1
            self._max_batch_size = (
                inference_config.forward_pass.max_dynamic_batch_size
                or DEFAULT_MAX_BATCH_SIZE
            )

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def key_points_classes(self) -> List[List[str]]:
        return self._key_points_classes

    @property
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        return self._skeletons

    @property
    def crop_input_size(self) -> Tuple[int, int]:
        return self._crop_config.input_size

    def pre_process(
        self,
        images: ImagesInput,
        boxes: Optional[List[BoxesInput]] = None,
        class_ids: Optional[
            List[Union[torch.Tensor, np.ndarray, Sequence[int]]]
        ] = None,
        detection_confidences: Optional[
            List[Union[torch.Tensor, np.ndarray, Sequence[float]]]
        ] = None,
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], TopDownBatchMetadata]:
        images_rgb = images_to_numpy_rgb(
            images=images, input_color_format=input_color_format
        )
        if boxes is None:
            boxes = [[full_image_box(image)] for image in images_rgb]
        if len(boxes) != len(images_rgb):
            raise ModelInputError(
                message=f"Stage-2 keypoint model received {len(images_rgb)} images but boxes for {len(boxes)} images.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        crops: List[np.ndarray] = []
        s_norms: List[float] = []
        records: List[CropRecord] = []
        for image_index, (image, image_boxes) in enumerate(zip(images_rgb, boxes)):
            image_boxes = _to_numpy(image_boxes, dtype=np.float32).reshape(-1, 4)
            image_class_ids = (
                _to_numpy(class_ids[image_index], dtype=np.int64).reshape(-1)
                if class_ids is not None
                else np.zeros(len(image_boxes), dtype=np.int64)
            )
            image_confidences = (
                _to_numpy(detection_confidences[image_index], dtype=np.float32).reshape(
                    -1
                )
                if detection_confidences is not None
                else None
            )
            for box_index, box in enumerate(image_boxes):
                class_id = int(image_class_ids[box_index])
                if not 0 <= class_id < len(self._key_point_slots):
                    raise ModelInputError(
                        message=f"Stage-2 keypoint model received class id {class_id}, but the model knows "
                        f"{len(self._key_point_slots)} classes.",
                        help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
                    )
                crop, geometry = crop_object(image, box, self._crop_config)
                crops.append(self._normalize_crop(crop))
                s_norms.append(s_norm_for_box(box, geometry))
                records.append(
                    CropRecord(
                        image_index=image_index,
                        class_id=class_id,
                        box_xyxy=box,
                        geometry=geometry,
                        detection_confidence=(
                            float(image_confidences[box_index])
                            if image_confidences is not None
                            else None
                        ),
                    )
                )
        metadata = TopDownBatchMetadata(crops=records, images_count=len(images_rgb))
        if not crops:
            height, width = self._crop_config.input_size
            empty_pixels = torch.zeros((0, 3, height, width), dtype=torch.float32)
            empty_s_norm = torch.zeros((0,), dtype=torch.float32)
            return (
                empty_pixels.to(self._device),
                empty_s_norm.to(self._device),
            ), metadata
        pixels = torch.from_numpy(np.stack(crops)).to(self._device)
        s_norm = torch.tensor(s_norms, dtype=torch.float32, device=self._device)
        return (pixels, s_norm), metadata

    def forward(
        self, pre_processed_images: Tuple[torch.Tensor, torch.Tensor], **kwargs
    ) -> RawPosePrediction:
        pixels, s_norm = pre_processed_images
        if pixels.shape[0] == 0:
            total_slots = sum(slots.count for slots in self._key_point_slots)
            return {
                KEYPOINTS_OUTPUT: torch.zeros((0, total_slots, 2), device=self._device),
                SIGMAS_OUTPUT: torch.zeros((0, total_slots, 2), device=self._device),
                SCORES_OUTPUT: torch.zeros((0, total_slots), device=self._device),
                PRESENCE_OUTPUT: torch.zeros((0, total_slots), device=self._device),
            }
        with self._session_thread_lock:
            outputs = run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs={self._pixels_input: pixels, self._s_norm_input: s_norm},
                min_batch_size=self._min_batch_size,
                max_batch_size=self._max_batch_size,
            )
        return dict(zip(self._output_names, outputs))

    def post_process(
        self,
        model_results: RawPosePrediction,
        pre_processing_meta: TopDownBatchMetadata,
        key_points_threshold: float = INFERENCE_MODELS_RFDETR_STAGE2_DEFAULT_KEY_POINTS_THRESHOLD,
        **kwargs,
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        keypoints = model_results[KEYPOINTS_OUTPUT].float().cpu().numpy()
        sigmas = model_results[SIGMAS_OUTPUT].float().cpu().numpy()
        presence = model_results[PRESENCE_OUTPUT].float().cpu().numpy()
        scores = model_results[SCORES_OUTPUT].float().cpu().numpy()
        per_image: List[List[_InstanceResult]] = [
            [] for _ in range(pre_processing_meta.images_count)
        ]
        for row, record in enumerate(pre_processing_meta.crops):
            slots = self._key_point_slots[record.class_id]
            slot_slice = slice(slots.offset, slots.offset + slots.count)
            xy = record.geometry.to_image(keypoints[row, slot_slice])
            confidence = presence[row, slot_slice].copy()
            covariance = _covariance_to_image(sigmas[row, slot_slice], record.geometry)
            suppressed = confidence < key_points_threshold
            xy[suppressed] = 0.0
            confidence[suppressed] = 0.0
            covariance[suppressed] = np.nan
            # The graph emits its instance score broadcast over the slots, so any
            # slot of the class carries it. With a caller-supplied detection
            # confidence (the two-stage model's detector) the instance confidence
            # is their product, the fusion the trainer evaluates with.
            pose_score = float(scores[row, slots.offset])
            instance_confidence = (
                record.detection_confidence * pose_score
                if record.detection_confidence is not None
                else pose_score
            )
            per_image[record.image_index].append(
                _InstanceResult(
                    xy=_pad_rows(xy, self._max_key_points),
                    confidence=_pad_rows(confidence, self._max_key_points),
                    covariance=_pad_rows(covariance, self._max_key_points, fill=np.nan),
                    class_id=record.class_id,
                    box_xyxy=record.box_xyxy,
                    detection_confidence=instance_confidence,
                )
            )
        all_key_points: List[KeyPoints] = []
        all_detections: List[Detections] = []
        for instances in per_image:
            all_key_points.append(self._assemble_key_points(instances))
            all_detections.append(self._assemble_detections(instances))
        return all_key_points, all_detections

    def _normalize_crop(self, crop_rgb: np.ndarray) -> np.ndarray:
        pixels = crop_rgb.astype(np.float32).transpose(2, 0, 1) / self._scaling_factor
        return (pixels - self._mean) / self._std

    def _assemble_key_points(self, instances: List["_InstanceResult"]) -> KeyPoints:
        if not instances:
            return KeyPoints(
                xy=torch.zeros(
                    (0, self._max_key_points, 2), dtype=torch.int32, device=self._device
                ),
                class_id=torch.zeros((0,), dtype=torch.int32, device=self._device),
                confidence=torch.zeros((0, self._max_key_points), device=self._device),
                covariance=torch.zeros(
                    (0, self._max_key_points, 2, 2), device=self._device
                ),
                detection_confidence=torch.zeros((0,), device=self._device),
            )
        return KeyPoints(
            xy=torch.from_numpy(np.stack([item.xy for item in instances]))
            .round()
            .int()
            .to(self._device),
            class_id=torch.tensor(
                [item.class_id for item in instances],
                dtype=torch.int32,
                device=self._device,
            ),
            confidence=torch.from_numpy(
                np.stack([item.confidence for item in instances])
            ).to(self._device),
            covariance=torch.from_numpy(
                np.stack([item.covariance for item in instances])
            ).to(self._device),
            detection_confidence=torch.tensor(
                [item.detection_confidence for item in instances],
                dtype=torch.float32,
                device=self._device,
            ),
        )

    def _assemble_detections(self, instances: List["_InstanceResult"]) -> Detections:
        if not instances:
            return Detections(
                xyxy=torch.zeros((0, 4), dtype=torch.int32, device=self._device),
                class_id=torch.zeros((0,), dtype=torch.int32, device=self._device),
                confidence=torch.zeros((0,), device=self._device),
            )
        return Detections(
            xyxy=torch.from_numpy(np.stack([item.box_xyxy for item in instances]))
            .round()
            .int()
            .to(self._device),
            class_id=torch.tensor(
                [item.class_id for item in instances],
                dtype=torch.int32,
                device=self._device,
            ),
            confidence=torch.tensor(
                [item.detection_confidence for item in instances],
                dtype=torch.float32,
                device=self._device,
            ),
        )


@dataclass(frozen=True)
class _InstanceResult:
    xy: np.ndarray
    confidence: np.ndarray
    covariance: np.ndarray
    class_id: int
    box_xyxy: np.ndarray
    detection_confidence: float


def parse_top_down_config(
    config_path: str,
    inference_config: InferenceConfig,
    class_names: List[str],
) -> Tuple[TopDownCropConfig, List[KeyPointSlots]]:
    """Reads the ``top_down`` block the stage-2 trainer adds to ``inference_config.json``."""
    with open(config_path, "r") as config_file:
        raw_config = json.load(config_file)
    top_down = raw_config.get("top_down")
    training_input_size = inference_config.network_input.training_input_size
    if not isinstance(top_down, dict) or training_input_size is None:
        raise CorruptedModelPackageError(
            message="Stage-2 inference_config.json must define `top_down` crop settings and "
            "`network_input.training_input_size`.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    try:
        crop_config = TopDownCropConfig(
            input_size=(
                int(training_input_size.height),
                int(training_input_size.width),
            ),
            context_padding=float(top_down["context_padding"]),
            preserve_aspect=bool(top_down["preserve_aspect"]),
            udp=bool(top_down["udp"]),
        )
        declared_classes = list(top_down["keypoint_classes"])
    except (KeyError, TypeError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Stage-2 `top_down` config is malformed: {error}",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error
    slots_by_name = {
        str(item["name"]): KeyPointSlots(
            offset=int(item["offset"]), count=int(item["count"])
        )
        for item in declared_classes
    }
    key_point_slots = []
    for class_index, class_name in enumerate(class_names):
        slots = slots_by_name.get(class_name)
        if slots is None and class_index < len(declared_classes):
            item = declared_classes[class_index]
            slots = KeyPointSlots(offset=int(item["offset"]), count=int(item["count"]))
        if slots is None:
            raise CorruptedModelPackageError(
                message=f"Stage-2 class `{class_name}` has no keypoint slots in `top_down.keypoint_classes`.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        key_point_slots.append(slots)
    return crop_config, key_point_slots


def images_to_numpy_rgb(
    images: ImagesInput, input_color_format: Optional[ColorFormat] = None
) -> List[np.ndarray]:
    """Any accepted image input -> list of ``(H, W, 3)`` uint8 RGB arrays.

    numpy inputs default to BGR and torch inputs to RGB, as everywhere in the
    library; torch images may be ``CHW`` or ``BCHW`` and in ``[0, 1]`` or ``[0, 255]``.
    """
    if isinstance(images, (np.ndarray, torch.Tensor)):
        if images.ndim == 4:
            images = list(images)
        elif images.ndim == 3:
            images = [images]
        else:
            raise ModelInputError(
                message=f"Stage-2 keypoint model expects images with 3 or 4 dimensions, got {images.ndim}.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
    result = []
    for image in images:
        if isinstance(image, torch.Tensor):
            color_format = input_color_format or "rgb"
            array = image.detach().cpu()
            if array.ndim != 3:
                raise ModelInputError(
                    message=f"Stage-2 keypoint model expects `CHW` torch images, got shape {tuple(array.shape)}.",
                    help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
                )
            if array.shape[0] in (1, 3) and array.shape[2] not in (1, 3):
                array = array.permute(1, 2, 0)
            array = array.numpy()
        else:
            color_format = input_color_format or "bgr"
            array = np.asarray(image)
        array = _to_uint8_hwc(array)
        if color_format == "bgr":
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        result.append(np.ascontiguousarray(array))
    return result


def _to_uint8_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] != 3:
        raise ModelInputError(
            message=f"Stage-2 keypoint model expects 3-channel images, got shape {array.shape}.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _resolve_input_names(input_names: List[str]) -> Tuple[str, str]:
    if PIXELS_INPUT in input_names and S_NORM_INPUT in input_names:
        return PIXELS_INPUT, S_NORM_INPUT
    if len(input_names) == 2:
        return input_names[0], input_names[1]
    raise CorruptedModelPackageError(
        message=f"Stage-2 ONNX graph must take `{PIXELS_INPUT}` and `{S_NORM_INPUT}` inputs, got {input_names}.",
        help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
    )


def _covariance_to_image(
    sigmas_normalized: np.ndarray, geometry: CropGeometry
) -> np.ndarray:
    """Axial normalized-crop sigmas -> (K, 2, 2) pixel-space covariance."""
    linear = geometry.normalized_to_image[:, :2].astype(np.float32)
    variances = np.square(np.asarray(sigmas_normalized, dtype=np.float32))
    diagonal = np.zeros((variances.shape[0], 2, 2), dtype=np.float32)
    diagonal[:, 0, 0] = variances[:, 0]
    diagonal[:, 1, 1] = variances[:, 1]
    return np.einsum("ij,kjl,ml->kim", linear, diagonal, linear)


def _pad_rows(values: np.ndarray, size: int, fill: float = 0.0) -> np.ndarray:
    if values.shape[0] == size:
        return values
    padded = np.full((size,) + values.shape[1:], fill, dtype=values.dtype)
    padded[: values.shape[0]] = values
    return padded


def _to_numpy(values, dtype) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)
