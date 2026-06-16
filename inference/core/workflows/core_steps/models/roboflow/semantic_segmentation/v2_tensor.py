"""Tensor-native sibling of `roboflow_core/roboflow_semantic_segmentation_model@v2`.

Numpy `semantic_segmentation/v2.py` produces a *dense* per-pixel class grid and
serialises it as an ``sv.Detections`` carrying one COCO-RLE mask per class (under
``data['rle_mask']``). Under ENABLE_TENSOR_DATA_REPRESENTATION this sibling must
instead emit a native ``inference_models`` prediction so the shared tensor
serialiser/consumers work unchanged.

DESIGN (Option B - per-class RLE detections, no dense carrier kind):
The block emits an ``inference_models.InstanceDetections`` (the same carrier the
instance-segmentation / seg_preview tensor blocks emit) with ONE instance per
non-background class id present in the segmentation map. Each instance carries:
  * ``xyxy`` - tight bbox enclosing all pixels of the class,
  * ``class_id`` - the class id,
  * ``confidence`` - mean per-pixel confidence over that class's pixels,
  * a COCO RLE in ``mask`` (``InstancesRLEMasks``), encoded one class at a time
    (``torch_mask_to_coco_rle`` / ``pycocotools``) so a giant ``N x H x W`` dense
    stack is never allocated.
This reuses the exact RLE approach of ``foundation/seg_preview/v1_tensor.py``.

Because the produced object is an ``InstanceDetections``, the existing tensor
serialiser ``serializers_tensor.py::serialise_sv_detections`` handles it
unchanged. (The serialiser turns each RLE instance into a polygon list - that is
the standard tensor-native instance-segmentation serialisation; see the loader
delta in this module's docstring tail.)

BACKGROUND / IGNORE (data-driven; NOT a hardcoded id 0):
Semantic-segmentation model packages are guaranteed (``inference_models``'
``validate_class_names``) to contain a ``background`` class by name, but it is NOT
guaranteed to be id 0 - its id is the index of ``background`` (case-insensitive)
in the model's class names / class_map. We therefore exclude:
  * the class id whose name is ``background`` (resolved from the class-name list
    for LOCAL, or from the response ``class_map`` for REMOTE), and
  * the ignore index ``255`` - the adapter wraps the ``-1`` "no class" sentinel to
    ``255`` and caps at 256 classes, so 255 may appear in the grid with no class
    name and must never become a detection.
This rule is fully data-driven (keyed off the model's own class names), so models
whose background is not at index 0 are handled correctly - unlike numpy
``v2.py``, which hardcodes ``cid != 0``.

LOADER DELTAS (return-only - do NOT apply here):

(a) serializer registration (``KINDS_SERIALIZERS``, loader.py ~1139) - the kind
    ``semantic_segmentation_prediction`` is currently mapped to the NUMPY sv
    serialiser ``serialise_rle_sv_detections``, which crashes on a native
    ``InstanceDetections``. Under the tensor flag it must point at the tensor
    serialiser ``serialise_sv_detections`` (already imported from
    ``serializers_tensor`` when ``ENABLE_TENSOR_DATA_REPRESENTATION`` is set):

        # loader.py KINDS_SERIALIZERS (inside the dict, ~line 1139)
        SEMANTIC_SEGMENTATION_PREDICTION_KIND.name: (
            serialise_sv_detections
            if ENABLE_TENSOR_DATA_REPRESENTATION
            else serialise_rle_sv_detections
        ),

(b) block import-swap (loader.py ~656, mirroring the object_detection@v3 swap at
    ~645) - import the tensor sibling under the flag:

        from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v1 import (
            RoboflowSemanticSegmentationModelBlockV1,
        )
        if ENABLE_TENSOR_DATA_REPRESENTATION:
            from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v2_tensor import (
                RoboflowSemanticSegmentationModelBlockV2,
            )
        else:
            from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v2 import (
                RoboflowSemanticSegmentationModelBlockV2,
            )
"""

import base64
import uuid
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import torch
from pydantic import ConfigDict, Field, model_validator

from inference.core.env import (
    HOSTED_SEMANTIC_SEGMENTATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    DETECTION_ID_KEY,
    INFERENCE_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_SEMANTIC_SEGMENTATION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

PREDICTION_TYPE = "semantic-segmentation"

# Conventional name (case-insensitive) of the class that must not become a
# detection. inference_models' `validate_class_names` guarantees it is present in
# every semantic-segmentation package, but NOT necessarily at id 0.
BACKGROUND_CLASS_NAME = "background"

# `-1` ("no class") sentinel wrapped by the adapter to 255 (it caps at 256
# classes). It can appear in the grid with no class name; never a detection.
IGNORE_CLASS_ID = 255


LONG_DESCRIPTION = """
Run inference on a semantic segmentation model hosted on or uploaded to Roboflow.

Semantic segmentation assigns a class label to every pixel in the image, producing a
dense segmentation mask rather than per-object bounding boxes or instance masks.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Semantic Segmentation Model",
            "version": "v2",
            "short_description": "Assign a class label to every pixel in the image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["semantic", "segmentation", "deeplab", "deep_lab"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-paint-brush",
                "blockPriority": 3,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_semantic_segmentation_model@v2"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    # TODO: add "best" option once model eval supports semantic segmentation.
    confidence_mode: Union[
        Literal["default", "custom"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="default",
        description="How confidence thresholds are determined.",
        json_schema_extra={
            "always_visible": True,
            "values_metadata": {
                "default": {
                    "name": "Default",
                    "description": "Use the model's built-in default threshold.",
                },
                "custom": {
                    "name": "Custom",
                    "description": "Specify a custom confidence threshold.",
                },
            },
        },
    )
    custom_confidence: Union[
        Optional[FloatZeroToOne],
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Custom confidence threshold for predictions.",
        examples=[0.3, "$inputs.confidence_threshold"],
        json_schema_extra={
            "relevant_for": {
                "confidence_mode": {"values": ["custom"], "required": True},
            },
        },
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if self.confidence_mode == "custom" and self.custom_confidence is None:
            raise ValueError(
                "`custom_confidence` is required when `confidence_mode` is 'custom'"
            )
        return self

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        return ["semantic-segmentation"]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[INFERENCE_ID_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[TENSOR_NATIVE_SEMANTIC_SEGMENTATION_PREDICTION_KIND],
            ),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowSemanticSegmentationModelBlockV2(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        confidence_mode: str,
        custom_confidence: Optional[float],
    ) -> BlockResult:
        confidence = (
            custom_confidence if confidence_mode == "custom" else confidence_mode
        )
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images, model_id=model_id, confidence=confidence
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images, model_id=model_id, confidence=confidence
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        confidence: Union[None, float, Literal["default"]],
    ) -> BlockResult:
        tensor_inputs = [img.tensor_image for img in images]
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        segmentation_results: List[SemanticSegmentationResult] = (
            self._model_manager.run_tensor_native_inference(
                model_id=model_id,
                images=tensor_inputs,
                input_color_format="rgb",
                confidence=confidence,
            )
        )
        # class_names is index-ordered (id -> name); same source as the adapter's
        # `class_map` ({str(idx): name}).
        class_names = list(self._model_manager.get_class_names(model_id))
        class_names_map = _class_names_map(class_names)
        excluded_ids = _excluded_class_ids(class_names_map)
        results: List[dict] = []
        for image, segmentation in zip(images, segmentation_results):
            inference_id = str(uuid.uuid4())
            detections = _build_instance_detections_from_segmentation(
                segmentation_map=segmentation.segmentation_map,
                confidence=segmentation.confidence,
                image=image,
                class_names_map=class_names_map,
                excluded_ids=excluded_ids,
                inference_id=inference_id,
            )
            results.append(
                {
                    INFERENCE_ID_KEY: inference_id,
                    "predictions": detections,
                    "model_id": model_id,
                }
            )
        return results

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        confidence: Union[None, float, Literal["default"]],
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_SEMANTIC_SEGMENTATION_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        client_config = InferenceConfiguration(
            confidence_threshold=confidence,
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)
        inference_images = [i.base64_image for i in images]
        predictions = client.infer(
            inference_input=inference_images,
            model_id=model_id,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        return self._post_process_remote_result(
            images=images, predictions=predictions, model_id=model_id
        )

    def _post_process_remote_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        model_id: str,
    ) -> BlockResult:
        results: List[dict] = []
        for image, prediction in zip(images, predictions):
            inference_id = prediction.get(INFERENCE_ID_KEY) or str(uuid.uuid4())
            detections = _build_instance_detections_from_inference_response(
                predictions_dict=prediction.get("predictions") or {},
                image=image,
                inference_id=inference_id,
            )
            results.append(
                {
                    INFERENCE_ID_KEY: inference_id,
                    "predictions": detections,
                    "model_id": model_id,
                }
            )
        return results


def _class_names_map(class_names: List[str]) -> Dict[int, str]:
    return {index: name for index, name in enumerate(class_names)}


def _excluded_class_ids(class_names_map: Dict[int, str]) -> set:
    """Data-driven set of class ids that must never become detections: the
    ``background`` class (by name, case-insensitive - NOT a hardcoded id 0) plus
    the ``255`` ignore/no-class sentinel."""
    excluded = {IGNORE_CLASS_ID}
    for class_id, class_name in class_names_map.items():
        if str(class_name).lower() == BACKGROUND_CLASS_NAME:
            excluded.add(int(class_id))
    return excluded


def _empty_instance_detections(
    image: WorkflowImageData,
    class_names_map: Dict[int, str],
    height: int,
    width: int,
    inference_id: str,
) -> InstanceDetections:
    detections = InstanceDetections(
        xyxy=torch.zeros(
            (0, 4), dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        class_id=torch.zeros(
            (0,), dtype=torch.int64, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=torch.zeros(
            (0,), dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        mask=InstancesRLEMasks(image_size=(height, width), masks=[]),
    )
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=class_names_map,
        prediction_type=PREDICTION_TYPE,
        inference_id=inference_id,
    )
    detections.bboxes_metadata = None
    return detections


def _assemble_instance_detections(
    xyxy: List[List[float]],
    class_ids: List[int],
    confidences: List[float],
    rle_dicts: List[dict],
    bboxes_metadata: List[dict],
    image: WorkflowImageData,
    class_names_map: Dict[int, str],
    height: int,
    width: int,
    inference_id: str,
) -> InstanceDetections:
    if len(rle_dicts) == 0:
        return _empty_instance_detections(
            image=image,
            class_names_map=class_names_map,
            height=height,
            width=width,
            inference_id=inference_id,
        )
    detections = InstanceDetections(
        xyxy=torch.tensor(
            xyxy, dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        class_id=torch.tensor(
            class_ids, dtype=torch.int64, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=torch.tensor(
            confidences, dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        mask=InstancesRLEMasks.from_coco_rle_masks(
            image_size=(height, width), masks=rle_dicts
        ),
    )
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=class_names_map,
        prediction_type=PREDICTION_TYPE,
        inference_id=inference_id,
    )
    detections.bboxes_metadata = bboxes_metadata
    return detections


def _build_instance_detections_from_segmentation(
    segmentation_map: torch.Tensor,
    confidence: torch.Tensor,
    image: WorkflowImageData,
    class_names_map: Dict[int, str],
    excluded_ids: set,
    inference_id: str,
) -> InstanceDetections:
    """LOCAL path: dense ``(H, W)`` class grid + ``(H, W)`` confidence ->
    one RLE instance per present non-background/non-ignore class.

    Masks are encoded one class at a time via ``torch_mask_to_coco_rle`` (the same
    compact COCO-RLE approach as ``seg_preview/v1_tensor.py``); no ``N x H x W``
    dense stack is ever materialised.
    """
    height, width = int(segmentation_map.shape[0]), int(segmentation_map.shape[1])
    xyxy: List[List[float]] = []
    class_ids: List[int] = []
    confidences: List[float] = []
    bboxes_metadata: List[dict] = []
    rle_dicts: List[dict] = []

    present_ids = [int(value) for value in torch.unique(segmentation_map).tolist()]
    for class_id in present_ids:
        if class_id in excluded_ids:
            continue
        class_name = class_names_map.get(class_id)
        if class_name is None:
            # id present in the grid but absent from the class map (e.g. an
            # unexpected sentinel) - cannot serialise a class name, so skip.
            continue
        binary_mask = segmentation_map == class_id
        bbox = _bbox_from_binary_mask(binary_mask)
        if bbox is None:
            continue
        # per-class mean confidence over that class's pixels
        class_confidence = float(confidence[binary_mask].mean().item())
        # single (H, W) bool/byte mask -> compact COCO RLE in C; no dense stack.
        rle = torch_mask_to_coco_rle(binary_mask.to(torch.uint8))
        rle = _normalise_rle_counts(rle)
        xyxy.append(bbox)
        class_ids.append(class_id)
        confidences.append(class_confidence)
        rle_dicts.append(rle)
        bboxes_metadata.append(
            {DETECTION_ID_KEY: str(uuid.uuid4()), CLASS_NAME_KEY: class_name}
        )

    return _assemble_instance_detections(
        xyxy=xyxy,
        class_ids=class_ids,
        confidences=confidences,
        rle_dicts=rle_dicts,
        bboxes_metadata=bboxes_metadata,
        image=image,
        class_names_map=class_names_map,
        height=height,
        width=width,
        inference_id=inference_id,
    )


def _build_instance_detections_from_inference_response(
    predictions_dict: Dict,
    image: WorkflowImageData,
    inference_id: str,
) -> InstanceDetections:
    """REMOTE path: standard inference semantic-seg response (a single ``dict``,
    matching numpy ``v2.py``'s ``_convert_to_sv_detections`` input) ->
    one RLE instance per present non-background/non-ignore class.

    Response shape (see numpy ``v2.py``):
      * ``segmentation_mask`` - base64 PNG, grayscale, pixel value == class id
      * ``confidence_mask``   - base64 PNG, grayscale, 0..255 (== confidence*255)
      * ``class_map``         - ``{str(class_id): class_name}``
    """
    seg_mask_b64 = predictions_dict.get("segmentation_mask", "")
    conf_mask_b64 = predictions_dict.get("confidence_mask", "")
    class_map: Dict[str, str] = predictions_dict.get("class_map", {})
    class_names_map = {int(k): v for k, v in class_map.items()}

    height, width = image._read_shape_without_materialization()

    mask_array = _decode_b64_grayscale_png(seg_mask_b64)
    if mask_array is None:
        return _empty_instance_detections(
            image=image,
            class_names_map=class_names_map,
            height=int(height),
            width=int(width),
            inference_id=inference_id,
        )
    height, width = int(mask_array.shape[0]), int(mask_array.shape[1])
    conf_array = _decode_b64_grayscale_png(conf_mask_b64)

    excluded_ids = _excluded_class_ids(class_names_map)
    xyxy: List[List[float]] = []
    class_ids: List[int] = []
    confidences: List[float] = []
    bboxes_metadata: List[dict] = []
    rle_dicts: List[dict] = []

    present_ids = [int(value) for value in np.unique(mask_array).tolist()]
    for class_id in present_ids:
        if class_id in excluded_ids:
            continue
        class_name = class_names_map.get(class_id, str(class_id))
        binary_mask = mask_array == class_id
        bbox = _bbox_from_numpy_mask(binary_mask)
        if bbox is None:
            continue
        if conf_array is not None:
            class_confidence = float(conf_array[binary_mask].mean()) / 255.0
        else:
            class_confidence = 1.0
        mask_tensor = torch.from_numpy(binary_mask.astype(np.uint8)).to(
            WORKFLOWS_IMAGE_TENSOR_DEVICE
        )
        rle = torch_mask_to_coco_rle(mask_tensor)
        rle = _normalise_rle_counts(rle)
        xyxy.append(bbox)
        class_ids.append(class_id)
        confidences.append(class_confidence)
        rle_dicts.append(rle)
        bboxes_metadata.append(
            {DETECTION_ID_KEY: str(uuid.uuid4()), CLASS_NAME_KEY: class_name}
        )

    return _assemble_instance_detections(
        xyxy=xyxy,
        class_ids=class_ids,
        confidences=confidences,
        rle_dicts=rle_dicts,
        bboxes_metadata=bboxes_metadata,
        image=image,
        class_names_map=class_names_map,
        height=height,
        width=width,
        inference_id=inference_id,
    )


def _bbox_from_binary_mask(binary_mask: torch.Tensor) -> Optional[List[float]]:
    """Tight ``[x_min, y_min, x_max, y_max]`` enclosing all True pixels of a
    ``(H, W)`` torch bool mask, or ``None`` if empty."""
    rows = torch.any(binary_mask, dim=1)
    cols = torch.any(binary_mask, dim=0)
    row_indices = torch.nonzero(rows, as_tuple=False).flatten()
    col_indices = torch.nonzero(cols, as_tuple=False).flatten()
    if row_indices.numel() == 0 or col_indices.numel() == 0:
        return None
    y_min = float(row_indices[0].item())
    y_max = float(row_indices[-1].item())
    x_min = float(col_indices[0].item())
    x_max = float(col_indices[-1].item())
    return [x_min, y_min, x_max, y_max]


def _bbox_from_numpy_mask(binary_mask: np.ndarray) -> Optional[List[float]]:
    """Tight ``[x_min, y_min, x_max, y_max]`` enclosing all True pixels of a
    ``(H, W)`` numpy bool mask, or ``None`` if empty (mirrors numpy ``v2.py``)."""
    rows = np.where(np.any(binary_mask, axis=1))[0]
    cols = np.where(np.any(binary_mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return [float(cols[0]), float(rows[0]), float(cols[-1]), float(rows[-1])]


def _decode_b64_grayscale_png(b64_value: str) -> Optional[np.ndarray]:
    if not b64_value:
        return None
    mask_bytes = base64.b64decode(b64_value)
    nparr = np.frombuffer(mask_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)


def _normalise_rle_counts(rle: dict) -> dict:
    """``InstancesRLEMasks.from_coco_rle_masks`` keeps the raw ``counts`` payload
    as-is; ``pycocotools`` returns it as ``bytes``. Decode to ``str`` so the
    serialised RLE matches the rest of the tensor pipeline (and the numpy block,
    which also stores ``counts`` as a utf-8 string)."""
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle = dict(rle)
        rle["counts"] = counts.decode("utf-8")
    return rle
