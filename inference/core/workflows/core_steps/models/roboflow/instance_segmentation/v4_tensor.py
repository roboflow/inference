"""Tensor-native sibling of `roboflow_core/roboflow_instance_segmentation_model@v4`.

Under ENABLE_TENSOR_DATA_REPRESENTATION this block emits a native
``inference_models.InstanceDetections`` (torch tensors on
``WORKFLOWS_IMAGE_TENSOR_DEVICE``; masks dense or RLE) under
``TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND`` /
``TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND`` instead of
``sv.Detections``.

- LOCAL: ``ModelManager.run_tensor_native_inference`` returns
  ``List[InstanceDetections]`` straight from the adapter. The mask carrier (dense
  ``torch.Tensor`` vs ``InstancesRLEMasks``) is adapter-decided; both are handled
  downstream by the helpers and the tensor serialiser. The block applies
  ``class_filter`` natively (the adapter/model does NOT read it on this path) and
  attaches the producer contract (``image_metadata[class_names]`` + per-box
  ``detection_id``) the tensor serialiser requires, via
  ``attach_native_detection_metadata`` (it preserves masks + per-box metadata).
- REMOTE: standard inference instance-seg prediction dicts are rebuilt into a
  native ``InstanceDetections`` via the in-file
  ``_native_instance_detections_from_inference_predictions`` converter (never
  ``sv.Detections``). The block requests ``response_mask_format="rle"`` and carries
  RLE masks when the server honours it; it degrades gracefully to a dense mask
  rasterised from polygon ``points`` when the server ignores that parameter
  (older/alternative servers), matching the numpy flag-OFF polygon path.

This block creates ONLY this file; it reuses the already-registered tensor
serialiser for ``instance_segmentation_prediction`` /
``rle_instance_segmentation_prediction`` (``serialise_sv_detections`` already
handles ``InstanceDetections``, dense or RLE). The numpy sibling lives in
``.../instance_segmentation/v4.py``; this manifest is identical except the output
kinds.
"""

import uuid
from typing import Dict, List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
import torch
from pydantic import ConfigDict, Field, PositiveInt, model_validator

from inference.core.env import (
    HOSTED_INSTANCE_SEGMENTATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    attach_native_detection_metadata,
    build_native_image_metadata,
    take_prediction_by_mask,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_ID_KEY,
    CLASS_NAME_KEY,
    CONFIDENCE_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    INFERENCE_ID_KEY,
    POLYGON_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
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
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

PREDICTION_TYPE = "instance-segmentation"

# Keys carrying the RLE mask in standard inference instance-seg prediction dicts.
# `InstanceSegmentationRLEPrediction` declares the mask under `rle`; some response
# paths use `rle_mask` (mirrors the numpy converter's `d.get("rle_mask") or d.get("rle")`).
RLE_MASK_KEYS = ("rle_mask", "rle")

LONG_DESCRIPTION = """
Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).

This version of block introduces breaking change in behaviour of mask construction - it uses
`rle` format instead `polygon` making it possible to retrieve
shapes of any kind from remote server.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Instance Segmentation Model",
            "version": "v4",
            "short_description": "Predict the shape, size, and location of objects.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["yolo", "rfdetr", "rf-detr"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 1,
                "inference": True,
                "popular": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_instance_segmentation_model@v4"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence_mode: Union[
        Literal["best", "default", "custom"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="best",
        description="How confidence thresholds are determined.",
        json_schema_extra={
            "always_visible": True,
            "values_metadata": {
                "best": {
                    "name": "Best (Recommended)",
                    "description": "Use F1-optimal thresholds from model evaluation.",
                },
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
    class_filter: Union[Optional[List[str]], Selector(kind=[LIST_OF_VALUES_KIND])] = (
        Field(
            default=None,
            description="List of accepted classes. Classes must exist in the model's training set.",
            examples=[["a", "b", "c"], "$inputs.class_filter"],
        )
    )
    iou_threshold: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=300,
        description="Maximum number of detections to return.",
        examples=[300, "$inputs.max_detections"],
    )
    class_agnostic_nms: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to specify if NMS is to be used in class-agnostic mode.",
        examples=[True, "$inputs.class_agnostic_nms"],
    )
    max_candidates: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=3000,
        description="Maximum number of candidates as NMS input to be taken into account.",
        examples=[3000, "$inputs.max_candidates"],
    )
    mask_decode_mode: Union[
        Literal["accurate", "tradeoff", "fast"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="accurate",
        description="Parameter of mask decoding in prediction post-processing.",
        examples=["accurate", "$inputs.mask_decode_mode"],
    )
    tradeoff_factor: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.0,
        description="Post-processing parameter to dictate tradeoff between fast and accurate.",
        examples=[0.3, "$inputs.tradeoff_factor"],
    )
    disable_active_learning: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to disable project-level active learning for this block.",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        Selector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for active learning, if enabled.",
        examples=["my_project", "$inputs.al_target_project"],
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
        return ["instance-segmentation"]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[INFERENCE_ID_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[
                    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowInstanceSegmentationModelBlockV4(WorkflowBlock):

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
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        mask_decode_mode: Literal["accurate", "tradeoff", "fast"],
        tradeoff_factor: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        confidence = (
            custom_confidence if confidence_mode == "custom" else confidence_mode
        )
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                max_candidates=max_candidates,
                mask_decode_mode=mask_decode_mode,
                tradeoff_factor=tradeoff_factor,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                max_candidates=max_candidates,
                mask_decode_mode=mask_decode_mode,
                tradeoff_factor=tradeoff_factor,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Union[None, float, Literal["best", "default"]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        mask_decode_mode: Literal["accurate", "tradeoff", "fast"],
        tradeoff_factor: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        # Feed the representation already materialised on the images to avoid forcing a
        # numpy->device conversion: GPU tensors (RGB) only when every image in the batch
        # already has one, otherwise the numpy frames (BGR) — matching the numpy block.
        if all(image.is_tensor_materialised() for image in images):
            model_inputs = [image.tensor_image for image in images]
            image_color_format = "rgb"
        else:
            model_inputs = [image.numpy_image for image in images]
            image_color_format = "bgr"
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        detections_batch: List[InstanceDetections] = (
            self._model_manager.run_tensor_native_inference(
                model_id=model_id,
                images=model_inputs,
                input_color_format=image_color_format,
                confidence=confidence,
                iou_threshold=iou_threshold,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                max_detections=max_detections,
                max_candidates=max_candidates,
                mask_decode_mode=mask_decode_mode,
                tradeoff_factor=tradeoff_factor,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        )
        class_names = _class_names_map(self._model_manager.get_class_names(model_id))
        results: List[dict] = []
        for image, detections in zip(images, detections_batch):
            # The adapter/model does NOT honour class_filter on the native path, so
            # filter here (the only place it is applied for LOCAL execution).
            detections = _filter_classes_native(detections, class_filter, class_names)
            # Reuse the adapter-provided inference id when present (numpy parity:
            # numpy v4 surfaces ``p.get(INFERENCE_ID_KEY)`` off the model dump);
            # the tensor-native adapter normally attaches none, so fall back to a
            # freshly minted uuid that is then shared with ``image_metadata``.
            inference_id = getattr(detections, "inference_id", None) or str(
                uuid.uuid4()
            )
            detections = attach_native_detection_metadata(
                detections=detections,
                image=image,
                class_names=class_names,
                prediction_type=PREDICTION_TYPE,
                inference_id=inference_id,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": detections,
                    "model_id": model_id,
                }
            )
        return results

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Union[None, float, Literal["best", "default"]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        mask_decode_mode: Literal["accurate", "tradeoff", "fast"],
        tradeoff_factor: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_INSTANCE_SEGMENTATION_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        client_config = InferenceConfiguration(
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            confidence_threshold=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            mask_decode_mode=mask_decode_mode,
            tradeoff_factor=tradeoff_factor,
            response_mask_format="rle",
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
            images=images,
            predictions=predictions,
            class_filter=class_filter,
            model_id=model_id,
        )

    def _post_process_remote_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        class_filter: Optional[List[str]],
        model_id: str,
    ) -> BlockResult:
        # Fallback class_id -> name map from the model, used to name boxes whose
        # remote prediction dict lacks a `class` key (otherwise the tensor
        # serialiser hard-raises "class_id missing from mapping").
        model_class_names = _class_names_map(
            self._model_manager.get_class_names(model_id)
        )
        results: List[dict] = []
        for image, response in zip(images, predictions):
            inference_id = response.get(INFERENCE_ID_KEY) or str(uuid.uuid4())
            detection_dicts = response.get("predictions", []) or []
            if class_filter:
                accepted = set(class_filter)
                detection_dicts = [
                    d for d in detection_dicts if d.get(CLASS_NAME_KEY) in accepted
                ]
            detections = _native_instance_detections_from_inference_predictions(
                image=image,
                predictions=detection_dicts,
                prediction_type=PREDICTION_TYPE,
                inference_id=inference_id,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
                model_class_names=model_class_names,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": detections,
                    "model_id": model_id,
                }
            )
        return results


def _class_names_map(class_names: List[str]) -> Dict[int, str]:
    return {index: name for index, name in enumerate(class_names)}


def _filter_classes_native(
    detections: InstanceDetections,
    class_filter: Optional[List[str]],
    class_names: Dict[int, str],
) -> InstanceDetections:
    if not class_filter:
        return detections
    accepted = set(class_filter)
    accepted_ids = sorted(
        class_id for class_id, name in class_names.items() if name in accepted
    )
    if not accepted_ids:
        return take_prediction_by_mask(
            detections,
            torch.zeros_like(detections.class_id, dtype=torch.bool),
        )
    accepted_tensor = torch.as_tensor(accepted_ids, device=detections.class_id.device)
    keep = torch.isin(detections.class_id, accepted_tensor)
    return take_prediction_by_mask(detections, keep)


def _extract_rle_mask(prediction: dict) -> Optional[dict]:
    """Pull the COCO-RLE mask ({"size": [H, W], "counts": ...}) from a standard
    inference instance-seg prediction dict, trying both response key variants."""
    for key in RLE_MASK_KEYS:
        mask = prediction.get(key)
        if mask is not None:
            return mask
    return None


def _extract_polygon_points(prediction: dict) -> Optional[List[dict]]:
    """Pull the polygon ``points`` ([{"x": .., "y": ..}, ...]) from a standard
    inference instance-seg prediction dict that carries a polygon mask instead of
    RLE. Returns ``None`` when the key is absent or the polygon is degenerate
    (< 3 points) - mirroring supervision's ``Detections.from_inference`` / the numpy
    ``filter_out_invalid_polygons``, which drop such instances entirely."""
    points = prediction.get(POLYGON_KEY)
    if points is not None and len(points) >= 3:
        return points
    return None


def _polygon_points_to_dense_mask(
    points: List[dict], height: int, width: int
) -> np.ndarray:
    """Rasterise polygon ``points`` into a dense boolean ``(H, W)`` mask,
    byte-identically to supervision's ``polygon_to_mask`` as used by
    ``Detections.from_inference`` (the numpy REMOTE path): vertices are integer
    *truncated* (``dtype=int``, NOT rounded), filled with ``cv2.fillPoly`` onto a
    zero ``uint8`` canvas, then cast to ``bool``. Keeping the exact rasterisation
    means the serialiser's mask->polygon re-contouring (``mask_to_polygon``, shared
    verbatim with the numpy serialiser) reproduces the numpy ``points`` output."""
    polygon = np.array([[point[X_KEY], point[Y_KEY]] for point in points], dtype=int)
    return sv.polygon_to_mask(polygon, resolution_wh=(width, height)).astype(bool)


def _native_instance_detections_from_inference_predictions(
    image: WorkflowImageData,
    predictions: List[dict],
    prediction_type: str,
    inference_id: Optional[str] = None,
    device: Optional[torch.device] = None,
    model_class_names: Optional[Dict[int, str]] = None,
) -> InstanceDetections:
    """REMOTE-path converter: build a native ``InstanceDetections`` from standard
    inference instance-segmentation prediction dicts.

    The block requests ``response_mask_format="rle"`` so a current inference server
    returns an RLE-encoded COCO mask (``{"size": [H, W], "counts": "<utf-8>"}``)
    under ``rle`` / ``rle_mask``; those are carried as ``InstancesRLEMasks`` (counts
    normalised from a utf-8 string to bytes, what ``pycocotools`` / the serialiser's
    RLE decode path expect). To stay compatible with servers that ignore that
    request parameter and return polygon ``points`` instead (older/alternative
    inference servers), the converter degrades gracefully: when the response is not
    fully RLE-backed it rebuilds the masks from the polygon ``points`` into a dense
    ``torch.bool`` ``(N, H, W)`` carrier - the same masks the numpy flag-OFF REMOTE
    path builds via ``sv.Detections.from_inference`` - so the serialised ``points``
    output matches numpy's (the serialiser re-contours the dense mask with the same
    ``mask_to_polygon``). Boxes are converted from center form to corner ``xyxy``.
    The ``class_id -> name`` map (required by the tensor serialiser) and per-box
    ``detection_id`` are built here too.

    Degenerate (< 3-point) polygons are dropped exactly like the numpy path
    (``filter_out_invalid_polygons`` / supervision's ``from_inference``). When a
    prediction omits its ``class`` key, the class name is backfilled from
    ``model_class_names`` (the model's ``get_class_names`` map) so the tensor
    serialiser does not hard-raise on an unmapped ``class_id``.
    """
    height, width = image._read_shape_without_materialization()
    # Prefer the RLE masks the block requests; fall back to dense polygon masks when
    # the response is not fully RLE-backed (server ignored `response_mask_format`).
    # `all([])` is True, so an empty response keeps the empty-RLE carrier unchanged.
    use_rle = all(
        _extract_rle_mask(prediction) is not None for prediction in predictions
    )
    if use_rle:
        kept_predictions = predictions
    else:
        # Drop degenerate polygons up front so the surviving box arrays and masks
        # stay aligned (matches the numpy path's instance drop).
        kept_predictions = [
            prediction
            for prediction in predictions
            if _extract_polygon_points(prediction) is not None
        ]
    xyxy: List[List[float]] = []
    class_id: List[int] = []
    confidence: List[float] = []
    rle_counts: List[bytes] = []
    dense_masks: List[np.ndarray] = []
    bboxes_metadata: List[dict] = []
    derived_class_names: Dict[int, str] = {}
    for prediction in kept_predictions:
        center_x = float(prediction[X_KEY])
        center_y = float(prediction[Y_KEY])
        box_width = float(prediction[WIDTH_KEY])
        box_height = float(prediction[HEIGHT_KEY])
        xyxy.append(
            [
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            ]
        )
        prediction_class_id = int(prediction.get(CLASS_ID_KEY, 0))
        class_id.append(prediction_class_id)
        confidence.append(float(prediction.get(CONFIDENCE_KEY, 1.0)))
        if CLASS_NAME_KEY in prediction:
            derived_class_names[prediction_class_id] = str(prediction[CLASS_NAME_KEY])
        elif model_class_names is not None and prediction_class_id in model_class_names:
            derived_class_names[prediction_class_id] = model_class_names[
                prediction_class_id
            ]
        if use_rle:
            mask = _extract_rle_mask(prediction)
            raw_counts = mask["counts"]
            # Normalise to bytes: pycocotools (used by the serialiser's RLE decode)
            # expects byte counts; the remote rle response carries them as utf-8 strings.
            rle_counts.append(
                raw_counts.encode("utf-8")
                if isinstance(raw_counts, str)
                else raw_counts
            )
        else:
            dense_masks.append(
                _polygon_points_to_dense_mask(
                    points=_extract_polygon_points(prediction),
                    height=height,
                    width=width,
                )
            )
        bboxes_metadata.append(
            {DETECTION_ID_KEY: str(prediction.get(DETECTION_ID_KEY) or uuid.uuid4())}
        )
    number_of_detections = len(xyxy)
    image_metadata = build_native_image_metadata(
        image=image,
        class_names=derived_class_names,
        prediction_type=prediction_type,
        inference_id=inference_id,
    )
    if use_rle:
        mask_carrier: Union[torch.Tensor, InstancesRLEMasks] = InstancesRLEMasks(
            image_size=(height, width), masks=rle_counts
        )
    elif dense_masks:
        mask_carrier = torch.as_tensor(
            np.stack(dense_masks), dtype=torch.bool, device=device
        )
    else:
        # No surviving polygons: an empty dense carrier the serialiser handles.
        mask_carrier = torch.zeros((0, height, width), dtype=torch.bool, device=device)
    return InstanceDetections(
        xyxy=torch.as_tensor(xyxy, dtype=torch.float32, device=device).reshape(-1, 4),
        class_id=torch.as_tensor(class_id, dtype=torch.long, device=device).reshape(-1),
        confidence=torch.as_tensor(
            confidence, dtype=torch.float32, device=device
        ).reshape(-1),
        mask=mask_carrier,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata if number_of_detections > 0 else None,
    )
