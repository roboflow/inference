"""Tensor-native sibling of ``roboflow_core/roboflow_keypoint_detection_model@v1``.

Under ENABLE_TENSOR_DATA_REPRESENTATION this block emits the native keypoint
prediction shape - a ``Tuple[inference_models.KeyPoints, inference_models.Detections]``
(torch tensors on ``WORKFLOWS_IMAGE_TENSOR_DEVICE``) - under
``TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND`` instead of ``sv.Detections``.

This is the v1 manifest (three accepted ``type`` literals; ``inference_id`` output is
``STRING_KIND``; no ``model_id`` output) wired to the v3_tensor run bodies. v1 carries a
plain ``confidence`` field (there is no ``confidence_mode`` / ``custom_confidence`` split
- that was introduced in v3), so ``confidence`` is forwarded directly.

Keypoint detection is unique among the tensor-native model blocks: the prediction
is a *2-tuple*. The producer contract that the tensor serialiser expects is carried
entirely on the **bbox** ``Detections`` component:

- ``attach_native_detection_metadata`` first attaches ``image_metadata`` (with the
  ``class_id -> name`` map) and a per-box ``detection_id`` to the bbox ``Detections``;
- then, for each instance ``i``, the per-instance keypoints are flattened into
  ``bboxes_metadata[i]`` under the four ``KEYPOINTS_*_KEY_IN_SV_DETECTIONS`` keys the
  serialiser reads. Keypoint class names are looked up via
  ``key_points_classes[object_class_id]`` (variable ``K`` per object class); keypoints
  with ``confidence <= 0.0`` are dropped (mirroring the numpy adapter's
  ``model_keypoints_to_response``).

The ``predictions`` output value is the full tuple ``(KeyPoints, Detections)`` so the
``KeyPoints`` component stays available to downstream tensor-native consumers; only the
serialiser unwraps the tuple back to the bbox ``Detections``.

- LOCAL: ``ModelManager.run_tensor_native_inference`` returns
  ``Tuple[List[KeyPoints], List[Detections]]`` from the adapter. ``class_filter`` is
  applied here natively (the adapter/model does NOT read it on this path) - the slice
  is applied to the tuple so the ``KeyPoints`` and bbox ``Detections`` stay aligned.
- REMOTE: standard keypoint inference prediction dicts are rebuilt into a native bbox
  ``Detections`` (never ``sv.Detections``); the per-detection ``keypoints`` lists are
  flattened straight into ``bboxes_metadata``, and a parallel ``KeyPoints`` component is
  reconstructed from the same dicts so the output keeps the tuple shape.
"""

import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from pydantic import ConfigDict, Field, PositiveInt

from inference.core.env import (
    HOSTED_DETECT_URL,
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
    native_detections_from_inference_predictions,
    take_prediction_by_mask,
)
from inference.core.workflows.execution_engine.constants import (
    CONFIDENCE_KEY,
    INFERENCE_ID_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
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
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

PREDICTION_TYPE = "keypoint-detection"

# Keypoint-detection responses (both inference_models adapter and remote API) carry
# the per-detection keypoint list under this key.
KEYPOINTS_KEY = "keypoints"

LONG_DESCRIPTION = """
Run inference on a keypoint detection model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Keypoint Detection Model",
            "version": "v1",
            "short_description": "Predict skeletons on objects.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["yolo"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 4,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal[
        "roboflow_core/roboflow_keypoint_detection_model@v1",
        "RoboflowKeypointDetectionModel",
        "KeypointsDetectionModel",
    ]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions.",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    keypoint_confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.0,
        description="Confidence threshold to predict a keypoint as visible.",
        examples=[0.3, "$inputs.keypoint_confidence"],
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

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        return ["keypoint-detection"]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[STRING_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowKeypointDetectionModelBlockV1(WorkflowBlock):

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
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        keypoint_confidence: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
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
                keypoint_confidence=keypoint_confidence,
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
                keypoint_confidence=keypoint_confidence,
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
        confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        keypoint_confidence: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        tensor_inputs = [img.tensor_image for img in images]
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        keypoints_batch: List[KeyPoints]
        detections_batch: Optional[List[Detections]]
        keypoints_batch, detections_batch = (
            self._model_manager.run_tensor_native_inference(
                model_id=model_id,
                images=tensor_inputs,
                input_color_format="rgb",
                confidence=confidence,
                iou_threshold=iou_threshold,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                max_detections=max_detections,
                max_candidates=max_candidates,
                # The adapter's `map_inference_kwargs` only sets `key_points_threshold`
                # from `kwargs["request"].keypoint_confidence`, which is absent on the
                # tensor-native path - so pass it explicitly here.
                key_points_threshold=keypoint_confidence,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        )
        if detections_batch is None:
            raise RuntimeError(
                "Keypoint detection model did not return the bounding-box component "
                "required by the tensor-native keypoint prediction. Models adapted from "
                "the `inference_models` package must provide instance detections."
            )
        class_names = _class_names_map(self._model_manager.get_class_names(model_id))
        # `key_points_classes` (List[List[str]], indexed by *object* class id) is only
        # exposed on the inference_models adapter - reach the adapter directly through
        # the manager's item access (the same handle that backs `get_class_names`).
        key_points_classes = self._model_manager[model_id].key_points_classes
        results: List[dict] = []
        for image, key_points, detections in zip(
            images, keypoints_batch, detections_batch
        ):
            # The adapter/model does NOT honour class_filter on the native path, so
            # filter here (the only place it is applied for LOCAL execution). The mask
            # slices the tuple consistently across KeyPoints + bbox Detections.
            key_points, detections = _filter_classes_native(
                key_points=key_points,
                detections=detections,
                class_filter=class_filter,
                class_names=class_names,
            )
            inference_id = str(uuid.uuid4())
            detections = attach_native_detection_metadata(
                detections=detections,
                image=image,
                class_names=class_names,
                prediction_type=PREDICTION_TYPE,
                inference_id=inference_id,
            )
            _attach_keypoints_to_bboxes_metadata(
                detections=detections,
                key_points=key_points,
                key_points_classes=key_points_classes,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": (key_points, detections),
                }
            )
        return results

    def run_remotely(
        self,
        images: Batch[Optional[WorkflowImageData]],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        keypoint_confidence: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_DETECT_URL
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
            keypoint_confidence_threshold=keypoint_confidence,
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
        )

    def _post_process_remote_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        class_filter: Optional[List[str]],
    ) -> BlockResult:
        results: List[dict] = []
        for image, response in zip(images, predictions):
            inference_id = response.get(INFERENCE_ID_KEY) or str(uuid.uuid4())
            detection_dicts = response.get("predictions", [])
            if class_filter:
                detection_dicts = [
                    d for d in detection_dicts if d.get("class") in class_filter
                ]
            detections = native_detections_from_inference_predictions(
                image=image,
                predictions=detection_dicts,
                prediction_type=PREDICTION_TYPE,
                inference_id=inference_id,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            )
            # The remote response already carries per-detection keypoint dicts
            # (class_id / class / confidence / x / y) - flatten them straight onto the
            # bbox Detections metadata and rebuild a parallel KeyPoints component so the
            # output keeps the native tuple shape.
            _attach_remote_keypoints_to_bboxes_metadata(
                detections=detections,
                detection_dicts=detection_dicts,
            )
            key_points = _native_key_points_from_inference_predictions(
                detection_dicts=detection_dicts,
                image_metadata=detections.image_metadata,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": (key_points, detections),
                }
            )
        return results


def _class_names_map(class_names: List[str]) -> Dict[int, str]:
    return {index: name for index, name in enumerate(class_names)}


def _filter_classes_native(
    key_points: KeyPoints,
    detections: Detections,
    class_filter: Optional[List[str]],
    class_names: Dict[int, str],
) -> Tuple[KeyPoints, Detections]:
    if not class_filter:
        return key_points, detections
    accepted = set(class_filter)
    keep = [
        class_names.get(int(class_id)) in accepted
        for class_id in detections.class_id.tolist()
    ]
    # take_prediction_by_mask slices the (KeyPoints, Detections) tuple consistently.
    return take_prediction_by_mask((key_points, detections), keep)


def _attach_keypoints_to_bboxes_metadata(
    detections: Detections,
    key_points: KeyPoints,
    key_points_classes: List[List[str]],
) -> None:
    """Flatten each instance's keypoints into the bbox ``Detections`` metadata under
    the four keys the tensor serialiser reads. ``key_points_classes`` is indexed by
    *object* class id (``K`` varies per class); keypoints with ``confidence <= 0.0``
    are dropped, mirroring the numpy adapter's ``model_keypoints_to_response``.

    Must run after ``attach_native_detection_metadata`` so ``bboxes_metadata`` exists.
    """
    bboxes_metadata = detections.bboxes_metadata
    if bboxes_metadata is None:
        return
    xy = key_points.xy.detach().cpu().tolist()
    confidence = key_points.confidence.detach().cpu().tolist()
    object_class_ids = [int(value) for value in key_points.class_id.detach().cpu().tolist()]
    for index, entry in enumerate(bboxes_metadata):
        instance_xy = xy[index]
        instance_confidence = confidence[index]
        object_class_id = object_class_ids[index]
        keypoint_class_names = key_points_classes[object_class_id]
        kept_class_id: List[int] = []
        kept_class_name: List[str] = []
        kept_confidence: List[float] = []
        kept_xy: List[List[float]] = []
        for keypoint_class_id, ((x, y), conf, keypoint_class_name) in enumerate(
            zip(instance_xy, instance_confidence, keypoint_class_names)
        ):
            if conf <= 0.0:
                continue
            kept_class_id.append(int(keypoint_class_id))
            kept_class_name.append(str(keypoint_class_name))
            kept_confidence.append(float(conf))
            kept_xy.append([float(x), float(y)])
        entry[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS] = kept_class_id
        entry[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS] = kept_class_name
        entry[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS] = kept_confidence
        entry[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = kept_xy


def _attach_remote_keypoints_to_bboxes_metadata(
    detections: Detections,
    detection_dicts: List[dict],
) -> None:
    """Flatten remote keypoint dicts (already keyed by class_id / class / confidence /
    x / y per detection) into the bbox ``Detections`` metadata. The remote API has
    already applied the keypoint confidence threshold, so no filtering is needed."""
    bboxes_metadata = detections.bboxes_metadata
    if bboxes_metadata is None:
        return
    for entry, detection_dict in zip(bboxes_metadata, detection_dicts):
        keypoints = detection_dict.get(KEYPOINTS_KEY, []) or []
        entry[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS] = [
            int(keypoint.get("class_id", keypoint_index))
            for keypoint_index, keypoint in enumerate(keypoints)
        ]
        entry[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS] = [
            str(keypoint.get("class", "")) for keypoint in keypoints
        ]
        entry[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS] = [
            float(keypoint.get(CONFIDENCE_KEY, 0.0)) for keypoint in keypoints
        ]
        entry[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = [
            [float(keypoint[X_KEY]), float(keypoint[Y_KEY])] for keypoint in keypoints
        ]


def _native_key_points_from_inference_predictions(
    detection_dicts: List[dict],
    image_metadata: Optional[dict],
    device: Optional[Any] = None,
) -> KeyPoints:
    """Rebuild a padded native ``KeyPoints`` from remote keypoint dicts so the REMOTE
    output keeps the native tuple shape. Padded to a uniform ``K`` (ragged keypoint
    counts across instances) with confidence 0.0 in the padding rows."""
    per_instance_xy: List[List[List[float]]] = []
    per_instance_confidence: List[List[float]] = []
    object_class_ids: List[int] = []
    for detection_dict in detection_dicts:
        keypoints = detection_dict.get(KEYPOINTS_KEY, []) or []
        per_instance_xy.append(
            [[float(keypoint[X_KEY]), float(keypoint[Y_KEY])] for keypoint in keypoints]
        )
        per_instance_confidence.append(
            [float(keypoint.get(CONFIDENCE_KEY, 0.0)) for keypoint in keypoints]
        )
        object_class_ids.append(int(detection_dict.get("class_id", 0)))
    number_of_instances = len(detection_dicts)
    max_key_points = max((len(xy) for xy in per_instance_xy), default=0)
    xy_tensor = torch.zeros(
        (number_of_instances, max_key_points, 2), dtype=torch.float32, device=device
    )
    confidence_tensor = torch.zeros(
        (number_of_instances, max_key_points), dtype=torch.float32, device=device
    )
    for index in range(number_of_instances):
        count = len(per_instance_xy[index])
        if count > 0:
            xy_tensor[index, :count] = torch.as_tensor(
                per_instance_xy[index], dtype=torch.float32, device=device
            )
            confidence_tensor[index, :count] = torch.as_tensor(
                per_instance_confidence[index], dtype=torch.float32, device=device
            )
    class_id_tensor = torch.as_tensor(
        object_class_ids, dtype=torch.long, device=device
    ).reshape(-1)
    return KeyPoints(
        xy=xy_tensor,
        class_id=class_id_tensor,
        confidence=confidence_tensor,
        image_metadata=image_metadata,
    )
