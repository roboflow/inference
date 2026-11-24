"""Tensor-native sibling of `roboflow_core/anomaly_detection_model@v1`.

Under ENABLE_TENSOR_DATA_REPRESENTATION this block emits native
``inference_models.ClassificationPrediction`` objects under
``TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND`` instead of the legacy
classification response dict. ``images_metadata[0]`` carries the anomaly
verdict (``anomaly_score``, ``anomaly_threshold``, ``anomaly_scale``,
``is_anomalous``) next to
the usual producer contract, and ``serialise_native_classification`` writes
those keys into the flag-off dict shape.

- LOCAL: ``ModelsProvider.run_tensor_native_inference`` returns ONE batched
  ``ClassificationPrediction`` whose ``images_metadata[i]`` holds the per-image
  verdict and, when requested, the raw ``anomaly_map`` ndarray. The block fans
  it out into single-row predictions and moves the map to its own output.
- REMOTE: the ``AnomalyDetectionResponse`` dicts are rebuilt into native
  predictions; the map arrives decoded by ``inference_sdk`` or as the wire
  payload.
"""

import uuid
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
import torch
from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.common.anomaly_detection import (
    ANOMALOUS_CLASS_NAME,
    ANOMALY_DECISION_THRESHOLD_KEY,
    ANOMALY_MAP_KEY,
    ANOMALY_METADATA_KEYS,
    ANOMALY_SCALE_KEY,
    ANOMALY_SCORE_KEY,
    ANOMALY_THRESHOLD_KEY,
    IS_ANOMALOUS_KEY,
    NORMAL_CLASS_NAME,
    apply_threshold_override,
    coerce_anomaly_map,
    decision_threshold,
    redecide_at_threshold,
    require_anomaly_response,
    validate_threshold_override,
)
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.environment import (
    HOSTED_CLASSIFICATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_KEY_TRANSPORT,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from roboflow_workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    CLASSIFICATION_STYLE_KEY,
    CLASSIFICATION_STYLE_MODEL,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    NUMPY_ARRAY_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    FloatZeroToOne,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    roboflow_platform_model,
    roboflow_platform_project,
)
from roboflow_workflows.prototypes.models_provider import ModelsProvider

from inference_models.models.base.classification import ClassificationPrediction
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

PREDICTION_TYPE = "classification"

LONG_DESCRIPTION = """
Run an anomaly detection model (PatchCore or FoundAD) trained on an anomaly
detection project hosted on Roboflow.

An anomaly model learns what *normal* looks like from normal images only and
scores a new image by how far it departs from that. The block reports:

- `predictions`: the classification-style response, with `top` being `normal`
  or `anomalous` and the two class confidences. The anomalous confidence is
  `sigmoid((anomaly_score - anomaly_threshold) / scale)`, with the scale saved
  with the model, so it is exactly 0.5 at the threshold.
- `anomaly_score`: the raw image score (unbounded, larger means more anomalous).
- `anomaly_threshold`: the threshold in effect for this image (saved with the
  model, or the override).
- `is_anomalous`: whether the anomalous confidence reached `confidence`.
- `anomaly_decision_threshold`: the raw score at which the anomalous confidence
  equals `confidence` (`anomaly_threshold` itself at 0.5). Connect it to the
  heatmap's `anomaly_threshold` so the middle of its ramp marks the decision.
- `anomaly_map`: the local anomaly evidence as a float32 `(height, width)`
  array at the network's input resolution (square), in the same units as
  `anomaly_score`. The map covers the whole image as the network saw it, so
  stretching it to the image size overlays it. Connect it to the **Anomaly
  Heatmap Visualization** block to draw it. Set `include_anomaly_map` to
  `False` to skip the map when only the verdict is needed.

`confidence` sets the operating point without retraining: the default 0.5
reproduces the decision calibrated during training; lower it to catch more
defects at the cost of more false alarms, raise it to do the opposite.
`anomaly_threshold` replaces the saved threshold on the raw score; the class
confidences are then recomputed relative to it and `confidence` applies to them
the same way.

You will need to set your Roboflow API key in your Inference environment to use
this block. To learn more about setting your Roboflow API key, [refer to the
Inference documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Anomaly Detection Model",
            "version": "v1",
            "short_description": "Score an image against a model of normal images and locate what differs.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "anomaly",
                "anomaly detection",
                "defect",
                "PatchCore",
                "FoundAD",
                "heatmap",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-magnifying-glass-chart",
                "blockPriority": 2.5,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/anomaly_detection_model@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    include_anomaly_map: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Return the local anomaly map at the network input resolution. Disable it when only the verdict and score are needed.",
        examples=[True, "$inputs.include_anomaly_map"],
    )
    confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.5,
        description="Minimum anomalous confidence for an anomalous verdict. 0.5 reproduces the decision calibrated during training; lower it to flag more images as anomalous.",
        examples=[0.5, "$inputs.confidence"],
    )
    anomaly_threshold: Union[Optional[float], Selector(kind=[FLOAT_KIND])] = Field(
        default=None,
        description="Threshold on the raw anomaly score, replacing the one saved with the model; the anomalous confidence is recomputed relative to it. Leave empty to use the saved threshold.",
        examples=[1.25, "$inputs.anomaly_threshold"],
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
        # Anomaly models are registered under the classification task.
        return ["classification"]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        resources = [roboflow_platform_model(model_id=self.model_id)]
        if self.disable_active_learning is True:
            return resources
        if self.active_learning_target_dataset is not None:
            resources.append(
                roboflow_platform_project(
                    project_url=self.active_learning_target_dataset
                )
            )
        return resources

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND],
            ),
            OutputDefinition(name=ANOMALY_SCORE_KEY, kind=[FLOAT_KIND]),
            OutputDefinition(name=ANOMALY_THRESHOLD_KEY, kind=[FLOAT_KIND]),
            OutputDefinition(name=ANOMALY_DECISION_THRESHOLD_KEY, kind=[FLOAT_KIND]),
            OutputDefinition(name=IS_ANOMALOUS_KEY, kind=[BOOLEAN_KIND]),
            OutputDefinition(name=ANOMALY_MAP_KEY, kind=[NUMPY_ARRAY_KIND]),
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[INFERENCE_ID_KIND]),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.MODEL_INFERENCE]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


class AnomalyDetectionModelBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelsProvider,
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
        include_anomaly_map: bool,
        confidence: float,
        anomaly_threshold: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        threshold_override = validate_threshold_override(anomaly_threshold)
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                include_anomaly_map=include_anomaly_map,
                confidence=confidence,
                anomaly_threshold=threshold_override,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                include_anomaly_map=include_anomaly_map,
                confidence=confidence,
                anomaly_threshold=threshold_override,
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
        include_anomaly_map: bool,
        confidence: float,
        anomaly_threshold: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        # Feed the representation already materialised on the images: GPU tensors
        # (RGB) only when every image in the batch has one, else numpy (BGR).
        if all(image.is_tensor_materialised() for image in images):
            model_inputs = [image.tensor_image for image in images]
            image_color_format = "rgb"
        else:
            model_inputs = [image.numpy_image for image in images]
            image_color_format = "bgr"
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        inference_start = perf_counter()
        batched_prediction: ClassificationPrediction = (
            self._model_manager.run_tensor_native_inference(
                model_id=model_id,
                images=model_inputs,
                input_color_format=image_color_format,
                include_anomaly_map=include_anomaly_map,
                confidence=confidence,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        )
        inference_time = perf_counter() - inference_start
        class_names = _class_names_map(self._model_manager.get_class_names(model_id))
        model_metadata = batched_prediction.images_metadata or []
        if len(model_metadata) != len(images):
            raise ValueError(
                f"Model {model_id!r} did not return an anomaly detection prediction: "
                f"expected per-image metadata for {len(images)} images, got "
                f"{len(model_metadata)}. The Anomaly Detection Model block needs a "
                f"PatchCore or FoundAD model."
            )
        results: List[dict] = []
        for index, (image, image_model_metadata) in enumerate(
            zip(images, model_metadata)
        ):
            require_anomaly_response(image_model_metadata, model_id=model_id)
            verdict = {key: image_model_metadata[key] for key in ANOMALY_METADATA_KEYS}
            confidence_row = batched_prediction.confidence[index : index + 1]
            if anomaly_threshold is not None:
                anomalous = redecide_at_threshold(
                    verdict, anomaly_threshold, confidence
                )
                confidence_row = torch.tensor(
                    [[1 - anomalous, anomalous]], dtype=torch.float32
                )
            anomaly_map = coerce_anomaly_map(image_model_metadata.get(ANOMALY_MAP_KEY))
            inference_id = str(uuid.uuid4())
            image_metadata = _build_image_metadata(
                image=image,
                class_names=class_names,
                inference_id=inference_id,
                verdict=verdict,
                inference_time=inference_time,
            )
            prediction = ClassificationPrediction(
                class_id=_decided_class_id(verdict).to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
                confidence=confidence_row.to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
                images_metadata=[image_metadata],
            )
            results.append(
                _result(
                    prediction=prediction,
                    verdict=verdict,
                    confidence=confidence,
                    anomaly_map=anomaly_map,
                    inference_id=inference_id,
                    model_id=model_id,
                )
            )
        return results

    def run_remotely(
        self,
        images: Batch[Optional[WorkflowImageData]],
        model_id: str,
        include_anomaly_map: bool,
        confidence: float,
        anomaly_threshold: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CLASSIFICATION_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        client_config = InferenceConfiguration(
            api_key_transport=WORKFLOWS_REMOTE_API_KEY_TRANSPORT,
            include_anomaly_map=include_anomaly_map,
            confidence_threshold=confidence,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)
        non_empty_inference_images = [i.base64_image for i in images]
        responses = client.infer(
            inference_input=non_empty_inference_images,
            model_id=model_id,
        )
        if not isinstance(responses, list):
            responses = [responses]
        results: List[dict] = []
        for image, response in zip(images, responses):
            require_anomaly_response(response, model_id=model_id)
            anomaly_map = coerce_anomaly_map(response.pop(ANOMALY_MAP_KEY, None))
            response = apply_threshold_override(response, anomaly_threshold, confidence)
            verdict = {key: response[key] for key in ANOMALY_METADATA_KEYS}
            inference_id = response.get(INFERENCE_ID_KEY) or str(uuid.uuid4())
            prediction = _native_prediction_from_response(
                image=image,
                response=response,
                verdict=verdict,
                inference_id=inference_id,
            )
            results.append(
                _result(
                    prediction=prediction,
                    verdict=verdict,
                    confidence=confidence,
                    anomaly_map=anomaly_map,
                    inference_id=inference_id,
                    model_id=model_id,
                )
            )
        return results


def _class_names_map(class_names: List[str]) -> Dict[int, str]:
    return {index: name for index, name in enumerate(class_names)}


def _decided_class_id(verdict: Dict[str, Any]) -> torch.Tensor:
    # The two-class convention of the anomaly models: 0 is normal, 1 is anomalous.
    class_id = 1 if bool(verdict[IS_ANOMALOUS_KEY]) else 0
    return torch.tensor([class_id], dtype=torch.long)


def _result(
    prediction: ClassificationPrediction,
    verdict: Dict[str, Any],
    confidence: float,
    anomaly_map: Optional[np.ndarray],
    inference_id: str,
    model_id: str,
) -> dict:
    return {
        "predictions": prediction,
        ANOMALY_SCORE_KEY: float(verdict[ANOMALY_SCORE_KEY]),
        ANOMALY_THRESHOLD_KEY: float(verdict[ANOMALY_THRESHOLD_KEY]),
        ANOMALY_DECISION_THRESHOLD_KEY: decision_threshold(
            threshold=float(verdict[ANOMALY_THRESHOLD_KEY]),
            scale=float(verdict[ANOMALY_SCALE_KEY]),
            confidence=confidence,
        ),
        IS_ANOMALOUS_KEY: bool(verdict[IS_ANOMALOUS_KEY]),
        ANOMALY_MAP_KEY: anomaly_map,
        INFERENCE_ID_KEY: inference_id,
        "model_id": model_id,
    }


def _build_image_metadata(
    image: WorkflowImageData,
    class_names: Dict[int, str],
    inference_id: str,
    verdict: Dict[str, Any],
    inference_time: Optional[float] = None,
) -> dict:
    height, width = image._read_shape_without_materialization()
    metadata = {
        CLASS_NAMES_KEY: class_names,
        CLASSIFICATION_STYLE_KEY: CLASSIFICATION_STYLE_MODEL,
        PREDICTION_TYPE_KEY: PREDICTION_TYPE,
        IMAGE_DIMENSIONS_KEY: [height, width],
        INFERENCE_ID_KEY: inference_id,
        PARENT_ID_KEY: image.parent_metadata.parent_id,
        ROOT_PARENT_ID_KEY: image.workflow_root_ancestor_metadata.parent_id,
        # No classification confidence cutoff: an anomaly response always reports
        # both classes, whatever their confidence.
        ANOMALY_SCORE_KEY: float(verdict[ANOMALY_SCORE_KEY]),
        ANOMALY_THRESHOLD_KEY: float(verdict[ANOMALY_THRESHOLD_KEY]),
        ANOMALY_SCALE_KEY: float(verdict[ANOMALY_SCALE_KEY]),
        IS_ANOMALOUS_KEY: bool(verdict[IS_ANOMALOUS_KEY]),
    }
    if inference_time is not None:
        metadata["time"] = inference_time
    return metadata


def _native_prediction_from_response(
    image: WorkflowImageData,
    response: dict,
    verdict: Dict[str, Any],
    inference_id: str,
) -> ClassificationPrediction:
    """Rebuild a native single-row ``ClassificationPrediction`` from an
    ``AnomalyDetectionResponse`` dict. Both classes are always present in the
    response, so the dense confidence vector is complete."""
    class_names: Dict[int, str] = {}
    confidence_by_id: Dict[int, float] = {}
    for entry in response.get("predictions", []) or []:
        class_id = int(entry["class_id"])
        class_names[class_id] = str(entry.get("class", class_id))
        confidence_by_id[class_id] = float(entry.get("confidence", 0.0))
    num_classes = (max(class_names.keys()) + 1) if class_names else 2
    for class_id, default_name in enumerate((NORMAL_CLASS_NAME, ANOMALOUS_CLASS_NAME)):
        class_names.setdefault(class_id, default_name)
    for class_id in range(num_classes):
        class_names.setdefault(class_id, str(class_id))
    confidence_vector = [confidence_by_id.get(i, 0.0) for i in range(num_classes)]
    image_metadata = _build_image_metadata(
        image=image,
        class_names=class_names,
        inference_id=inference_id,
        verdict=verdict,
        inference_time=response.get("time"),
    )
    return ClassificationPrediction(
        class_id=_decided_class_id(verdict).to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        confidence=torch.tensor(
            [confidence_vector],
            dtype=torch.float32,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        images_metadata=[image_metadata],
    )
