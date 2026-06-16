"""Tensor-native sibling of `roboflow_core/roboflow_classification_model@v3`.

Under ENABLE_TENSOR_DATA_REPRESENTATION this block emits native
``inference_models.ClassificationPrediction`` objects (torch tensors on
``WORKFLOWS_IMAGE_TENSOR_DEVICE``) under
``TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND`` instead of the legacy
classification response dict.

- LOCAL: ``ModelManager.run_tensor_native_inference`` returns ONE batched
  ``ClassificationPrediction`` (``class_id`` shape ``(bs,)``, ``confidence`` shape
  ``(bs, num_classes)`` full softmax). The consumer indexes per-image, so the
  block fans the batched object out into ``bs`` single-row predictions, each
  carrying the producer contract in ``images_metadata`` (PLURAL) that the
  tensor classification serialiser requires (mirrors
  ``formatters/vlm_as_classifier/v1_tensor.py``).
- REMOTE: standard inference classification response dicts are rebuilt into a
  native ``ClassificationPrediction`` via an inline converter (never the legacy
  response dict).
"""

import uuid
from typing import Dict, List, Literal, Optional, Type, Union

import torch
from pydantic import ConfigDict, Field, model_validator

from inference.core.env import (
    HOSTED_CLASSIFICATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
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

from inference_models.models.base.classification import ClassificationPrediction

PREDICTION_TYPE = "classification"

# Numpy parity: the adapter `postprocess` (inference_models_adapters.py) drops
# classes whose score < confidence_threshold, falling back to 0.5 when the
# resolved confidence is non-numeric (e.g. the string "default"). The native
# LOCAL path bypasses `postprocess`, so we (a) resolve the string mode to this
# numeric default BEFORE the native model call and (b) carry the resolved float
# in `image_metadata` for `serialise_native_classification` to apply the cutoff.
_DEFAULT_CONFIDENCE_THRESHOLD = 0.5
CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY = "classification_confidence_threshold"


def _resolve_confidence_threshold(confidence: Optional[Union[float, str]]) -> float:
    if isinstance(confidence, (int, float)):
        return float(confidence)
    return _DEFAULT_CONFIDENCE_THRESHOLD

LONG_DESCRIPTION = """
Run inference on a multi-class classification model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Single-Label Classification Model",
            "version": "v3",
            "short_description": "Apply a single tag to an image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 2,
                "inference": True,
                "popular": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_classification_model@v3"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    # Single-label classification opts out of per-class refinement — top-1
    # drives the response, so the "best" (F1-optimal) mode from model eval has
    # no meaningful effect here and is omitted.
    confidence_mode: Union[
        Literal["default", "custom"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="default",
        description="How to determine the confidence threshold.",
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
                "confidence_mode": {
                    "values": ["custom"],
                    "required": True,
                },
            },
        },
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
    def validate_custom_confidence(self) -> "BlockManifest":
        if self.confidence_mode == "custom" and self.custom_confidence is None:
            raise ValueError(
                "custom_confidence must be provided when confidence_mode is 'custom'"
            )
        return self

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        return ["classification"]

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
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[INFERENCE_ID_KIND]),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowClassificationModelBlockV3(WorkflowBlock):

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
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        confidence = (
            custom_confidence if confidence_mode == "custom" else confidence_mode
        )
        # Resolve the string mode ("default") to a concrete float BEFORE dispatch
        # so the native model `__call__` (which bypasses the adapter `postprocess`
        # string->0.5 fallback) never receives a non-numeric confidence.
        confidence = _resolve_confidence_threshold(confidence)
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                confidence=confidence,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                confidence=confidence,
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
        confidence: Union[None, float, Literal["default"]],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        tensor_inputs = [img.tensor_image for img in images]
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        # Single-label returns ONE batched ClassificationPrediction (NOT a list):
        #   class_id   -> (bs,)
        #   confidence -> (bs, num_classes)  full softmax
        batched_prediction: ClassificationPrediction = (
            self._model_manager.run_tensor_native_inference(
                model_id=model_id,
                images=tensor_inputs,
                input_color_format="rgb",
                confidence=confidence,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        )
        class_names = _class_names_map(self._model_manager.get_class_names(model_id))
        confidence_threshold = _resolve_confidence_threshold(confidence)
        results: List[dict] = []
        for index, image in enumerate(images):
            inference_id = str(uuid.uuid4())
            # Fan the batched object out into a single-row (bs=1) prediction the
            # consumer can index per-image. confidence stays the FULL per-class
            # softmax vector (do NOT collapse to a scalar).
            prediction = _single_row_prediction(
                batched_prediction=batched_prediction,
                index=index,
                image=image,
                class_names=class_names,
                inference_id=inference_id,
                confidence_threshold=confidence_threshold,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": prediction,
                    "model_id": model_id,
                }
            )
        return results

    def run_remotely(
        self,
        images: Batch[Optional[WorkflowImageData]],
        model_id: str,
        confidence: Union[None, float, Literal["default"]],
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
            confidence_threshold=confidence,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)
        non_empty_inference_images = [i.base64_image for i in images]
        predictions = client.infer(
            inference_input=non_empty_inference_images,
            model_id=model_id,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        confidence_threshold = _resolve_confidence_threshold(confidence)
        results: List[dict] = []
        for image, response in zip(images, predictions):
            inference_id = response.get(INFERENCE_ID_KEY) or str(uuid.uuid4())
            prediction = _native_classification_from_inference_response(
                image=image,
                response=response,
                inference_id=inference_id,
                confidence_threshold=confidence_threshold,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": prediction,
                    "model_id": model_id,
                }
            )
        return results


def _class_names_map(class_names: List[str]) -> Dict[int, str]:
    return {index: name for index, name in enumerate(class_names)}


def _build_image_metadata(
    image: WorkflowImageData,
    class_names: Dict[int, str],
    inference_id: str,
    confidence_threshold: float,
) -> dict:
    height, width = image._read_shape_without_materialization()
    return {
        CLASS_NAMES_KEY: class_names,
        PREDICTION_TYPE_KEY: PREDICTION_TYPE,
        IMAGE_DIMENSIONS_KEY: [height, width],
        INFERENCE_ID_KEY: inference_id,
        PARENT_ID_KEY: image.parent_metadata.parent_id,
        ROOT_PARENT_ID_KEY: image.workflow_root_ancestor_metadata.parent_id,
        # C2: resolved float threshold so `serialise_native_classification`
        # drops sub-threshold classes from `predictions`, matching numpy.
        CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY: confidence_threshold,
    }


def _single_row_prediction(
    batched_prediction: ClassificationPrediction,
    index: int,
    image: WorkflowImageData,
    class_names: Dict[int, str],
    inference_id: str,
    confidence_threshold: float,
) -> ClassificationPrediction:
    image_metadata = _build_image_metadata(
        image=image,
        class_names=class_names,
        inference_id=inference_id,
        confidence_threshold=confidence_threshold,
    )
    # Slice (not index) to keep the leading batch dim: class_id -> (1,),
    # confidence -> (1, num_classes). The confidence row stays the FULL softmax.
    return ClassificationPrediction(
        class_id=batched_prediction.class_id[index : index + 1].to(
            WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=batched_prediction.confidence[index : index + 1].to(
            WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        images_metadata=[image_metadata],
    )


def _native_classification_from_inference_response(
    image: WorkflowImageData,
    response: dict,
    inference_id: str,
    confidence_threshold: float,
) -> ClassificationPrediction:
    """Rebuild a native single-row ``ClassificationPrediction`` from a standard
    inference classification response dict.

    The remote response carries ``predictions`` as a list of
    ``{"class_id", "class", "confidence"}`` (already confidence-filtered/sorted)
    plus a ``top`` class. We reconstruct a dense confidence vector indexed by
    ``class_id`` and a ``class_id -> name`` map from those entries; classes not
    in the response default to 0.0 confidence (the model package's full class
    list is not available on this path).
    """
    detection_dicts = response.get("predictions", []) or []
    class_names: Dict[int, str] = {}
    confidence_by_id: Dict[int, float] = {}
    for entry in detection_dicts:
        class_id = int(entry["class_id"])
        class_names[class_id] = str(entry.get("class", class_id))
        confidence_by_id[class_id] = float(entry.get("confidence", 0.0))
    num_classes = (max(class_names.keys()) + 1) if class_names else 0
    # Fill gaps so the dense vector and the class_names map agree on every index.
    for class_id in range(num_classes):
        class_names.setdefault(class_id, str(class_id))
    confidence_vector = [confidence_by_id.get(i, 0.0) for i in range(num_classes)]
    top_class = response.get("top")
    top_class_id = next(
        (cid for cid, name in class_names.items() if name == top_class),
        int(max(confidence_by_id, key=confidence_by_id.get)) if confidence_by_id else 0,
    )
    image_metadata = _build_image_metadata(
        image=image,
        class_names=class_names,
        inference_id=inference_id,
        confidence_threshold=confidence_threshold,
    )
    return ClassificationPrediction(
        class_id=torch.tensor(
            [top_class_id], dtype=torch.long, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=torch.tensor(
            [confidence_vector],
            dtype=torch.float32,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        images_metadata=[image_metadata],
    )
