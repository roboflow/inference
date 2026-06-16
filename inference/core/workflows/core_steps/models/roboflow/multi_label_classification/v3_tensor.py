"""Tensor-native sibling of `roboflow_core/roboflow_multi_label_classification_model@v3`.

Under ENABLE_TENSOR_DATA_REPRESENTATION this block emits a native
``inference_models.MultiLabelClassificationPrediction`` (torch tensors on
``WORKFLOWS_IMAGE_TENSOR_DEVICE``) under
``TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND`` instead of the standard
multi-label classification prediction dict.

- LOCAL: ``ModelManager.run_tensor_native_inference`` returns a
  ``List[MultiLabelClassificationPrediction]`` (one per image) straight from the
  adapter. Each carries ``class_ids`` (the already-threshold-filtered predicted
  label ids — the model's ``post_process`` applied the full priority chain, so we
  do NOT re-threshold / re-build them) and ``confidence`` (the FULL sigmoid score
  vector, shape ``(num_classes,)``). The block only ATTACHES the per-image
  ``image_metadata`` (SINGULAR dict — NOT plural; this is the key shape difference
  from multi-class ``ClassificationPrediction``) that the tensor serialiser needs:
  the ``class_id -> name`` map plus image dimensions and parent/root lineage.
- REMOTE: the standard inference multi-label response dict (a ``predictions`` dict
  keyed by class name with per-class ``confidence``/``class_id`` plus the
  ``predicted_classes`` name list) is rebuilt into a native
  ``MultiLabelClassificationPrediction`` by the inline converter below.
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

from inference_models.models.base.classification import (
    MultiLabelClassificationPrediction,
)

PREDICTION_TYPE = "classification"

LONG_DESCRIPTION = """
Run inference on a multi-label classification model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Multi-Label Classification Model",
            "version": "v3",
            "short_description": "Apply multiple tags to an image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 3,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_multi_label_classification_model@v3"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence_mode: Union[
        Literal["best", "default", "custom"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="best",
        description="How to determine the confidence threshold.",
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
        return ["multi-label-classification"]

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


class RoboflowMultiLabelClassificationModelBlockV3(WorkflowBlock):

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
        confidence: Union[None, float, Literal["best", "default"]],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        tensor_inputs = [img.tensor_image for img in images]
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        predictions: List[MultiLabelClassificationPrediction] = (
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
        results: List[dict] = []
        for image, prediction in zip(images, predictions):
            inference_id = str(uuid.uuid4())
            # The adapter/model already applied the full threshold chain when it
            # built `class_ids`, and `confidence` is the FULL sigmoid vector. We do
            # NOT re-threshold / re-softmax / rebuild `class_ids` — only attach the
            # (SINGULAR) image_metadata the tensor serialiser requires.
            # Pin the prediction tensors to WORKFLOWS_IMAGE_TENSOR_DEVICE so LOCAL
            # output matches the single-label sibling and this block's REMOTE path
            # (which builds them on-device via torch.as_tensor).
            prediction.class_ids = prediction.class_ids.to(WORKFLOWS_IMAGE_TENSOR_DEVICE)
            prediction.confidence = prediction.confidence.to(
                WORKFLOWS_IMAGE_TENSOR_DEVICE
            )
            prediction.image_metadata = _build_native_classification_metadata(
                image=image,
                class_names=class_names,
                inference_id=inference_id,
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
        confidence: Union[None, float, Literal["best", "default"]],
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
        return self._post_process_remote_result(
            images=images,
            predictions=predictions,
            model_id=model_id,
        )

    def _post_process_remote_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        model_id: str,
    ) -> BlockResult:
        results: List[dict] = []
        for image, response in zip(images, predictions):
            inference_id = response.get(INFERENCE_ID_KEY) or str(uuid.uuid4())
            prediction = _native_multi_label_from_inference_response(
                image=image,
                response=response,
                inference_id=inference_id,
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


def _build_native_classification_metadata(
    image: WorkflowImageData,
    class_names: Dict[int, str],
    inference_id: str,
) -> dict:
    """Build the (SINGULAR) ``image_metadata`` dict carried by a tensor-native
    ``MultiLabelClassificationPrediction``.

    Mirrors the ``build_native_image_metadata`` key conventions and the
    ``vlm_as_classifier/v1_tensor`` classification pattern: the ``class_id ->
    name`` map (required by the serialiser to resolve labels), the image
    dimensions, and the parent/root lineage. The image shape is read without
    forcing a device->host materialization so tensor-only inputs stay on device.
    """
    height, width = image._read_shape_without_materialization()
    return {
        CLASS_NAMES_KEY: class_names,
        PREDICTION_TYPE_KEY: PREDICTION_TYPE,
        IMAGE_DIMENSIONS_KEY: [height, width],
        INFERENCE_ID_KEY: inference_id,
        PARENT_ID_KEY: image.parent_metadata.parent_id,
        ROOT_PARENT_ID_KEY: image.workflow_root_ancestor_metadata.parent_id,
    }


def _native_multi_label_from_inference_response(
    image: WorkflowImageData,
    response: dict,
    inference_id: str,
) -> MultiLabelClassificationPrediction:
    """Rebuild a native ``MultiLabelClassificationPrediction`` from the standard
    inference multi-label response dict.

    The response ``predictions`` is a dict keyed by class name, each value holding
    ``{"confidence": float, "class_id": int}``; ``predicted_classes`` is the list
    of above-threshold class names. We reconstruct the dense (``num_classes``,)
    sigmoid ``confidence`` vector indexed by ``class_id``, the ``class_ids`` of the
    predicted (above-threshold) labels, and the ``class_id -> name`` map. The
    server already applied the threshold chain, so ``predicted_classes`` is taken
    as authoritative — we do NOT re-threshold here.
    """
    per_class = response.get("predictions", {}) or {}
    class_names: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    confidence_by_id: Dict[int, float] = {}
    for class_name, entry in per_class.items():
        class_id = int(entry["class_id"])
        class_names[class_id] = class_name
        name_to_id[class_name] = class_id
        confidence_by_id[class_id] = float(entry.get("confidence", 0.0))
    num_classes = (max(class_names) + 1) if class_names else 0
    confidence_vector = [confidence_by_id.get(i, 0.0) for i in range(num_classes)]
    predicted_class_ids = [
        name_to_id[class_name]
        for class_name in response.get("predicted_classes", [])
        if class_name in name_to_id
    ]
    return MultiLabelClassificationPrediction(
        class_ids=torch.as_tensor(
            predicted_class_ids,
            dtype=torch.long,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        confidence=torch.as_tensor(
            confidence_vector,
            dtype=torch.float32,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        image_metadata=_build_native_classification_metadata(
            image=image,
            class_names=class_names,
            inference_id=inference_id,
        ),
    )
