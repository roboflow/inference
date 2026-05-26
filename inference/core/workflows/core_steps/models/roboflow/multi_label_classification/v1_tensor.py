from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference_models.models.base.classification import MultiLabelClassificationPrediction

from inference.core.env import (
    HOSTED_CLASSIFICATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.remote_response_converters import (
    dict_response_to_multi_label_classification,
)
from inference.core.workflows.core_steps.common.tensor_prediction_metadata import (
    attach_prediction_metadata,
)
from inference.core.workflows.core_steps.common.to_supervision import (
    build_dual_multi_label_dict,
)

from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v2_tensor import (
    _harvest_class_names,
)
from inference.core.workflows.execution_engine.constants import INFERENCE_ID_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
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

LONG_DESCRIPTION = """
Run inference on a multi-label classification model hosted on or uploaded to Roboflow.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Multi-Label Classification Model",
            "version": "v1",
            "short_description": "Apply multiple tags to an image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal[
        "roboflow_core/roboflow_multi_label_classification_model@v1",
        "RoboflowMultiLabelClassificationModel",
        "MultiLabelClassificationModel",
    ]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(default=0.4)
    disable_active_learning: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(default=True)
    active_learning_target_dataset: Union[
        Selector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(default=None)

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        return ["multi-label-classification"]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowMultiLabelClassificationModelBlockV1(WorkflowBlock):

    def __init__(
        self, model_manager, api_key, step_execution_mode,
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
        self, images, model_id, confidence,
        disable_active_learning, active_learning_target_dataset,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images, model_id, confidence,
                disable_active_learning, active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images, model_id, confidence,
                disable_active_learning, active_learning_target_dataset,
            )
        raise ValueError(f"Unknown step execution mode: {self._step_execution_mode}")

    def run_locally(
        self, images, model_id, confidence,
        disable_active_learning, active_learning_target_dataset,
    ) -> BlockResult:
        tensor_inputs = [img.tensor_image for img in images]
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        predictions: List[MultiLabelClassificationPrediction] = (
            self._model_manager.run_tensor_native_inference(
                model_id=model_id, images=tensor_inputs, input_color_format="rgb",
                confidence=confidence,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        )
        class_names = dict(
            enumerate(self._model_manager.get_class_names(model_id=model_id))
        )
        results: BlockResult = []
        for image, prediction in zip(images, predictions):
            inference_id = attach_prediction_metadata(
                prediction, image=image, model_id=model_id,
                prediction_type="multi-label-classification",
                class_names=class_names,
            )
            results.append({
                "predictions": build_dual_multi_label_dict(prediction),
                "inference_id": inference_id,
            })
        return results

    def run_remotely(
        self, images, model_id, confidence,
        disable_active_learning, active_learning_target_dataset,
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CLASSIFICATION_URL
        )
        client = InferenceHTTPClient(api_url=api_url, api_key=self._api_key)
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        client_config = InferenceConfiguration(
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            confidence_threshold=confidence,
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)
        non_empty_inference_images = [i.base64_image for i in images]
        responses = client.infer(
            inference_input=non_empty_inference_images, model_id=model_id,
        )
        if not isinstance(responses, list):
            responses = [responses]
        predictions = [dict_response_to_multi_label_classification(r) for r in responses]
        class_names = _harvest_class_names(responses)
        results: BlockResult = []
        for image, prediction in zip(images, predictions):
            inference_id = attach_prediction_metadata(
                prediction, image=image, model_id=model_id,
                prediction_type="multi-label-classification",
                class_names=class_names or None,
            )
            results.append({
                "predictions": build_dual_multi_label_dict(prediction),
                "inference_id": inference_id,
            })
        return results
