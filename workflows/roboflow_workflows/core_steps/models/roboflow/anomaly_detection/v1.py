from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.common.anomaly_detection import (
    ANOMALY_MAP_KEY,
    ANOMALY_SCORE_KEY,
    ANOMALY_THRESHOLD_KEY,
    IS_ANOMALOUS_KEY,
    coerce_anomaly_map,
    require_anomaly_response,
)
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.environment import (
    HOSTED_CLASSIFICATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_KEY_TRANSPORT,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from roboflow_workflows.execution_engine.constants import INFERENCE_ID_KEY
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    NUMPY_ARRAY_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
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

from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Run an anomaly detection model (PatchCore or FoundAD) trained on an anomaly
detection project hosted on Roboflow.

An anomaly model learns what *normal* looks like from normal images only and
scores a new image by how far it departs from that. There is no class
confidence: the model speaks a raw score, a threshold and a verdict. The block
reports:

- `anomaly_score`: the raw image score (unbounded, larger means more anomalous).
  Its scale depends on the trained model; it is not a probability.
- `anomaly_threshold`: the decision threshold in effect for this image.
- `is_anomalous`: `anomaly_score >= anomaly_threshold`.
- `anomaly_map`: the local anomaly evidence as a float32 `(height, width)`
  array at the network's input resolution (square), in the same units as
  `anomaly_score`. The map covers the whole image as the network saw it, so
  stretching it to the image size overlays it. Connect it to the **Anomaly
  Heatmap Visualization** block to draw it. Set `include_anomaly_map` to
  `False` to skip the map when only the verdict is needed.

The threshold saved with the model was fitted on validation images during
training. `anomaly_threshold` lets you tune the operating point without
retraining: lower it to catch more defects at the cost of more false alarms,
raise it to do the opposite. The value is sent to the inference server, which
decides `is_anomalous` against it and reports it back as `anomaly_threshold`.
Leave it empty to use the saved threshold.

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
    anomaly_threshold: Union[Optional[float], Selector(kind=[FLOAT_KIND])] = Field(
        default=None,
        description="Decision threshold on the raw anomaly score, replacing the one saved with the model. The server decides `is_anomalous` against it. Lower it to flag more images as anomalous. Leave empty to use the saved threshold.",
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
            OutputDefinition(name=ANOMALY_SCORE_KEY, kind=[FLOAT_KIND]),
            OutputDefinition(name=ANOMALY_THRESHOLD_KEY, kind=[FLOAT_KIND]),
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
        anomaly_threshold: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                include_anomaly_map=include_anomaly_map,
                anomaly_threshold=anomaly_threshold,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                include_anomaly_map=include_anomaly_map,
                anomaly_threshold=anomaly_threshold,
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
        anomaly_threshold: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )
        # The classification request schema needs a confidence although an anomaly
        # model never reads it: the verdict is the raw score against the threshold.
        predictions = self._model_manager.run_classification(
            model_id=model_id,
            images=inference_images,
            api_key=self._api_key,
            confidence="default",
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            include_anomaly_map=include_anomaly_map,
            anomaly_threshold=anomaly_threshold,
        )
        return self._post_process_result(
            predictions=predictions,
            images=images,
            model_id=model_id,
            include_anomaly_map=include_anomaly_map,
        )

    def run_remotely(
        self,
        images: Batch[Optional[WorkflowImageData]],
        model_id: str,
        include_anomaly_map: bool,
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
            anomaly_threshold=anomaly_threshold,
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
        return self._post_process_result(
            predictions=predictions,
            images=images,
            model_id=model_id,
            include_anomaly_map=include_anomaly_map,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        model_id: str,
        include_anomaly_map: bool,
    ) -> BlockResult:
        if len(predictions) != len(images):
            raise ValueError(
                f"Model {model_id!r} returned {len(predictions)} response(s) for "
                f"{len(images)} image(s)."
            )
        results: List[Dict[str, Any]] = []
        for prediction in predictions:
            require_anomaly_response(
                prediction, model_id=model_id, include_anomaly_map=include_anomaly_map
            )
            anomaly_map: Optional[np.ndarray] = coerce_anomaly_map(
                prediction.get(ANOMALY_MAP_KEY)
            )
            results.append(
                {
                    ANOMALY_SCORE_KEY: float(prediction[ANOMALY_SCORE_KEY]),
                    ANOMALY_THRESHOLD_KEY: float(prediction[ANOMALY_THRESHOLD_KEY]),
                    IS_ANOMALOUS_KEY: bool(prediction[IS_ANOMALOUS_KEY]),
                    ANOMALY_MAP_KEY: anomaly_map,
                    INFERENCE_ID_KEY: prediction.get(INFERENCE_ID_KEY),
                    "model_id": model_id,
                }
            )
        return results
