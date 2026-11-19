from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
)
from roboflow_workflows.environment import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_KEY_TRANSPORT,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    FloatZeroToOne,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    roboflow_platform_model,
)
from roboflow_workflows.prototypes.models_provider import ModelsProvider

from inference_sdk import InferenceConfiguration, InferenceHTTPClient


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OWLv2 Model",
            "version": "v1",
            "block_type": "model",
            "short_description": "Detect objects with uploaded OWLv2 weights and text queries.",
            "long_description": "Supply the uploaded model ID and one text query for each object class. Each output box uses the matching query as its class name.",
            "license": "Apache-2.0",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/owlv2_model@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    class_names: Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]] = Field(
        description="Text queries in output class ID order.",
        examples=[["bicycle", "bus"]],
    )
    confidence: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = 0.1

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        if ENABLE_TENSOR_DATA_REPRESENTATION:
            from roboflow_workflows.execution_engine.entities.tensor_native_types import (
                TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            )

            kind = TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND
        else:
            kind = OBJECT_DETECTION_PREDICTION_KIND
        return [OutputDefinition(name="predictions", kind=[kind])]

    def discover_dependent_resources(self) -> List[DependentResource]:
        return [roboflow_platform_model(model_id=self.model_id)]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.MODEL_INFERENCE]

    @classmethod
    def get_execution_engine_compatibility(cls) -> str:
        return ">=1.3.0,<2.0.0"


class OwlV2ModelBlockV1(WorkflowBlock):
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
        class_names: List[str],
        confidence: float,
    ) -> BlockResult:
        if self._step_execution_mode == StepExecutionMode.LOCAL:
            self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
            responses = self._model_manager.run_owlv2_text(
                model_id=model_id,
                images=[i.to_inference_format(numpy_preferred=True) for i in images],
                text=class_names,
                confidence=confidence,
                api_key=self._api_key,
            )
        else:
            client = InferenceHTTPClient(
                api_url=(
                    HOSTED_CORE_MODEL_URL
                    if WORKFLOWS_REMOTE_API_TARGET == "hosted"
                    else LOCAL_INFERENCE_API_URL
                ),
                api_key=self._api_key,
            )
            client.configure(
                InferenceConfiguration(
                    api_key_transport=WORKFLOWS_REMOTE_API_KEY_TRANSPORT,
                    max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
                )
            )
            responses = client.infer_from_owlv2(
                inference_input=[i.to_inference_format()["value"] for i in images],
                model_id=model_id,
                class_names=class_names,
                confidence=confidence,
            )
        if ENABLE_TENSOR_DATA_REPRESENTATION:
            from roboflow_workflows.core_steps.common.tensor_native import (
                native_detections_from_inference_predictions,
            )
            from roboflow_workflows.environment import WORKFLOWS_IMAGE_TENSOR_DEVICE

            names = dict(enumerate(class_names))
            return [
                {
                    "predictions": native_detections_from_inference_predictions(
                        image=image,
                        predictions=response["predictions"],
                        class_names=names,
                        prediction_type="object-detection",
                        device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
                    )
                }
                for image, response in zip(images, responses)
            ]
        predictions = convert_inference_detections_batch_to_sv_detections(responses)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions, prediction_type="object-detection"
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images, predictions=predictions
        )
        return [{"predictions": detection} for detection in predictions]
