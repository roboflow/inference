from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Deque, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, PositiveInt, model_validator

from inference.core.entities.requests.inference import (
    InstanceSegmentationInferenceRequest,
)
from inference.core.entities.responses.inference import (
    InstanceSegmentationInferenceResponseDC,
    _is_response_dc_to_dict,
)
from inference.core.env import (
    HOSTED_INSTANCE_SEGMENTATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    filter_out_unwanted_classes_from_sv_detections_batch,
)
from inference.core.workflows.execution_engine.constants import INFERENCE_ID_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
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
from inference_models.configuration import get_rfdetr_pipeline_depth
from inference_models.models.base.async_handoff import get_async_response_future
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


@dataclass(frozen=True)
class _StreamPredictionContext:
    images: Batch[WorkflowImageData]
    class_filter: Optional[List[str]]
    model_id: str


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Instance Segmentation Model",
            "version": "v3",
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
    type: Literal["roboflow_core/roboflow_instance_segmentation_model@v3"]
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
    enforce_dense_masks_in_inference_models: Union[
        bool, Selector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=True,
        description="Boolean flag to enforce dense masks when inference models backend is in use "
        "(irrelevant in other cases). Dense masks are faster to process, but require more memory. "
        "Users can't tweak this flag when running on Roboflow serverless platform.",
        examples=[True, "$inputs.enforce_dense_masks_in_inference_models"],
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
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowInstanceSegmentationModelBlockV3(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self._last_model_id: Optional[str] = None
        self._stream_response_executor: Optional[ThreadPoolExecutor] = None
        self._pending_stream_prediction_contexts: Deque[_StreamPredictionContext] = (
            deque()
        )

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
        enforce_dense_masks_in_inference_models: bool,
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
                enforce_dense_masks_in_inference_models=enforce_dense_masks_in_inference_models,
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
        enforce_dense_masks_in_inference_models: bool,
    ) -> BlockResult:
        inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
        self._last_model_id = model_id
        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )
        stream_context = _StreamPredictionContext(
            images=images,
            class_filter=class_filter,
            model_id=model_id,
        )
        if self.stream_pipeline_depth() > 0:
            self._pending_stream_prediction_contexts.append(stream_context)
        request = InstanceSegmentationInferenceRequest(
            api_key=self._api_key,
            model_id=model_id,
            image=inference_images,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            mask_decode_mode=mask_decode_mode,
            tradeoff_factor=tradeoff_factor,
            source="workflow-execution",
            enforce_dense_masks_in_inference_models=enforce_dense_masks_in_inference_models,
        )
        predictions = self._model_manager.infer_from_request_sync(
            model_id=model_id, request=request
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        async_response_future = self._extract_async_response_future(
            predictions=predictions
        )
        if async_response_future is not None:
            stream_context = self._pop_stream_prediction_context(default=stream_context)
            return self._submit_async_post_process_result(
                predictions_future=async_response_future,
                stream_context=stream_context,
            )
        return self._finalize_prediction_responses(
            predictions=predictions,
            stream_context=stream_context,
        )

    def _extract_async_response_future(
        self,
        predictions: List[object],
    ) -> Optional[Future]:
        for prediction in predictions:
            async_response_future = get_async_response_future(prediction)
            if isinstance(async_response_future, Future):
                return async_response_future
        return None

    def _get_stream_response_executor(self) -> ThreadPoolExecutor:
        if self._stream_response_executor is None:
            self._stream_response_executor = ThreadPoolExecutor(max_workers=1)
        return self._stream_response_executor

    def _submit_async_post_process_result(
        self,
        predictions_future: Future,
        stream_context: _StreamPredictionContext,
    ) -> BlockResult:
        finalized_result_future = self._get_stream_response_executor().submit(
            self._finalize_async_prediction_value,
            predictions_future,
            stream_context,
        )
        return [
            {
                "inference_id": None,
                "predictions": self._submit_async_prediction_selector(
                    result_future=finalized_result_future,
                    image_index=image_index,
                ),
                "model_id": stream_context.model_id,
            }
            for image_index in range(len(stream_context.images))
        ]

    def _submit_async_prediction_selector(
        self,
        result_future: Future,
        image_index: int,
    ) -> Future:
        return self._get_stream_response_executor().submit(
            self._select_async_prediction_value,
            result_future,
            image_index,
        )

    def _finalize_async_prediction_value(
        self,
        predictions_future: Future,
        stream_context: _StreamPredictionContext,
    ) -> BlockResult:
        predictions = predictions_future.result()
        if not isinstance(predictions, list):
            predictions = [predictions]
        return self._finalize_prediction_responses(
            predictions=predictions,
            stream_context=stream_context,
        )

    def _finalize_prediction_responses(
        self,
        predictions: List[object],
        stream_context: _StreamPredictionContext,
    ) -> BlockResult:
        # The adapter returns dataclass responses when source="workflow-execution"
        # (cheaper construct + dict-walk than pydantic). Any other response type
        # (e.g. if a non-rfdetr backend is bound to the same block) falls back
        # to `model_dump`.
        predictions = [
            (
                _is_response_dc_to_dict(e)
                if isinstance(e, InstanceSegmentationInferenceResponseDC)
                else e.model_dump(by_alias=True, exclude_none=True)
            )
            for e in predictions
        ]
        return self._post_process_result(
            images=stream_context.images,
            predictions=predictions,
            class_filter=stream_context.class_filter,
            model_id=stream_context.model_id,
        )

    def _pop_stream_prediction_context(
        self,
        default: _StreamPredictionContext,
    ) -> _StreamPredictionContext:
        if self._pending_stream_prediction_contexts:
            return self._pending_stream_prediction_contexts.popleft()
        return default

    def _select_async_prediction_value(
        self,
        result_future: Future,
        image_index: int,
    ):
        result = result_future.result()
        if image_index >= len(result):
            return []
        return result[image_index]["predictions"]

    def is_stream_pipelined(self) -> bool:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            return False
        if (
            self._last_model_id is None
            or self._last_model_id not in self._model_manager
        ):
            return False
        model = self._model_manager[self._last_model_id]
        return (
            callable(getattr(model, "flush", None))
            and getattr(model, "_pipeline_depth", 1) > 1
        )

    def can_activate_stream_pipeline(self) -> bool:
        return (
            self._step_execution_mode is StepExecutionMode.LOCAL
            and get_rfdetr_pipeline_depth() > 1
        )

    def stream_pipeline_depth(self) -> int:
        if not self.is_stream_pipelined():
            return 0
        model = self._model_manager[self._last_model_id]
        return max(0, int(getattr(model, "_pipeline_depth", 1)) - 1)

    def flush_stream_pipeline(self) -> List[BlockResult]:
        if (
            self._last_model_id is None
            or self._last_model_id not in self._model_manager
        ):
            self._pending_stream_prediction_contexts.clear()
            return []
        model = self._model_manager[self._last_model_id]
        flush_fn = getattr(model, "flush", None)
        if not callable(flush_fn):
            self._pending_stream_prediction_contexts.clear()
            return []
        predictions = flush_fn()
        if not isinstance(predictions, list):
            predictions = [predictions]

        results = []
        offset = 0
        while self._pending_stream_prediction_contexts:
            stream_context = self._pending_stream_prediction_contexts.popleft()
            batch_size = len(stream_context.images)
            prediction_batch = predictions[offset : offset + batch_size]
            offset += batch_size
            if len(prediction_batch) != batch_size:
                raise RuntimeError(
                    "Stream pipeline flush returned fewer predictions than expected"
                )
            results.append(
                self._finalize_prediction_responses(
                    predictions=prediction_batch,
                    stream_context=stream_context,
                )
            )
        if offset != len(predictions):
            raise RuntimeError(
                "Stream pipeline flush returned more predictions than expected"
            )
        return results

    def close_stream_pipeline(self) -> None:
        if self._stream_response_executor is not None:
            self._stream_response_executor.shutdown(wait=False)
            self._stream_response_executor = None
        if (
            self._last_model_id is None
            or self._last_model_id not in self._model_manager
        ):
            return None
        model = self._model_manager[self._last_model_id]
        shutdown_fn = getattr(model, "shutdown_pipeline", None)
        if callable(shutdown_fn):
            shutdown_fn()

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
        return self._post_process_result(
            images=images,
            predictions=predictions,
            class_filter=class_filter,
            model_id=model_id,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        class_filter: Optional[List[str]],
        model_id: str,
    ) -> BlockResult:
        inference_ids = [p.get(INFERENCE_ID_KEY, None) for p in predictions]
        predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="instance-segmentation",
        )
        predictions = filter_out_unwanted_classes_from_sv_detections_batch(
            predictions=predictions,
            classes_to_accept=class_filter,
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [
            {
                "inference_id": inference_id,
                "predictions": prediction,
                "model_id": model_id,
            }
            for inference_id, prediction in zip(inference_ids, predictions)
        ]
