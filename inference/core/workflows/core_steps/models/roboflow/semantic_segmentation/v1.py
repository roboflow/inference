import base64
from typing import Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

import cv2
import numpy as np
import supervision as sv
from pydantic import ConfigDict

from inference.core.entities.requests.inference import (
    SemanticSegmentationInferenceRequest,
)
from inference.core.env import (
    HOSTED_SEMANTIC_SEGMENTATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    INFERENCE_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
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
            "version": "v1",
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
    type: Literal["roboflow_core/roboflow_semantic_segmentation_model@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[INFERENCE_ID_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[SEMANTIC_SEGMENTATION_PREDICTION_KIND],
            ),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowSemanticSegmentationModelBlockV1(WorkflowBlock):

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
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(images=images, model_id=model_id)
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(images=images, model_id=model_id)
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
    ) -> BlockResult:
        inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
        request = SemanticSegmentationInferenceRequest(
            api_key=self._api_key,
            model_id=model_id,
            image=inference_images,
            source="workflow-execution",
        )
        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )
        predictions = self._model_manager.infer_from_request_sync(
            model_id=model_id, request=request
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(predictions=predictions, model_id=model_id)

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
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
        client_config = InferenceConfiguration(
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
        return self._post_process_result(predictions=predictions, model_id=model_id)

    def _post_process_result(
        self,
        predictions: List[dict],
        model_id: str,
    ) -> BlockResult:
        return [
            {
                INFERENCE_ID_KEY: prediction.get(INFERENCE_ID_KEY),
                "predictions": self._convert_to_sv_detections(
                    prediction.get("predictions") or {}
                ),
                "model_id": model_id,
            }
            for prediction in predictions
        ]

    @staticmethod
    def _convert_to_sv_detections(predictions_dict: Dict) -> sv.Detections:
        seg_mask_b64 = predictions_dict.get("segmentation_mask", "")
        conf_mask_b64 = predictions_dict.get("confidence_mask", "")
        class_map: Dict[str, str] = predictions_dict.get("class_map", {})

        mask_bytes = base64.b64decode(seg_mask_b64)
        nparr = np.frombuffer(mask_bytes, np.uint8)
        mask_array = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if mask_array is None:
            return sv.Detections.empty()

        unique_class_ids = [cid for cid in np.unique(mask_array).tolist() if cid != 0]
        if not unique_class_ids:
            return sv.Detections.empty()

        conf_array = None
        if conf_mask_b64:
            conf_bytes = base64.b64decode(conf_mask_b64)
            conf_nparr = np.frombuffer(conf_bytes, np.uint8)
            conf_array = cv2.imdecode(conf_nparr, cv2.IMREAD_GRAYSCALE)

        xyxy_list, masks_list, class_id_list, class_name_list, confidence_list = (
            [],
            [],
            [],
            [],
            [],
        )
        for class_id in unique_class_ids:
            binary_mask = mask_array == class_id
            rows = np.where(np.any(binary_mask, axis=1))[0]
            cols = np.where(np.any(binary_mask, axis=0))[0]
            xyxy_list.append([cols[0], rows[0], cols[-1], rows[-1]])
            masks_list.append(binary_mask)
            class_id_list.append(class_id)
            class_name_list.append(class_map.get(str(class_id), str(class_id)))
            if conf_array is not None:
                confidence_list.append(float(conf_array[binary_mask].mean()) / 255.0)
            else:
                confidence_list.append(1.0)

        detection_ids = np.array([str(uuid4()) for _ in class_id_list])
        result = sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float64),
            mask=np.array(masks_list, dtype=bool),
            class_id=np.array(class_id_list),
            confidence=np.array(confidence_list, dtype=np.float32),
            data={
                "class_name": np.array(class_name_list),
                DETECTION_ID_KEY: detection_ids,
            },
        )

        if conf_array is not None:
            result["confidence_mask"] = conf_array

        return result
