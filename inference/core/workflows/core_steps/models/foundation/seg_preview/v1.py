from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Type, Union

import numpy as np
import requests
from pydantic import ConfigDict, Field

from inference.core.roboflow_api import build_roboflow_api_headers

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Point,
)
from inference.core.env import (
    API_BASE_URL,
    ROBOFLOW_INTERNAL_SERVICE_NAME,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

DETECTIONS_CLASS_NAME_FIELD = "class_name"
DETECTION_ID_FIELD = "detection_id"


LONG_DESCRIPTION = "Seg Preview"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Seg Preview",
            "version": "v1",
            "short_description": "Seg Preview",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["Seg Preview"],
            "ui_manifest": {
                "section": "model",
                "icon": "fa-solid fa-eye",
                "blockPriority": 9.49,
                "needsGPU": True,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/seg-preview@v1"]

    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    class_names: Union[
        List[str], str, Selector(kind=[LIST_OF_VALUES_KIND, STRING_KIND])
    ] = Field(
        title="Class Names",
        default=None,
        description="List of classes to recognise",
        examples=[["car", "person"], "$inputs.classes"],
    )
    threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.5, description="Threshold for predicted mask scores", examples=[0.3]
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "boxes"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class SegPreviewBlockV1(WorkflowBlock):

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
        class_names: Optional[Union[List[str], str]],
        threshold: float,
    ) -> BlockResult:

        if isinstance(class_names, str):
            class_names = class_names.split(",")
        elif isinstance(class_names, list):
            class_names = class_names
        else:
            raise ValueError(f"Invalid class names type: {type(class_names)}")

        return self.run_via_request(
            images=images,
            class_names=class_names,
            threshold=threshold,
        )

    def run_via_request(
        self,
        images: Batch[WorkflowImageData],
        class_names: Optional[List[str]],
        threshold: float,
    ) -> BlockResult:
        predictions = []
        if class_names is None:
            class_names = []
        if len(class_names) == 0:
            class_names.append(None)

        endpoint = f"{API_BASE_URL}/inferenceproxy/seg-preview"
        api_key = self._api_key

        for single_image in images:
            prompt_class_ids: List[Optional[int]] = []
            prompt_class_names: List[Optional[str]] = []
            prompt_detection_ids: List[Optional[str]] = []

            # Build unified prompt list payloads for HTTP
            http_prompts: List[dict] = []
            for class_name in class_names:
                http_prompts.append({"type": "text", "text": class_name})

            # Prepare image for remote API (base64)
            http_image = {"type": "base64", "value": single_image.base64_image}

            payload = {
                "image": http_image,
                "prompts": http_prompts,
                "output_prob_thresh": threshold,
            }

            try:
                headers = {"Content-Type": "application/json"}
                if ROBOFLOW_INTERNAL_SERVICE_NAME:
                    headers["X-Roboflow-Internal-Service-Name"] = (
                        ROBOFLOW_INTERNAL_SERVICE_NAME
                    )
                if ROBOFLOW_INTERNAL_SERVICE_SECRET:
                    headers["X-Roboflow-Internal-Service-Secret"] = (
                        ROBOFLOW_INTERNAL_SERVICE_SECRET
                    )

                headers = build_roboflow_api_headers(explicit_headers=headers)

                response = requests.post(
                    f"{endpoint}?api_key={api_key}",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                response.raise_for_status()
                resp_json = response.json()
            except Exception:
                resp_json = {"prompt_results": []}

            class_predictions: List[InstanceSegmentationPrediction] = []
            for prompt_result in resp_json.get("prompt_results", []):
                idx = prompt_result.get("prompt_index", 0)
                class_name = class_names[idx] if idx < len(class_names) else None
                raw_predictions = prompt_result.get("predictions", [])
                # Adapt JSON dicts to objects with attribute-style access
                adapted_predictions = [SimpleNamespace(**p) for p in raw_predictions]
                class_pred = convert_segmentation_response_to_inference_instances_seg_response(
                    segmentation_predictions=adapted_predictions,  # type: ignore[arg-type]
                    image=single_image,
                    prompt_class_ids=prompt_class_ids,
                    prompt_class_names=prompt_class_names,
                    prompt_detection_ids=prompt_detection_ids,
                    threshold=threshold,
                    text_prompt=class_name,
                    specific_class_id=idx,
                )
                class_predictions.extend(class_pred.predictions)

            image_width = single_image.numpy_image.shape[1]
            image_height = single_image.numpy_image.shape[0]
            final_inference_prediction = InstanceSegmentationInferenceResponse(
                predictions=class_predictions,
                image=InferenceResponseImage(width=image_width, height=image_height),
            )
            predictions.append(final_inference_prediction)

        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="instance-segmentation",
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [{"predictions": prediction} for prediction in predictions]


def convert_segmentation_response_to_inference_instances_seg_response(
    segmentation_predictions: List[Any],
    image: WorkflowImageData,
    prompt_class_ids: List[Optional[int]],
    prompt_class_names: List[Optional[str]],
    prompt_detection_ids: List[Optional[str]],
    threshold: float,
    text_prompt: Optional[str] = None,
    specific_class_id: Optional[int] = None,
) -> InstanceSegmentationInferenceResponse:
    image_width = image.numpy_image.shape[1]
    image_height = image.numpy_image.shape[0]
    predictions = []
    if len(prompt_class_ids) == 0:
        prompt_class_ids = [
            specific_class_id if specific_class_id else 0
            for _ in range(len(segmentation_predictions))
        ]
        prompt_class_names = [
            text_prompt if text_prompt else "foreground"
            for _ in range(len(segmentation_predictions))
        ]
        prompt_detection_ids = [None for _ in range(len(segmentation_predictions))]
    for prediction, class_id, class_name, detection_id in zip(
        segmentation_predictions,
        prompt_class_ids,
        prompt_class_names,
        prompt_detection_ids,
    ):
        for mask in prediction.masks:
            if len(mask) < 3:
                # skipping empty masks
                continue
            if prediction.confidence < threshold:
                # skipping masks below threshold
                continue
            x_coords = [coord[0] for coord in mask]
            y_coords = [coord[1] for coord in mask]
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            predictions.append(
                InstanceSegmentationPrediction(
                    **{
                        "x": center_x,
                        "y": center_y,
                        "width": max_x - min_x,
                        "height": max_y - min_y,
                        "points": [Point(x=point[0], y=point[1]) for point in mask],
                        "confidence": prediction.confidence,
                        "class": class_name,
                        "class_id": class_id,
                        "parent_id": detection_id,
                    }
                )
            )
    return InstanceSegmentationInferenceResponse(
        predictions=predictions,
        image=InferenceResponseImage(width=image_width, height=image_height),
    )
