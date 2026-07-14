"""Tensor-native `roboflow_core/florence_2@v1` block.

Florence-2 outputs (`raw_output` / `parsed_output` / `classes`) are text/dict/list,
not prediction kinds; the only tensor-native surface is the `grounding_detection`
input, which arrives in the `TensorNativeGrounding` shapes.
"""

import json
from typing import List, Optional, Tuple, Union

import torch

from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    split_key_point_prediction,
)
from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
    TASK_TYPE_TO_FLORENCE_TASK,
    TASKS_REQUIRING_DETECTION_GROUNDING,
    TASKS_TO_EXTRACT_LABELS_AS_CLASSES,
    BlockManifest,
    GroundingSelectionMode,
    TaskType,
    _coordinate_to_loc,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock
from inference_models import Detections, InstanceDetections, KeyPoints
from inference_sdk import InferenceHTTPClient

# inference_models native prediction shapes accepted on the grounding input.
TensorNativeGrounding = Union[
    Detections,  # OD
    InstanceDetections,  # IS
    Tuple[KeyPoints, Optional[Detections]],  # KP dual representation
]


class Florence2BlockV1(WorkflowBlock):

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
    def get_manifest(cls):
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        task_type: TaskType,
        prompt: Optional[str],
        classes: Optional[List[str]],
        grounding_detection: Optional[
            Union[Batch[TensorNativeGrounding], List[int], List[float]]
        ],
        grounding_selection_mode: GroundingSelectionMode,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                task_type=task_type,
                model_version=model_version,
                prompt=prompt,
                classes=classes,
                grounding_detection=grounding_detection,
                grounding_selection_mode=grounding_selection_mode,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                task_type=task_type,
                model_version=model_version,
                prompt=prompt,
                classes=classes,
                grounding_detection=grounding_detection,
                grounding_selection_mode=grounding_selection_mode,
            )
        raise ValueError(f"Unknown step execution mode: {self._step_execution_mode}")

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        task_type: TaskType,
        prompt: Optional[str],
        classes: Optional[List[str]],
        grounding_detection: Optional[
            Union[Batch[TensorNativeGrounding], List[int], List[float]]
        ],
        grounding_selection_mode: GroundingSelectionMode,
    ) -> BlockResult:
        requires_detection_grounding = task_type in TASKS_REQUIRING_DETECTION_GROUNDING
        is_not_florence_task = task_type == "custom"
        florence_task = TASK_TYPE_TO_FLORENCE_TASK[task_type]

        prompts = _build_prompts(
            images=images,
            classes=classes,
            base_prompt=prompt,
            grounding_detection=grounding_detection,
            grounding_selection_mode=grounding_selection_mode,
        )
        if classes is None:
            classes = []

        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image, single_prompt in zip(images, prompts):
            if single_prompt is None and requires_detection_grounding:
                predictions.append(
                    {"raw_output": None, "parsed_output": None, "classes": None}
                )
                continue
            if is_not_florence_task:
                final_prompt = single_prompt or ""
            else:
                final_prompt = florence_task + (single_prompt or "")

            # The adapter derives the Florence task from the prompt string.
            if image.is_tensor_materialised():
                model_image, image_color_format = image.tensor_image, "rgb"
            else:
                model_image, image_color_format = image.numpy_image, "bgr"
            result = self._model_manager.run_tensor_native_inference(
                model_version,
                images=[model_image],
                input_color_format=image_color_format,
                prompt=final_prompt,
            )
            prediction_data = result[0]

            extracted_classes = classes
            if florence_task in TASKS_TO_EXTRACT_LABELS_AS_CLASSES and isinstance(
                prediction_data, dict
            ):
                extracted_classes = prediction_data.get("labels", [])

            predictions.append(
                {
                    "raw_output": json.dumps(prediction_data),
                    "parsed_output": (
                        prediction_data if isinstance(prediction_data, dict) else None
                    ),
                    "classes": extracted_classes,
                }
            )
        return predictions

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        task_type: TaskType,
        prompt: Optional[str],
        classes: Optional[List[str]],
        grounding_detection: Optional[
            Union[Batch[TensorNativeGrounding], List[int], List[float]]
        ],
        grounding_selection_mode: GroundingSelectionMode,
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(api_url=api_url, api_key=self._api_key)
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()

        requires_detection_grounding = task_type in TASKS_REQUIRING_DETECTION_GROUNDING
        is_not_florence_task = task_type == "custom"
        florence_task = TASK_TYPE_TO_FLORENCE_TASK[task_type]

        prompts = _build_prompts(
            images=images,
            classes=classes,
            base_prompt=prompt,
            grounding_detection=grounding_detection,
            grounding_selection_mode=grounding_selection_mode,
        )
        if classes is None:
            classes = []

        predictions = []
        for image, single_prompt in zip(images, prompts):
            if single_prompt is None and requires_detection_grounding:
                predictions.append(
                    {"raw_output": None, "parsed_output": None, "classes": None}
                )
                continue
            if is_not_florence_task:
                final_prompt = single_prompt or ""
            else:
                final_prompt = florence_task + (single_prompt or "")

            result = client.infer_lmm(
                inference_input=image.base64_image,
                model_id=model_version,
                prompt=final_prompt,
                model_id_in_path=True,
            )
            response = result.get("response", {})
            if is_not_florence_task:
                if isinstance(response, dict) and len(response) > 0:
                    prediction_data = response[list(response.keys())[0]]
                else:
                    prediction_data = response
            else:
                prediction_data = response.get(florence_task, response)

            extracted_classes = classes
            if florence_task in TASKS_TO_EXTRACT_LABELS_AS_CLASSES and isinstance(
                prediction_data, dict
            ):
                extracted_classes = prediction_data.get("labels", [])

            predictions.append(
                {
                    "raw_output": json.dumps(prediction_data),
                    "parsed_output": (
                        prediction_data if isinstance(prediction_data, dict) else None
                    ),
                    "classes": extracted_classes,
                }
            )
        return predictions


def _build_prompts(
    images: Batch[WorkflowImageData],
    classes: Optional[List[str]],
    base_prompt: Optional[str],
    grounding_detection,
    grounding_selection_mode: GroundingSelectionMode,
) -> List[Optional[str]]:
    if grounding_detection is not None:
        return prepare_detection_grounding_prompts(
            images=images,
            grounding_detection=grounding_detection,
            grounding_selection_mode=grounding_selection_mode,
        )
    if classes is not None:
        return ["<and>".join(classes)] * len(images)
    return [base_prompt] * len(images)


def prepare_detection_grounding_prompts(
    images: Batch[WorkflowImageData],
    grounding_detection,
    grounding_selection_mode: GroundingSelectionMode,
) -> List[Optional[str]]:
    if isinstance(grounding_detection, list):
        return [
            _location_prompt_from_static_box(
                image=image, bounding_box=grounding_detection
            )
            for image in images
        ]
    return [
        _location_prompt_from_tensor_detections(
            image=image,
            prediction=prediction,
            grounding_selection_mode=grounding_selection_mode,
        )
        for image, prediction in zip(images, grounding_detection)
    ]


def _location_prompt_from_tensor_detections(
    image: WorkflowImageData,
    prediction,
    grounding_selection_mode: GroundingSelectionMode,
) -> Optional[str]:
    # Pull the bbox-bearing Detections out of OD / IS / (KeyPoints, Detections).
    _, detections = split_key_point_prediction(prediction)
    if len(detections) == 0:
        return None
    height, width = image._read_shape_without_materialization()
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _select_box(
        detections=detections, mode=grounding_selection_mode
    )
    return (
        f"<loc_{_coordinate_to_loc(left_top_x / width)}>"
        f"<loc_{_coordinate_to_loc(left_top_y / height)}>"
        f"<loc_{_coordinate_to_loc(right_bottom_x / width)}>"
        f"<loc_{_coordinate_to_loc(right_bottom_y / height)}>"
    )


def _select_box(detections, mode: GroundingSelectionMode) -> List[float]:
    xyxy = detections.xyxy
    if mode == "first":
        index = 0
    elif mode == "last":
        index = len(detections) - 1
    elif mode in ("biggest", "smallest"):
        area = (xyxy[:, 2] - xyxy[:, 0]) * (
            xyxy[:, 3] - xyxy[:, 1]
        )  # no .area on inference_models
        index = int(torch.argmax(area) if mode == "biggest" else torch.argmin(area))
    elif mode in ("most-confident", "least-confident"):
        confidence = detections.confidence
        index = int(
            torch.argmax(confidence)
            if mode == "most-confident"
            else torch.argmin(confidence)
        )
    else:
        raise ValueError(f"Unknown grounding selection mode: {mode}")
    return xyxy[index].detach().to("cpu").tolist()


def _location_prompt_from_static_box(
    image: WorkflowImageData,
    bounding_box: Union[List[float], List[int]],
) -> str:
    height, width = image._read_shape_without_materialization()
    coordinates = bounding_box[:4]
    if len(coordinates) != 4:
        raise ValueError(
            "Could not extract 4 coordinates of bounding box to perform detection "
            "grounded Florence 2 prediction."
        )
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = coordinates
    if all(isinstance(c, float) for c in coordinates):
        return (
            f"<loc_{_coordinate_to_loc(left_top_x)}>"
            f"<loc_{_coordinate_to_loc(left_top_y)}>"
            f"<loc_{_coordinate_to_loc(right_bottom_x)}>"
            f"<loc_{_coordinate_to_loc(right_bottom_y)}>"
        )
    if all(isinstance(c, int) for c in coordinates):
        return (
            f"<loc_{_coordinate_to_loc(left_top_x / width)}>"
            f"<loc_{_coordinate_to_loc(left_top_y / height)}>"
            f"<loc_{_coordinate_to_loc(right_bottom_x / width)}>"
            f"<loc_{_coordinate_to_loc(right_bottom_y / height)}>"
        )
    raise ValueError(
        "Provided coordinates in mixed format - coordinates must be all integers "
        "or all floats in range [0.0-1.0]"
    )
