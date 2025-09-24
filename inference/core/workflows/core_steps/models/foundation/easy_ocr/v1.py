import hashlib
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import ConfigDict, Field

import supervision as sv

from inference.core.cache.lru_cache import LRUCache
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from supervision.config import CLASS_NAME_DATA_FIELD
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PREDICTION_TYPE_KEY,
)

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import attach_parents_coordinates_to_sv_detections, load_core_model
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    EMBEDDING_KIND,
    IMAGE_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient

LANGUAGES = Literal[
    "English",
    "Japanese",
    "Kannada",
    "Korean",
    "Latin",
    "Telugu",
    "Simplified Chinese",
]

MODELS: Dict[str, str] = {
    "English": "english_g2",
    "Japanese": "japanese_g2",
    "Kannada": "kannada_g2",
    "Korean": "korean_g2",
    "Latin": "latin_g2",
    "Telugu": "telugu_g2",
    "Simplified Chinese": "zh_sim_g2",
}

LONG_DESCRIPTION = """
"""

class BlockManifest(WorkflowBlockManifest):

    model_config = ConfigDict(
        json_schema_extra={
            "name": "EasyOCR",
            "version": "v1",
            "short_description": "Extract text from an image using EasyOCR.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-text",
                "blockPriority": 11,
                "inDevelopment": False,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/easy_ocr@v1", "EasyOCR"]
    name: str = Field(description="Unique name of step in workflows")
    data: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image"],
    )
    language: LANGUAGES = Field(
        title="Language",
        description="Language model to use for OCR",
        default="English",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def ocr_result_to_detections(
    image: WorkflowImageData, result: List[Tuple[List[List[int]], str, float]]
) -> sv.Detections:
    # Prepare lists for bounding boxes, confidences, class IDs, and labels
    xyxy, confidences, class_ids, class_names = [], [], [], []

    # Extract data from OCR result
    for detection in result:
        bbox, text, confidence = detection[0], detection[1], detection[2]

        # Convert bounding box format
        x_min = bbox[0][0]
        y_min = bbox[0][1]
        x_max = bbox[2][0]
        y_max = bbox[2][1]

        # Append data to lists
        xyxy.append([x_min, y_min, x_max, y_max])
        confidences.append(confidence)
        class_ids.append(0)
        class_names.append(text)

    # Convert to NumPy arrays
    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidences),
        class_id=np.array(class_ids),
        data={CLASS_NAME_DATA_FIELD: np.array(class_names)},
    )
    detections[DETECTION_ID_KEY] = np.array([uuid4() for _ in range(len(detections))])
    detections[PREDICTION_TYPE_KEY] = np.array(["easy-ocr"] * len(detections))
    img_height, img_width = image.numpy_image.shape[:2]
    detections[IMAGE_DIMENSIONS_KEY] = np.array(
        [[img_height, img_width]] * len(detections)
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )

class EasyOCRBlockV1(WorkflowBlock):
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
        data: WorkflowImageData,
        language: str = "English",
    ) -> BlockResult:
        version = MODELS[language]
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(data=data, version=version)
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for EasyOCR. Please use a local or dedicated inference server."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        data: WorkflowImageData,
        version: str,
    ) -> BlockResult:

        inference_request = EasyOCRInferenceRequest(
            easy_ocr_version_id=version,
            image=[data.to_inference_format(numpy_preferred=True)],
            api_key=self._api_key,
        )
        model_id = load_core_model(
            model_manager=self._model_manager,
            inference_request=inference_request,
            core_model="easy_ocr",
        )
        predictions = self._model_manager.infer_from_request_sync(
            model_id, inference_request
        )

        detections = sv.Detections.empty() if len(predictions.result)==0 else ocr_result_to_detections(data, predictions.result)

        return {"predictions": detections}
