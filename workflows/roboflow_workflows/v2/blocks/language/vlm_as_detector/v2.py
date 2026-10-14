"""Decode a VLM detection answer into boxes on the image.

Canonical type: ``roboflow_core/vlm_as_detector@v2``.

``model_type`` selects the box format through ``MODEL_BOX_FORMATS`` in
``_shared/parsers.py``, whose module docstring has the full table. Florence-2
answers go to ``decode_florence_2_detections`` instead.
"""

import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, model_validator
from roboflow_workflows.v2.blocks.language._shared.parsers import (
    MODEL_BOX_FORMATS,
    decode_florence_2_detections,
)
from roboflow_workflows.v2.blocks.language.decoding import (
    decode_object_detections,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import LIST_OF_VALUES_KIND, STRING_KIND
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref

# Exactly the MODEL_BOX_FORMATS keys plus "florence-2"; test_parsers.py checks it.
ModelType = Literal[
    "openai",
    "google-gemini",
    "anthropic-claude",
    "spacexai",
    "qwen",
    "muse",
    "zai",
    "zai-flash",
    "florence-2",
]

DetectorTaskType = Literal[
    "object-detection",
    "object-detection-and-caption",
    "open-vocabulary-object-detection",
    "phrase-grounded-object-detection",
    "region-proposal",
    "ocr-with-text-detection",
]


class VLMAsDetectorBlock(Block):
    """Decode a VLM detection answer into boxes on the image.

    ``model_type`` names the model that answered, which fixes the coordinate
    format to read (see the module table). Labels outside ``classes`` keep
    their name with class id ``-1``. Stateless and prunable.
    """

    type = "roboflow_core/vlm_as_detector@v2"
    prunable = True
    metadata = {
        "name": "VLM As Detector",
        "port_of": "roboflow_core/vlm_as_detector@v2",
        "section": "advanced",
    }
    outputs = vlm_prediction_outputs("object-detection")

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "The image the VLM answered about; sets the prediction's size, "
                "frame and device."
            ),
            examples=["$inputs.image", "$steps.crop.crops"],
        )
        vlm_output: str | Ref(LANGUAGE_MODEL_OUTPUT_KIND, STRING_KIND) = Field(
            description=(
                "Answer of a VLM or LLM step: text with JSON, raw or inside a Markdown "
                "fence (the first ```json fence is read), or an already decoded dict."
            ),
            examples=["$steps.vlm.output"],
        )
        classes: Optional[List[str]] | Ref(LIST_OF_VALUES_KIND) = Field(
            default=None,
            description=(
                "Class names the VLM was asked to detect; classes[i] has class "
                "id i. Names must be unique. Required except for Florence-2 "
                "tasks other than open-vocabulary-object-detection."
            ),
            examples=[["dog", "cat"], "$inputs.classes"],
        )
        model_type: ModelType = Field(
            description=(
                "Model that produced the answer; selects the box coordinate "
                "format to read."
            ),
            examples=["google-gemini", "qwen"],
        )
        task_type: DetectorTaskType = Field(
            default="object-detection",
            description=(
                "Task the model ran. Every model supports object-detection; "
                "the other tasks are Florence-2 tasks."
            ),
            examples=["object-detection", "ocr-with-text-detection"],
        )

        @model_validator(mode="after")
        def _task_and_classes_fit_the_model(self) -> "VLMAsDetectorBlock.Params":
            florence = self.model_type == "florence-2"
            if not florence and self.task_type != "object-detection":
                raise ValueError(
                    f"model_type {self.model_type!r} supports only task_type "
                    f"'object-detection', got {self.task_type!r}"
                )
            needs_classes = (
                not florence or self.task_type == "open-vocabulary-object-detection"
            )
            if needs_classes and self.classes is None:
                raise ValueError(
                    f"classes is required for model_type {self.model_type!r} "
                    f"and task_type {self.task_type!r}"
                )

            return self

    def run(
        self,
        *,
        image: ImageData,
        vlm_output: Any,
        model_type: str,
        task_type: str = "object-detection",
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Decode ``vlm_output`` into ``Detections`` on ``image``.

        Args:
            image: The image the model saw.
            vlm_output: Answer text or decoded dict.
            model_type: Model that answered.
            task_type: Task the model ran.
            classes: Unique class names, or ``None`` where allowed.

        Returns:
            ``predictions``, ``error_status`` and a fresh ``inference_id``,
            also the namespace of the rows' identity.

        Raises:
            ValueError: When ``classes`` is not a list of unique strings.
        """
        inference_id = uuid.uuid4().hex
        if model_type == "florence-2":
            error_status, predictions = decode_florence_2_detections(
                vlm_output,
                image=image,
                task_type=task_type,
                classes=classes,
                inference_id=inference_id,
            )
        else:
            box_format, upload_size = MODEL_BOX_FORMATS[model_type]
            height, width = image.size_hw
            error_status, predictions = decode_object_detections(
                vlm_output,
                image=image,
                classes=classes,
                box_format=box_format,
                inference_id=inference_id,
                upload_size_wh=upload_size((width, height)) if upload_size else None,
            )
        result = {
            "predictions": predictions,
            "error_status": error_status,
            "inference_id": inference_id,
        }

        return result
