"""Decode a VLM classification answer into a classification prediction.

Canonical type: ``roboflow_core/vlm_as_classifier@v2``.
"""

import uuid
from typing import Any, Dict, List
from pydantic import Field
from roboflow_workflows.v2.blocks.language.decoding import (
    decode_classification,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import LIST_OF_VALUES_KIND, STRING_KIND
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref


class VLMAsClassifierBlock(Block):
    """Decode a VLM classification answer into a classification prediction.

    ``{"class_name": "cat", "confidence": 0.9}`` gives a single-label
    prediction with the full confidence vector over ``classes``;
    ``{"predicted_classes": [{"class": ..., "confidence": ...}]}`` or a bare
    list of such entries gives a multi-label prediction. A label outside
    ``classes`` is appended after them. Stateless and prunable.
    """

    type = "roboflow_core/vlm_as_classifier@v2"
    prunable = True
    metadata = {
        "name": "VLM As Classifier",
        "port_of": "roboflow_core/vlm_as_classifier@v2",
        "section": "advanced",
    }
    outputs = vlm_prediction_outputs("classification")

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
        classes: List[str] | Ref(LIST_OF_VALUES_KIND) = Field(
            description=(
                "Class names the VLM was asked to choose from; classes[i] has "
                "class id i. Names must be unique."
            ),
            examples=[["dog", "cat"], "$inputs.classes"],
        )

    def run(
        self, *, image: ImageData, vlm_output: Any, classes: List[str]
    ) -> Dict[str, Any]:
        """Decode ``vlm_output`` for ``image``.

        Args:
            image: The classified image.
            vlm_output: Answer text or decoded dict.
            classes: Unique class names.

        Returns:
            ``predictions``, ``error_status`` and a fresh ``inference_id``.

        Raises:
            ValueError: When ``classes`` is not a list of unique strings.
        """
        error_status, predictions = decode_classification(
            vlm_output, image=image, classes=classes
        )
        result = {
            "predictions": predictions,
            "error_status": error_status,
            "inference_id": uuid.uuid4().hex,
        }

        return result
