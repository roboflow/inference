"""Test-only block turning a single-label classification into a decision.

The classification block keeps the full distribution and the top class; a
downstream step decides what is confident enough. V2 has no product block for
this comparison yet, so the gating tests use this one with ``roboflow_core/continue_if_boolean@v1``.
"""

from pydantic import Field, StrictFloat
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND
from roboflow_workflows.v2.data.predictions import CLASSIFICATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.classification import ClassificationPrediction


class TopConfidenceAtLeast(Block):
    """``confident`` is true when the top class's confidence reaches a minimum."""

    type = "test/top_confidence_at_least@v1"
    outputs = {
        "confident": Output(BOOLEAN_KIND, source="prediction"),
        "top_confidence": Output(FLOAT_KIND, source="prediction"),
    }

    class Params(BlockParams):
        prediction: Ref(CLASSIFICATION_PREDICTION_KIND) = Field(
            description="One image's single-label classification."
        )
        minimum: StrictFloat = Field(ge=0, le=1, description="Lowest accepted value.")

    def run(self, *, prediction: ClassificationPrediction, minimum: float) -> dict:
        """Compare the confidence of the top class with ``minimum``."""
        top_class = int(prediction.class_id[0])
        top_confidence = float(prediction.confidence[0, top_class])
        result = {
            "confident": top_confidence >= minimum,
            "top_confidence": top_confidence,
        }

        return result
