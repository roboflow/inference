"""Workflows-local twins of the two RESPONSE-level instance-segmentation DTOs.

`InferenceResponseImage` and `InstanceSegmentationInferenceResponse` are only
ever built from locals - `width`/`height` from `numpy_image.shape` and a list
of prediction objects the block has already built or validated (sweep: every
constructor call site, `## Entity-class validation-parity sweep`). The
prediction-level classes are NOT twinned: `Point`,
`InstanceSegmentationPrediction`, `InstanceSegmentationRLEPrediction` and
`Sam2SegmentationPrediction` are built from remote responses too, so they keep
their pydantic validation and live in `segmentation_entities.py` (moved from the
server, which re-exports them).

NOTE for callers: `sv.Detections.from_inference` subscripts its argument (it
only unwraps objects exposing `.dict()`/`.json()`), so pass `to_dict()`, never
the dataclass.

`tests/workflows/unit_tests/core_steps/common/test_segmentation_entities.py`
pins every `to_dict()` to the corresponding pydantic dump and pins the
supervision conversion.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from inference.core.workflows.core_steps.common.segmentation_entities import (
    InstanceSegmentationPrediction,
    InstanceSegmentationRLEPrediction,
)


@dataclass(slots=True)
class InferenceResponseImageDC:
    width: int
    height: int

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height}


@dataclass(slots=True)
class InstanceSegmentationInferenceResponseDC:
    image: InferenceResponseImageDC
    predictions: List[
        Union[InstanceSegmentationPrediction, InstanceSegmentationRLEPrediction]
    ] = field(default_factory=list)
    inference_id: Optional[str] = None
    frame_id: Optional[int] = None
    time: Optional[float] = None
    visualization: Optional[Any] = None

    def to_dict(self) -> dict:
        """The exact dict `InstanceSegmentationInferenceResponse.model_dump(
        by_alias=True, exclude_none=True)` produces for the same content."""
        result = {
            "image": self.image.to_dict(),
            "predictions": [
                prediction.model_dump(by_alias=True, exclude_none=True)
                for prediction in self.predictions
            ],
        }
        if self.inference_id is not None:
            result["inference_id"] = self.inference_id
        if self.frame_id is not None:
            result["frame_id"] = self.frame_id
        if self.time is not None:
            result["time"] = self.time
        if self.visualization is not None:
            result["visualization"] = self.visualization
        return result
