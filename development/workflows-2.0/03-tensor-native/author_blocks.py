"""Author exercise: synthetic native detections and ordinary confidence selection."""

from typing import Any, Dict

import torch
from pydantic import Field
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.prediction_ops import (
    select_predictions,
)
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import BOOLEAN_KIND, FLOAT_KIND

from inference_models.models.base.object_detection import Detections


class SyntheticDetections(Block):
    """Emit two fixed boxes with brightness-dependent scores; no model is run."""

    type = "tensor_demo/synthetic_detections"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Image defining prediction geometry."
        )

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        """Build native predictions directly on the image's device.

        Args:
            image: The crop that owns the prediction coordinates.

        Returns:
            Two synthetic detections with stable row IDs and image provenance.
        """
        bright = bool(image.tensor_image.float().mean() >= 100)
        prediction = Detections(
            xyxy=torch.tensor(
                [[2, 4, 6, 8], [10, 10, 14, 14]],
                dtype=torch.float32,
                device=image.device,
            ),
            class_id=torch.tensor([0, 0], dtype=torch.int64, device=image.device),
            confidence=torch.tensor(
                [0.95, 0.2] if bright else [0.3, 0.1],
                dtype=torch.float32,
                device=image.device,
            ),
            image_metadata={
                **image.prediction_metadata(),
                "class_names": {0: "synthetic-marker"},
                "fixture": "offline; not model inference",
            },
            bboxes_metadata=[
                {
                    "detection_id": f"{image.image_id}:box-{index}",
                    "class_name": "synthetic-marker",
                }
                for index in range(2)
            ],
        )
        result = {"predictions": prediction}

        return result


class SelectConfidence(Block):
    """Select all aligned native fields, then expose a plain boolean condition."""

    type = "tensor_demo/select_confidence"
    outputs = {
        "predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="predictions"),
        "keep": Output(BOOLEAN_KIND, source="predictions"),
    }

    class Params(BlockParams):
        predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Native detections to select."
        )
        minimum: float | Ref(FLOAT_KIND) = Field(
            default=0.5, ge=0, le=1, description="Inclusive confidence threshold."
        )

    def run(self, *, predictions: Detections, minimum: float) -> Dict[str, Any]:
        """Keep high-confidence rows without rebuilding metadata manually.

        Args:
            predictions: Input native detection carrier.
            minimum: Inclusive score threshold.

        Returns:
            Selected native rows and whether at least one row survived.
        """
        selected = select_predictions(
            predictions, mask=predictions.confidence >= minimum
        )
        result = {"predictions": selected, "keep": len(selected) > 0}

        return result


class AcceptPrediction(Block):
    """Make a conditional consumer visible while preserving payload identity."""

    type = "tensor_demo/accept"
    outputs = {
        "predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="predictions"),
        "image": Output(IMAGE_KIND, source="image"),
    }

    class Params(BlockParams):
        predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Prediction retained by the confidence condition."
        )
        image: Ref(IMAGE_KIND) = Field(description="The prediction's own image frame.")

    def run(self, *, predictions: Detections, image: ImageData) -> Dict[str, Any]:
        """Forward the same payload objects after the engine's gate.

        Args:
            predictions: Selected native prediction.
            image: Matching image.

        Returns:
            The original objects, making alias preservation directly observable.
        """
        result = {"predictions": predictions, "image": image}

        return result


def create_demo_catalogue() -> Catalogue:
    """Add the three example blocks to the opt-in media catalogue.

    Returns:
        Catalogue suitable for the example workflow definitions.
    """
    catalogue = create_catalogue().with_blocks(
        [SyntheticDetections, SelectConfidence, AcceptPrediction]
    )

    return catalogue
