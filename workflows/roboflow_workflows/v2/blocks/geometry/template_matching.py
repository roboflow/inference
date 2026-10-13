"""Find copies of a template image in an image: ``v2/template_matching@v1``.

Port of V1 ``roboflow_core/template_matching@v1``. As V1::

    score  = cv2.matchTemplate(gray(image), gray(template), TM_CCOEFF_NORMED)
    boxes  = [x, y, x + w, y + h] for every (x, y) with score >= matching_threshold
    if apply_nms:  boxes = NMS(boxes, nms_threshold)

``w`` and ``h`` are the template size. Boxes are pixels of ``image`` in its own
frame, scanned row by row; every box has class 0, ``"template_match"``.

CPU boundary: OpenCV converts both images to gray and scores on the host; the
detections come back to ``image``'s device.

Differences from V1: ``confidence`` is the match score (V1: always 1.0, so V1
NMS keeps an arbitrary one of overlapping boxes; here it keeps the best
match). A template larger than the image is a ``ContractError``. V1 raises an
OpenCV error when the template is larger in one dimension only; when it is
larger in both, OpenCV swaps the two images and V1 returns boxes outside the
image. Rows carry a V2 row identity instead of V1 ``detection_id`` columns.
"""

from typing import Any, Dict

import cv2
import numpy as np
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.geometry.overlap import non_max_suppression
from roboflow_workflows.v2.blocks.geometry.sift import host_gray
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.object_detection import Detections

CLASS_NAME = "template_match"


class TemplateMatchingBlock(Block):
    """Detect every place where ``template`` appears in ``image``.

    Example: a 20 x 10 logo pasted twice into a frame gives two boxes of
    that size at the two paste corners, with confidences close to 1.0.
    """

    type = "v2/template_matching@v1"
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="One box per match; confidence is the match score.",
        ),
        "number_of_matches": Output(
            INTEGER_KIND, source="image", description="Number of boxes."
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to search.")
        template: Ref(IMAGE_KIND) = Field(
            description="Image to find; not larger than image."
        )
        matching_threshold: float | Ref(FLOAT_KIND) = Field(
            default=0.8,
            description="Minimum normalized correlation score, at most 1.0.",
            examples=[0.8, "$inputs.threshold"],
        )
        apply_nms: bool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description="Remove overlapping matches, keeping the best score.",
        )
        nms_threshold: float | Ref(FLOAT_KIND) = Field(
            default=0.5,
            ge=0,
            le=1,
            description="IoU above which NMS removes the weaker match.",
        )

    def run(
        self,
        *,
        image: ImageData,
        template: ImageData,
        matching_threshold: float,
        apply_nms: bool,
        nms_threshold: float,
    ) -> Dict[str, Any]:
        """Match ``template`` against every position of ``image``.

        Args:
            image: Image to search.
            template: Image to find.
            matching_threshold: Minimum ``TM_CCOEFF_NORMED`` score.
            apply_nms: Apply NMS to overlapping boxes.
            nms_threshold: IoU threshold of the NMS.

        Returns:
            ``predictions`` and ``number_of_matches``.

        Raises:
            ContractError: When the template is larger than the image.
        """
        template_height, template_width = template.size_hw
        if template_height > image.height or template_width > image.width:
            raise ContractError(
                f"{self.type}: template (height, width) {template.size_hw} does "
                f"not fit into image {image.size_hw}"
            )

        scores = cv2.matchTemplate(
            host_gray(image), host_gray(template), cv2.TM_CCOEFF_NORMED
        )
        ys, xs = np.nonzero(scores >= matching_threshold)
        corners = torch.from_numpy(np.stack([xs, ys], axis=1)).to(torch.float32)
        size = torch.tensor([template_width, template_height], dtype=torch.float32)
        detections = with_row_identity(
            Detections(
                xyxy=torch.cat([corners, corners + size], dim=1).to(image.device),
                class_id=torch.zeros(len(xs), dtype=torch.int64, device=image.device),
                confidence=torch.from_numpy(scores[ys, xs]).to(image.device),
                image_metadata={
                    **image.prediction_metadata(),
                    "class_names": {0: CLASS_NAME},
                    "prediction_type": "object-detection",
                },
            )
        )
        if apply_nms:
            detections = non_max_suppression(
                detections, threshold=nms_threshold, use_masks=False
            )

        result = {"predictions": detections, "number_of_matches": len(detections.xyxy)}

        return result
