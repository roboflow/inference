"""Find and read barcodes of every zxing-cpp format, including QR codes.

Canonical type: ``roboflow_core/barcode_detector@v1``.
"""

from typing import Any, Dict
import zxingcpp
from pydantic import Field
from roboflow_workflows.v2.blocks.codes._shared.detectors import (
    _zxing_corners,
    code_detections,
    host_luminance,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import BAR_CODE_DETECTION_KIND, OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class BarcodeDetectorBlock(Block):
    """Find and read barcodes of every zxing-cpp format, including QR codes.

    Stateless, does not mutate or keep its input, and is prunable.
    """

    type = "roboflow_core/barcode_detector@v1"
    prunable = True
    metadata = {
        "name": "Barcode Detection",
        "port_of": "roboflow_core/barcode_detector@v1",
        "section": "model",
    }
    outputs = {
        "predictions": Output(
            BAR_CODE_DETECTION_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description=(
                "One row per barcode in image pixels; decoded text in "
                "bboxes_metadata[i]['data']."
            ),
        )
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Image to read barcodes from; RGB or grayscale.",
            examples=["$inputs.image"],
        )

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        """Read the barcodes in ``image``.

        Args:
            image: Image on any device.

        Returns:
            ``predictions``: barcode Detections on the image's device.
        """
        luminance = host_luminance(image)
        found = zxingcpp.read_barcodes(luminance)
        corners = [_zxing_corners(barcode.position) for barcode in found]
        texts = [barcode.text for barcode in found]

        predictions = code_detections(
            image,
            corners=corners,
            texts=texts,
            class_name="barcode",
            prediction_type="barcode-detection",
        )
        result = {"predictions": predictions}

        return result
