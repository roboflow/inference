"""Find and read QR codes with zxing-cpp, restricted to the QR format.

Canonical type: ``roboflow_core/qr_code_detector@v1``.
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
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND, QR_CODE_DETECTION_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class QRCodeDetectorBlock(Block):
    """Find and read QR codes with zxing-cpp, restricted to the QR format.

    Stateless, does not mutate or keep its input, and is prunable.
    """

    type = "roboflow_core/qr_code_detector@v1"
    prunable = True
    metadata = {
        "name": "QR Code Detection",
        "port_of": "roboflow_core/qr_code_detector@v1",
        "section": "model",
    }
    outputs = {
        "predictions": Output(
            QR_CODE_DETECTION_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description=(
                "One row per QR code in image pixels; decoded text in "
                "bboxes_metadata[i]['data']."
            ),
        )
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Image to read QR codes from; RGB or grayscale.",
            examples=["$inputs.image"],
        )

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        """Read the QR codes in ``image``.

        Args:
            image: Image on any device.

        Returns:
            ``predictions``: QR code Detections on the image's device.
        """
        luminance = host_luminance(image)
        found = zxingcpp.read_barcodes(luminance, formats=zxingcpp.BarcodeFormat.QRCode)
        corners = [_zxing_corners(code.position) for code in found]
        texts = [code.text for code in found]

        predictions = code_detections(
            image,
            corners=corners,
            texts=texts,
            class_name="qr_code",
            prediction_type="qrcode-detection",
        )
        result = {"predictions": predictions}

        return result
