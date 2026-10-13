"""Blocks that read and write barcodes and QR codes.

``detectors`` reads barcodes and QR codes in images (zxing-cpp).
``generator`` draws text as a QR code image (qrcode).
"""

from roboflow_workflows.v2.blocks.codes.detectors import (
    BarcodeDetectorBlock,
    QRCodeDetectorBlock,
)
from roboflow_workflows.v2.blocks.codes.generator import (
    QRCodeGeneratorBlock,
)
from roboflow_workflows.v2.blocks.codes.ocr_stitch import (
    StitchOCRDetectionsBlock,
)

BLOCKS = (
    StitchOCRDetectionsBlock,
    QRCodeGeneratorBlock,
    BarcodeDetectorBlock,
    QRCodeDetectorBlock,
)

__all__ = [
    "BLOCKS",
    "StitchOCRDetectionsBlock",
    "QRCodeGeneratorBlock",
    "BarcodeDetectorBlock",
    "QRCodeDetectorBlock",
]
