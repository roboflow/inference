"""Blocks that read and write barcodes and QR codes.

``detectors`` reads barcodes and QR codes in images (zxing-cpp).
``generator`` draws text as a QR code image (qrcode).
"""

from roboflow_workflows.v2.blocks.codes.barcode_detector.v1 import BarcodeDetectorBlock
from roboflow_workflows.v2.blocks.codes.qr_code_detector.v1 import QRCodeDetectorBlock
from roboflow_workflows.v2.blocks.codes.qr_code_generator.v1 import QRCodeGeneratorBlock
from roboflow_workflows.v2.blocks.codes.stitch_ocr_detections.v2 import StitchOCRDetectionsBlock

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
