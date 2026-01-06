from typing import List, Literal, Optional, Type
from uuid import uuid4

import cv2
import numpy as np
import supervision as sv
from pydantic import ConfigDict
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    QR_CODE_DETECTION_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Detect QR codes in images and extract their encoded data, returning bounding box coordinates and decoded text for each detected QR code.

## How This Block Works

This block uses OpenCV's QRCodeDetector to detect and decode QR codes in images. The block:

1. Takes images as input (supports batch processing)
2. Scans each image for QR code patterns using OpenCV's QR code detector
3. Detects QR code regions and extracts the encoded data/text from each QR code
4. Generates bounding boxes around each detected QR code
5. Returns predictions with bounding box coordinates, confidence scores, and the decoded QR code data/text

The block can detect multiple QR codes in a single image and automatically decodes the information stored in each code. QR codes are 2D barcodes that can store more data than traditional linear barcodes, including URLs, contact information, text, and other data. Each detected QR code includes its location (bounding box), decoded text/data, and detection metadata.

## Common Use Cases

- **Contactless Information Access**: Read QR codes containing URLs, contact information, WiFi credentials, or other encoded data for automated information retrieval
- **Product Authentication**: Verify product authenticity by reading QR codes on packaging for anti-counterfeiting, supply chain traceability, or brand protection
- **Event and Ticketing**: Process QR code tickets, event passes, or access codes for entry verification, attendance tracking, or event management
- **Marketing and Advertising**: Extract data from QR codes in advertisements, promotional materials, or product packaging for campaign tracking or customer engagement
- **Mobile Payment Processing**: Read payment QR codes for transaction processing, payment verification, or financial services
- **Asset and Inventory Tracking**: Track assets, equipment, or inventory items by reading QR code labels for identification, location tracking, or maintenance scheduling

## Connecting to Other Blocks

The QR code detections from this block can be connected to:

- **Crop blocks** (e.g., Dynamic Crop, Absolute Static Crop) to isolate individual QR code regions for further processing, validation, or quality checks
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log QR code data and metadata for record-keeping, analytics, or tracking
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on detected QR code values, URLs, or the presence of specific QR codes
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts when specific QR codes are detected, when QR code reading succeeds or fails, or for access control notifications
- **Webhook blocks** to send QR code data to external systems, databases, or APIs for authentication, payment processing, inventory updates, or tracking
- **Expression blocks** to parse, validate, or transform QR code data (e.g., extract URLs, validate formats, parse structured data from QR code content)
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "QR Code Detection",
            "version": "v1",
            "short_description": "Detect and read QR codes in an image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-qrcode",
                "blockPriority": 13,
                "opencv": True,
            },
        }
    )
    type: Literal[
        "roboflow_core/qr_code_detector@v1", "QRCodeDetector", "QRCodeDetection"
    ]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[QR_CODE_DETECTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class QRCodeDetectorBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
    ) -> BlockResult:
        results = []
        for image in images:
            qr_code_detections = detect_qr_codes(image=image)
            results.append({"predictions": qr_code_detections})
        return results


def detect_qr_codes(image: WorkflowImageData) -> sv.Detections:
    detector = cv2.QRCodeDetector()
    retval, detections, points_list, _ = detector.detectAndDecodeMulti(
        image.numpy_image
    )
    xyxy = []
    confidence = []
    class_id = []
    class_name = []
    extracted_data = []
    for data, points in zip(detections, points_list):
        width = points[2][0] - points[0][0]
        height = points[2][1] - points[0][1]
        x_min = points[0][0]
        y_min = points[0][1]
        x_max = x_min + width
        y_max = y_min + height
        xyxy.append([x_min, y_min, x_max, y_max])
        class_id.append(0)
        class_name.append("qr_code")
        confidence.append(1.0)
        extracted_data.append(data)
    xyxy = np.array(xyxy) if len(xyxy) > 0 else np.empty((0, 4))
    confidence = np.array(confidence) if len(confidence) > 0 else np.empty(0)
    class_id = np.array(class_id).astype(int) if len(class_id) > 0 else np.empty(0)
    class_name = np.array(class_name) if len(class_name) > 0 else np.empty(0)
    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=confidence,
        class_id=class_id,
        data={CLASS_NAME_DATA_FIELD: class_name},
    )
    detections[DETECTION_ID_KEY] = np.array([uuid4() for _ in range(len(detections))])
    detections[PREDICTION_TYPE_KEY] = np.array(["qrcode-detection"] * len(detections))
    detections[DETECTED_CODE_KEY] = np.array(extracted_data)
    img_height, img_width = image.numpy_image.shape[:2]
    detections[IMAGE_DIMENSIONS_KEY] = np.array(
        [[img_height, img_width]] * len(detections)
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )
