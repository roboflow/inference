from typing import List, Literal, Optional, Type
from uuid import uuid4

import numpy as np
import supervision as sv
import zxingcpp
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
    BAR_CODE_DETECTION_KIND,
    IMAGE_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Detect barcodes in images and extract their encoded data, returning bounding box coordinates and decoded text for each detected barcode.

## How This Block Works

This block uses the zxingcpp library to detect and decode barcodes in images. The block:

1. Takes images as input (supports batch processing)
2. Scans each image for barcode patterns using the zxingcpp barcode reader
3. Detects barcode regions and extracts the encoded data/text from each barcode
4. Generates bounding boxes around each detected barcode
5. Returns predictions with bounding box coordinates, confidence scores, and the decoded barcode data/text

The block automatically detects various barcode formats and extracts the encoded information. Each detected barcode includes its location (bounding box), decoded text/data, and detection metadata. This makes it useful for processing images that contain barcodes, allowing you to locate and extract information from them automatically.

## Common Use Cases

- **Inventory Management**: Automatically scan and read barcodes on products for inventory tracking, stock management, or warehouse operations
- **Retail Point of Sale**: Detect and read product barcodes for price lookup, checkout systems, or product identification in retail environments
- **Package Tracking**: Read shipping barcodes or tracking codes on packages for logistics, delivery tracking, or postal services
- **Manufacturing Quality Control**: Verify product barcodes during manufacturing processes, ensuring correct labeling and traceability
- **Document Processing**: Extract barcode information from documents, tickets, or forms for automated data entry or document indexing
- **Asset Management**: Track assets by reading barcode labels on equipment, tools, or inventory items

## Connecting to Other Blocks

The barcode detections from this block can be connected to:

- **Crop blocks** (e.g., Dynamic Crop, Absolute Static Crop) to isolate individual barcode regions for further processing or validation
- **OCR blocks** (e.g., OCR Model, EasyOCR) to read any text that appears alongside or within barcode regions
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log barcode data and metadata for record-keeping or analysis
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on detected barcode values or the presence of barcodes
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts when specific barcodes are detected or when barcode reading fails
- **Webhook blocks** to send barcode data to external systems, databases, or APIs for inventory updates or tracking
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Barcode Detection",
            "version": "v1",
            "short_description": "Detect and read barcodes in an image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-barcode",
                "blockPriority": 12,
            },
        }
    )
    type: Literal[
        "roboflow_core/barcode_detector@v1", "BarcodeDetector", "BarcodeDetection"
    ]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="predictions", kind=[BAR_CODE_DETECTION_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BarcodeDetectorBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
    ) -> BlockResult:
        results = []
        for image in images:
            qr_code_detections = detect_barcodes(image=image)
            results.append({"predictions": qr_code_detections})
        return results


def detect_barcodes(image: WorkflowImageData) -> sv.Detections:
    barcodes = zxingcpp.read_barcodes(image.numpy_image)
    xyxy = []
    confidence = []
    class_id = []
    class_name = []
    extracted_data = []
    for barcode in barcodes:
        x_min = barcode.position.top_left.x
        y_min = barcode.position.top_left.y
        x_max = barcode.position.bottom_right.x
        y_max = barcode.position.bottom_right.y
        xyxy.append([x_min, y_min, x_max, y_max])
        class_id.append(0)
        class_name.append("barcode")
        confidence.append(1.0)
        extracted_data.append(barcode.text)
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
    detections[PREDICTION_TYPE_KEY] = np.array(["barcode-detection"] * len(detections))
    detections[DETECTED_CODE_KEY] = np.array(extracted_data)
    img_height, img_width = image.numpy_image.shape[:2]
    detections[IMAGE_DIMENSIONS_KEY] = np.array(
        [[img_height, img_width]] * len(detections)
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )
