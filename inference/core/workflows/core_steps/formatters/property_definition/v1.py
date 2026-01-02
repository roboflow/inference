from typing import Any, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import Selector
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Extract specific properties or fields from workflow step outputs using configurable operation chains to extract class names, confidences, counts, coordinates, OCR text, metadata, and other properties from model predictions or workflow data for data transformation, property extraction, metadata access, and value extraction workflows.

## How This Block Works

This block extracts specific properties from data by applying a chain of operations that navigate and extract values from complex data structures. The block:

1. Receives input data from any workflow step (detections, classifications, OCR results, images, or other data types)
2. Applies a chain of operations defined in the operations parameter:
   - Each operation performs a specific extraction or transformation task
   - Operations are executed sequentially, with each operation working on the result of the previous one
   - Operations can extract properties, filter data, transform formats, or combine values
3. Extracts properties based on operation type:

   **For Detection Properties:**
   - Extracts properties from object detection, instance segmentation, or keypoint detection predictions
   - Can extract: class names, confidences, counts, bounding box coordinates (x_min, y_min, x_max, y_max), centers, sizes, tracker IDs, velocities, speeds, path deviations, time in zone, polygons, and more
   - Returns lists of values (one per detection) or aggregated values

   **For Classification Properties:**
   - Extracts properties from classification predictions
   - Can extract: predicted class, confidence scores, all classes, all confidences
   - Returns single values or lists depending on the property

   **For OCR Properties:**
   - Extracts text, coordinates, and metadata from OCR results
   - Can extract: recognized text, bounding box information, confidence scores

   **For Image Properties:**
   - Extracts metadata and properties from images
   - Can extract: dimensions, format information, and other image metadata

4. Supports compound operations for complex extractions:
   - Operations can be chained to perform multi-step extractions
   - Can filter detections before extracting properties
   - Can select specific detections, transform formats, or combine multiple properties
5. Returns the extracted property value:
   - Output type depends on the property extracted (list, string, number, dictionary, etc.)
   - Returns a single output value containing the extracted property

The block uses a flexible operation system that allows extracting virtually any property from workflow data. Operations can be simple (extract a single property) or compound (filter, transform, then extract). This makes the block highly versatile for accessing specific fields from complex data structures without needing custom code.

## Common Use Cases

- **Property Extraction**: Extract specific fields from model predictions (e.g., extract class names from detections, get confidence scores, extract OCR text, get detection counts), enabling property extraction workflows
- **Metadata Access**: Access metadata and computed properties from workflow steps (e.g., extract tracker IDs, get velocity values, access time in zone, retrieve path deviations), enabling metadata access workflows
- **Data Transformation**: Transform complex data structures into simpler values for downstream use (e.g., convert detections to lists, extract coordinates, get bounding box centers, extract class lists), enabling data transformation workflows
- **Conditional Logic**: Extract values for use in conditional logic or decision making (e.g., extract counts for thresholds, get confidences for filtering, extract class names for classification, get coordinates for calculations), enabling conditional logic workflows
- **Data Formatting**: Format data for storage, display, or API responses (e.g., extract values for JSON output, format data for storage, prepare data for visualization, extract for API responses), enabling data formatting workflows
- **Analytics Extraction**: Extract metrics and measurements for analysis (e.g., extract detection counts, get confidence statistics, extract measurement values, retrieve analytics metrics), enabling analytics extraction workflows

## Connecting to Other Blocks

This block receives data from any workflow step and produces extracted property values:

- **After model blocks** (detection, classification, OCR, etc.) to extract properties from predictions (e.g., extract class names from detections, get classification results, extract OCR text), enabling model-to-property workflows
- **After analytics blocks** to extract computed metrics and measurements (e.g., extract velocity values, get time in zone, retrieve path deviations, access tracking information), enabling analytics-to-property workflows
- **Before logic blocks** like Continue If to use extracted values in conditions (e.g., continue if count exceeds threshold, filter based on extracted confidence, make decisions using extracted values), enabling property-based decision workflows
- **Before data storage blocks** to format extracted values for storage (e.g., store extracted properties, format values for logging, prepare data for storage), enabling property-to-storage workflows
- **Before visualization blocks** to provide extracted values for display (e.g., display extracted counts, show extracted text, visualize extracted metrics), enabling property visualization workflows
- **Before notification blocks** to use extracted values in notifications (e.g., include extracted counts in alerts, send extracted text in messages, use extracted values in notifications), enabling property-based notification workflows

## Requirements

This block works with any data type from workflow steps. The operations parameter defines a list of operations to perform on the input data. Each operation must be compatible with the data type and previous operation outputs. Common operations include DetectionsPropertyExtract (for detection properties), ClassificationPropertyExtract (for classification properties), and other extraction operations. The block supports compound operations (operations that can contain other operations) for complex extractions. The output type depends on the operations performed and the properties extracted - it can be a list, string, number, dictionary, or other types depending on what is extracted.
"""

SHORT_DESCRIPTION = "Define a variable from model predictions, such as the class names, confidences, or number of detections."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Property Definition",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "search_keywords": [
                "property",
                "field",
                "number",
                "count",
                "classes",
                "confidences",
                "labels",
                "coordinates",
            ],
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-gear-code",
                "blockPriority": 0,
                "popular": True,
            },
        }
    )
    type: Literal[
        "roboflow_core/property_definition@v1",
        "PropertyDefinition",
        "PropertyExtraction",
    ]
    data: Selector() = Field(
        description="Input data from any workflow step to extract properties from. Can be detections, classifications, OCR results, images, or any other workflow output. The data type determines which operations are applicable. Examples: detection predictions for extracting class names, classification results for extracting predicted class, OCR results for extracting text.",
        examples=["$steps.object_detection_model.predictions", "$steps.classification_model.top", "$steps.ocr_model.text"],
    )
    operations: List[AllOperationsType] = Field(
        description="List of operations to perform sequentially on the input data. Each operation performs extraction, filtering, transformation, or combination. Operations execute in order, with each operation working on the previous result. Common operations: DetectionsPropertyExtract (extract properties like class_name, confidence, count, coordinates from detections), ClassificationPropertyExtract (extract class, confidence from classifications), DetectionsFilter (filter detections before extraction), DetectionsSelection (select specific detections). Can include single or compound operations for complex extractions.",
        examples=[
            [{"type": "DetectionsPropertyExtract", "property_name": "class_name"}],
            [{"type": "DetectionsPropertyExtract", "property_name": "count"}],
            [{"type": "ClassificationPropertyExtract", "property_name": "class"}],
        ],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class PropertyDefinitionBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        data: Any,
        operations: List[AllOperationsType],
    ) -> BlockResult:
        operations_chain = build_operations_chain(operations=operations)
        return {"output": operations_chain(data, global_parameters={})}
