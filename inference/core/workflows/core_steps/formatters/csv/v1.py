from copy import copy
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Literal, Optional, Type, Union

import pandas as pd
from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Convert workflow data into structured CSV format by defining custom columns, applying data transformations, and aggregating batch data into CSV documents with automatic timestamp tracking for logging, reporting, and data export workflows.

## How This Block Works

This block formats workflow data into CSV (Comma-Separated Values) format by organizing data from multiple sources into structured columns. The block:

1. Takes data references from `columns_data` dictionary that maps column names to workflow data sources (selectors, static values, or workflow inputs)
2. Optionally applies data transformation operations using `columns_operations`, which uses the Query Language (UQL) to transform column data (e.g., extract properties from detections, perform calculations, format values)
3. Automatically adds a `timestamp` column with the current UTC time in ISO format (e.g., `2024-10-18T14:09:57.622297+00:00`) to each row - note that "timestamp" is a reserved column name
4. Handles batch inputs by aggregating multiple data points into rows:
   - For single input (`batch_size=1`): Creates CSV with header row and one data row
   - For batch inputs (`batch_size>1`): Creates CSV with header row and one row per input, aggregating all rows into a single CSV document that is output only in the last batch element (earlier elements return empty CSV content)
5. Aligns batch parameters when multiple batch inputs are provided, broadcasting non-batch parameters to match the maximum batch size
6. Converts the structured data dictionary into CSV format using pandas DataFrame serialization
7. Returns `csv_content` as a string containing the complete CSV document (header and data rows)

The block supports flexible column definition where each column can reference different workflow data sources (detection predictions, classification results, workflow inputs, computed values, etc.) and optionally apply transformations to extract specific properties or format data. The automatic timestamp column enables temporal tracking of when each CSV row was generated, useful for logging and time-series data collection. Batch aggregation allows the block to collect data from multiple workflow executions and combine them into a single CSV document, which is particularly useful for batch processing workflows where you want to log multiple detections, images, or analysis results into one CSV file.

## Common Use Cases

- **Detection Logging and Reporting**: Create CSV logs of detection results (e.g., log class names, confidence scores, bounding box coordinates from object detection models), enabling structured logging of inference results for analysis, debugging, or audit trails
- **Time-Series Data Collection**: Aggregate workflow metrics, counts, or analysis results over time into CSV format (e.g., log line counter counts, zone occupancy, detection frequencies), creating time-stamped datasets for trend analysis or reporting
- **Batch Data Export**: Collect and aggregate data from batch processing workflows into CSV files (e.g., export all detections from a batch of images, collect metrics from multiple workflow runs), enabling efficient bulk data export and reporting
- **Structured Data Transformation**: Extract and format specific properties from complex workflow outputs (e.g., extract class names from detections, convert nested data structures into flat CSV columns), enabling data transformation for downstream analysis or external systems
- **Integration with External Systems**: Format workflow data for compatibility with external tools (e.g., create CSV files for spreadsheet analysis, database import, or business intelligence tools), enabling seamless data export and integration workflows
- **Data Aggregation and Analysis**: Combine data from multiple workflow sources into structured CSV format (e.g., merge detection results with metadata, combine model outputs with reference data), enabling comprehensive data collection and analysis workflows

## Connecting to Other Blocks

The CSV content from this block can be connected to:

- **Detection or analysis blocks** (e.g., Object Detection Model, Instance Segmentation Model, Classification Model, Keypoint Detection Model, Line Counter, Time in Zone) to format their outputs into CSV columns, enabling structured logging and export of inference results and analytics data
- **Data storage blocks** (e.g., Local File Sink) to save CSV files to disk, enabling persistent storage of formatted workflow data for later analysis or reporting
- **Notification blocks** (e.g., Email Notification, Slack Notification) to attach or include CSV content in notifications, enabling CSV reports to be sent as email attachments or included in message bodies
- **Webhook blocks** (e.g., Webhook Sink) to send CSV content to external APIs or services, enabling integration with external systems that consume CSV data
- **Other formatter blocks** (e.g., JSON Parser, Expression) to further process CSV content or convert it to other formats, enabling multi-stage data transformation workflows
- **Batch processing workflows** where multiple data points need to be aggregated into a single CSV document, allowing comprehensive logging and export of batch processing results
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "CSV Formatter",
            "version": "v1",
            "short_description": "Create CSV files with specified columns.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-file-csv",
                "blockPriority": 2,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/csv_formatter@v1"]
    columns_data: Dict[
        str,
        Union[
            Selector(),
            str,
            int,
            float,
            bool,
        ],
    ] = Field(
        description="Dictionary mapping column names to data sources for constructing CSV columns. Keys are column names (note: 'timestamp' is reserved and cannot be used). Values can be selectors referencing workflow data (e.g., '$steps.model.predictions', '$inputs.data'), static values (strings, numbers, booleans), or a mix of both. Each key-value pair creates one CSV column. Supports batch inputs - if values are batches, the CSV will aggregate all batch elements into rows. Example: {'predictions': '$steps.object_detection.predictions', 'count': '$steps.line_counter.count_in'} creates CSV columns named 'predictions' and 'count'.",
        examples=[
            {
                "predictions": "$steps.model.predictions",
                "reference": "$inputs.reference_class_names",
            }
        ],
    )
    columns_operations: Dict[str, List[AllOperationsType]] = Field(
        description="Optional dictionary mapping column names to Query Language (UQL) operation definitions for transforming column data before CSV formatting. Keys must match column names defined in columns_data. Values are lists of UQL operations (e.g., DetectionsPropertyExtract to extract class names from detections, string operations, calculations) that transform the raw column data. Operations are applied in sequence to each column's data. If a column name is not in this dictionary, the data is used as-is without transformation. Example: {'predictions': [{'type': 'DetectionsPropertyExtract', 'property_name': 'class_name'}]} extracts class names from detection predictions.",
        examples=[
            {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            }
        ],
        default_factory=lambda: {},
    )

    @field_validator("columns_data", "columns_operations")
    @classmethod
    def protect_timestamp_column(cls, value: dict) -> dict:
        if "timestamp" in value:
            raise ValueError(
                "Attempted to register column with reserved name `timestamp`."
            )
        return value

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return ["columns_data"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="csv_content", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CSVFormatterBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        columns_data: Dict[str, Any],
        columns_operations: Dict[str, List[AllOperationsType]],
    ) -> BlockResult:
        has_batch_oriented_inputs = any(
            isinstance(v, Batch) for v in columns_data.values()
        )
        csv_rows = prepare_csv_content(
            batch_columns_data=columns_data,
            columns_operations=columns_operations,
        )
        csv_content = to_csv(data=csv_rows)
        if not has_batch_oriented_inputs:
            return {"csv_content": csv_content}
        batch_size = len(csv_rows)
        result: List[Dict[str, Any]] = [{"csv_content": None}] * (batch_size - 1)
        result.append({"csv_content": csv_content})
        return result


def prepare_csv_content(
    batch_columns_data: Dict[str, Any],
    columns_operations: Dict[str, List[AllOperationsType]],
) -> List[dict]:
    result = []
    for columns_data in unfold_parameters(batch_columns_data=batch_columns_data):
        for variable_name, operations in columns_operations.items():
            operations_chain = build_operations_chain(operations=operations)
            columns_data[variable_name] = operations_chain(
                columns_data[variable_name], global_parameters={}
            )
        columns_data["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        result.append(columns_data)
    return result


def unfold_parameters(
    batch_columns_data: Dict[str, Any],
) -> Generator[Dict[str, Any], None, None]:
    batch_parameters = {
        k for k, v in batch_columns_data.items() if isinstance(v, Batch)
    }
    non_batch_parameters = set(batch_columns_data.keys()).difference(batch_parameters)
    if len(batch_parameters) == 0:
        yield batch_columns_data
        return None
    max_batch_size = max(len(batch_columns_data[k]) for k in batch_parameters)
    aligned_batch_parameters = {
        k: batch_columns_data[k].broadcast(n=max_batch_size) for k in batch_parameters
    }
    non_batch_parameters = {k: batch_columns_data[k] for k in non_batch_parameters}
    for i in range(max_batch_size):
        result = {}
        for batch_parameter in batch_parameters:
            result[batch_parameter] = aligned_batch_parameters[batch_parameter][i]
        result.update(non_batch_parameters)
        yield result
    return None


def to_csv(data: List[Dict[str, Any]]) -> str:
    return pd.DataFrame(data).to_csv(index=False)
