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
The **CSV Formatter** block prepares structured CSV content based on specified data configurations within 
a workflow. It allows users to:

* choose which data appears as columns

* apply operations to transform the data within the block
 
* aggregate whole batch of data into single CSV document (see **Data Aggregation** section)

The generated CSV content can be used as input for other blocks, such as File Sink or Email Notifications.

### Defining columns

Use `columns_data` property to specify name of the columns and data sources. Defining UQL operations in 
`columns_operations` you can perform specific operation on each column.

!!! Note "Timestamp column"

    The block automatically adds `timestamp` column and this column name is reserved and cannot be used.
    
    The value of timestamp would be in the following format: `2024-10-18T14:09:57.622297+00:00`, values 
    **are scaled** to UTC time zone.


For example, the following definition
```
columns_data = {
    "predictions": "$steps.model.predictions",
    "reference": "$inputs.reference_class_names",
}
columns_operations = {
    "predictions": [
        {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
    ],
}
```

Will generate CSV content:
```csv
timestamp,predictions,reference
"2024-10-16T11:15:15.336322+00:00","['a', 'b', 'c']","['a', 'b']"
```

When applied on object detection predictions from a single image, assuming that `$inputs.reference_class_names`
holds a list of reference classes.

### Data Aggregation

The block may take input from different blocks, hence its behavior may differ depending on context:

* **data `batch_size=1`:** whenever single input is provided - block will provide the output as in the example above - 
CSV header will be placed in the first row, the second row will hold the data

* **data `batch_size>1`:** each datapoint will create one row in CSV document, but only the last batch element
will be fed with the aggregated output, leaving other batch elements' outputs empty

#### When should I expect `batch_size=1`?

You may expect `batch_size=1` in the following scenarios:

* **CSV Formatter** was connected to the output of block that only operates on one image and produces one prediction

* **CSV Formatter** was connected to the output of block that aggregates data for whole batch and produces single 
non-empty output (which is exactly the characteristics of **CSV Formatter** itself)

#### When should I expect `batch_size>1`?

You may expect `batch_size=1` in the following scenarios:

* **CSV Formatter** was connected to the output of block that produces single prediction for single image, but batch
of images were fed - then **CSV Formatter** will aggregate the CSV content and output it in the position of
the last batch element:

```
--- input_batch[0] ----> ┌───────────────────────┐ ---->  <Empty>
--- input_batch[1] ----> │                       │ ---->  <Empty>
        ...              │      CSV Formatter    │ ---->  <Empty>
        ...              │                       │ ---->  <Empty>           
--- input_batch[n] ----> └───────────────────────┘ ---->  {"csv_content": "..."}
```

!!! Note "Format of CSV document for `batch_size>1`"

    If the example presented above is applied for larger input batch sizes - the output document structure 
    would be as follows:
    
    ```csv
    timestamp,predictions,reference
    "2024-10-16T11:15:15.336322+00:00","['a', 'b', 'c']","['a', 'b']"
    "2024-10-16T11:15:15.436322+00:00","['b', 'c']","['a', 'b']"
    "2024-10-16T11:15:15.536322+00:00","['a', 'c']","['a', 'b']"
    ```
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
        description="References data to be used to construct each and every column",
        examples=[
            {
                "predictions": "$steps.model.predictions",
                "reference": "$inputs.reference_class_names",
            }
        ],
    )
    columns_operations: Dict[str, List[AllOperationsType]] = Field(
        description="UQL definitions of operations to be performed on defined data w.r.t. each column",
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
