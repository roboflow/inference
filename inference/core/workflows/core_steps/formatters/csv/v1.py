from copy import copy
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type, Union

import pandas as pd
from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    STRING_KIND,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector, INTEGER_KIND, BOOLEAN_KIND,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The **CSV Formatter** block allows you to format and output data as a CSV file within a Workflow. 
The configuration options provide flexibility in terms of how and when the CSV file is produced, 
as well as how data is processed and organized. 


### When the CSV file is produced?

CSV files usually contain multiple rows with data, whereas Workflow execution would usually 
produce one or few rows in the CSV file. It may be needed 

"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "CSV Formatter",
            "version": "v1",
            "short_description": "Creates CSV files with specified columns.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
        }
    )
    type: Literal["roboflow_core/csv_formatter@v1"]
    produces_csv_on: Literal["each_datapoint", "interval"] = Field(
        default="each_datapoint",
        description="Specifies how frequently block output CSV file is yielded.",
        json_schema_extra={
            "values_metadata": {
                "each_datapoint": {
                    "name": "Each Datapoint",
                    "description": "Produces CSV file each time new input data is provided. "
                                   "In this mode, rows are discarded when max size of CSV file is reached.",
                },
                "interval": {
                    "name": "Interval",
                    "description": "Produces CSV file on a specified time interval.",
                },
            }
        },
    )
    columns_data: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
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
    max_rows: Optional[Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])]] = Field(
        default=None,
        description="Specifies the maximum number of rows to keep in memory",
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "produces_csv_on": {
                    "values": ["each_datapoint"],
                    "required": True,
                },
            }
        },
    )
    flush_interval: Optional[Union[int, WorkflowParameterSelector(kind=INTEGER_KIND)]] = Field(
        default=None,
        description="Specifies the interval for producing CSV file in the output.",
        gt=0,
        json_schema_extra={
            "relevant_for": {
                "produces_csv_on": {
                    "values": ["interval"],
                    "required": True,
                },
            }
        },
    )
    interval_unit: Optional[Literal["seconds", "minutes", "hours"]] = Field(
        default=None,
        description="Specifies interval unit",
        json_schema_extra={
            "relevant_for": {
                "produces_csv_on": {
                    "values": ["interval"],
                    "required": True,
                },
            }
        },
    )

    @field_validator("columns_data", "columns_operations")
    @classmethod
    def protect_timestamp_column(cls, value: dict) -> dict:
        if "timestamp" in value:
            raise ValueError(
                "Attempted to register column with reserved name `timestamp`."
            )
        return value

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if self.output_production_mode == "on_each_datapoint" and self.max_rows is None:
            raise ValueError(
                "`max_rows` must be specified when `output_production_mode` is chosen to be `on_each_datapoint`."
            )
        if self.flush_interval is None or self.interval_unit is None:
            raise ValueError(
                "`flush_interval` and `interval_unit` must be specified when "
                "`output_production_mode` is chosen to be `on_interval`."
            )
        return self

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="csv_content", kind=[STRING_KIND]),
            OutputDefinition(name="flushed", kind=[BOOLEAN_KIND])
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


INTERVAL_UNIT_TO_SECONDS = {
    "seconds": 1,
    "minutes": 60,
    "hours": 60 * 60,
}


class CSVFormatterBlockV1(WorkflowBlock):

    def __init__(self):
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush_timestamp = datetime.now()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        produces_csv_on: Literal["each_datapoint", "interval"],
        columns_data: Dict[str, Any],
        columns_operations: Dict[str, List[AllOperationsType]],
        max_rows: Optional[int],
        flush_interval: Optional[int],
        interval_unit: Optional[Literal["seconds", "minutes", "hours"]],
    ) -> BlockResult:
        csv_row = prepare_csv_row(columns_data=columns_data, columns_operations=columns_operations)
        self._buffer.append(csv_row)
        if produces_csv_on == "each_datapoint":
            ensure_value_provided(value=max_rows, error_message="`max_rows` must be provided.")
            csv_content = to_csv(data=self._buffer)
            flushed = False
            if len(self._buffer) >= flush_interval:
                self._buffer = []
                self._last_flush_timestamp = datetime.now()
                flushed = True
            return {"csv_content": csv_content, "flushed": flushed}
        ensure_value_provided(value=flush_interval, error_message="`flush_interval` must be provided.")
        ensure_value_provided(value=interval_unit, error_message="`interval_unit` must be provided.")
        second_since_last_flush = datetime.now() - self._last_flush_timestamp
        interval_seconds = flush_interval * INTERVAL_UNIT_TO_SECONDS[interval_unit]
        if second_since_last_flush < interval_seconds:
            return {"csv_content": None, "flushed": False}
        csv_content = to_csv(data=self._buffer)
        self._buffer = []
        self._last_flush_timestamp = datetime.now()
        return {"csv_content": csv_content, "flushed": True}


def prepare_csv_row(
    columns_data: Dict[str, Any],
    columns_operations: Dict[str, List[AllOperationsType]],
) -> dict:
    columns_data = copy(columns_data)
    for variable_name, operations in columns_operations.items():
        operations_chain = build_operations_chain(operations=operations)
        columns_data[variable_name] = operations_chain(
            columns_data[variable_name], global_parameters={}
        )
    columns_data["timestamp"] = datetime.now().isoformat()
    return columns_data


def to_csv(data: List[Dict[str, Any]]) -> str:
    return pd.DataFrame(data).to_csv(index=False)


def ensure_value_provided(value: Optional[Any], error_message: str) -> None:
    if value is None:
        raise ValueError(error_message)
