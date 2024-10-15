from copy import copy
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type, Union

import pandas as pd
from pydantic import ConfigDict, Field, field_validator

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
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Creates CSV files with specified columns aggregating data over specified interval. 
"""

SHORT_DESCRIPTION = "Creates CSV files with specified columns aggregating data over specified interval. "


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "CSV Formatter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
        }
    )
    type: Literal["roboflow_core/csv_formatter@v1"]
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
    interval: int = Field(
        description="Number of minutes to trigger output flush - 0 means on each prediction",
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
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="csv_content", kind=[STRING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class CSVFormatterBlockV1(WorkflowBlock):

    def __init__(self):
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush_timestamp = datetime.now()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        columns_data: Dict[str, Any],
        columns_operations: Dict[str, List[AllOperationsType]],
        interval: int,
    ) -> BlockResult:
        columns_data = copy(columns_data)
        for variable_name, operations in columns_operations.items():
            operations_chain = build_operations_chain(operations=operations)
            columns_data[variable_name] = operations_chain(
                columns_data[variable_name], global_parameters={}
            )
        columns_data["timestamp"] = datetime.now().isoformat()
        print("Appending column data", columns_data)
        self._buffer.append(columns_data)
        print("DEBUG", (datetime.now() - self._last_flush_timestamp).total_seconds())
        minutes_since_last_flush = (
            datetime.now() - self._last_flush_timestamp
        ).total_seconds() // 60
        if minutes_since_last_flush >= interval:
            csv_content = to_csv(data=self._buffer)
            self._buffer = []
            self._last_flush_timestamp = datetime.now()
            print("Flushing CSV content")
            return {"csv_content": csv_content}
        print(f"To flush: {interval -  minutes_since_last_flush} minutes")
        return {"csv_content": None}


def to_csv(data: List[Dict[str, Any]]) -> str:
    return pd.DataFrame(data).to_csv(index=False)
