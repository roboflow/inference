import json
import logging
import os.path
from datetime import datetime
from typing import Any, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, field_validator

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    STRING_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The **Local File Sink** block saves workflow data as files on a local file system. It allows users to configure how 
the data is stored, either:
 
* aggregating **multiple entries into a single file**
  
* or saving **each entry as a separate file**. 

This block is useful for logging, data export, or preparing files for subsequent processing.

### File Content, File Type and Output Mode

`content` is expected to be the output from another block producing string values of specific types
denoted by `file_type`.

`output_mode` set into `append_log` will make the block appending single file with consecutive entries
passed to `content` input up to `max_entries_per_file`. In this mode it is important that 

!!! Note "`file_type` in `append_log` mode"

    Contrary to `separate_files` output mode, `append_log` mode may introduce subtle changes into
    the structure of the `content` to properly append it into existing file, hence setting proper
    `file_type` is crucial:
    
    * **`file_type=json`**: in `append_log` mode, the block will create `*.jsonl` file in 
    [JSON Lines](https://jsonlines.org/) format - for that to be possible, each JSON document
    will be parsed and dumped to ensure that it will fit into single line.
    
    * **`file_type=csv`**: in `append_log` mode, the block will deduct the first line from the 
    content (making it **required for CSV content to always be shipped with header row**) of 
    consecutive updates into the content of already created file.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Local File Sink",
            "version": "v1",
            "short_description": "Saves data into local file",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
        }
    )
    type: Literal["roboflow_core/local_file_sink@v1"]
    content: StepOutputSelector(kind=[STRING_KIND]) = Field(
        description="Content of the file to save",
    )
    file_type: Literal["csv", "json", "txt"] = Field(
        default="csv",
        description="Type of the file",
    )
    output_mode: Literal["append_log", "separate_files"] = Field(
        description="Decides how to organise the content of the file",
        json_schema_extra={
            "values_metadata": {
                "append_log": {
                    "name": "Append Log",
                    "description": "Aggregates multiple documents in single file",
                },
                "custom": {
                    "name": "Separate File",
                    "description": "Outputs single document for each input datapoint",
                },
            }
        },
    )
    target_directory: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Target directory",
    )
    file_name_prefix: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        default="workflow_output",
        description="File name prefix",
    )
    max_entries_per_file: Union[int, WorkflowParameterSelector(kind=[STRING_KIND])] = (
        Field(
            default=1024,
            description="Defines how many datapoints can be appended to a single file",
            json_schema_extra={
                "relevant_for": {
                    "output_mode": {
                        "values": ["append_log"],
                        "required": True,
                    },
                }
            },
        )
    )

    @field_validator("max_entries_per_file")
    @classmethod
    def ensure_receiver_email_is_not_an_empty_list(cls, value: Any) -> dict:
        if isinstance(value, int) and value < 1:
            raise ValueError("`max_entries_per_file` cannot be lower than 1.")
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class LocalFileSinkBlockV1(WorkflowBlock):

    def __init__(self, allow_data_store_in_file_system: bool):
        if not allow_data_store_in_file_system:
            raise RuntimeError(
                "`roboflow_core/local_file_sink@v1` block cannot run in this environment - "
                "local file system usage is forbidden."
            )
        self._active_file: Optional[str] = None
        self._entries_in_file = 0

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["allow_data_store_in_file_system"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        content: str,
        file_type: Literal["csv", "json", "txt"],
        output_mode: Literal["append_log", "separate_files"],
        target_directory: str,
        file_name_prefix: str,
        max_entries_per_file: int,
    ) -> BlockResult:
        if output_mode == "separate_files":
            target_path = generate_new_file_path(
                target_directory=target_directory,
                file_name_prefix=file_name_prefix,
                file_type=file_type,
            )
            return handle_content_saving(
                target_path=target_path,
                file_operation_mode="w",
                content=content,
            )
        if file_type == "json":
            try:
                content = dump_json_inline(content=content)
            except Exception as error:
                logging.warning(f"Could not process JSON file in append mode: {error}")
                return {"error_status": True, "message": "Invalid JSON content"}
        if self._active_file is None or self._entries_in_file >= max_entries_per_file:
            extension = file_type if file_type != "json" else "jsonl"
            self._active_file = generate_new_file_path(
                target_directory=target_directory,
                file_name_prefix=file_name_prefix,
                file_type=extension,
            )
            self._entries_in_file = 0
        elif file_type == "csv":
            content = deduct_csv_header(content=content)
        result = handle_content_saving(
            target_path=self._active_file,
            file_operation_mode="a",
            content=content,
        )
        if not result["error_status"]:
            self._entries_in_file += 1
        return result


def generate_new_file_path(
    target_directory: str, file_name_prefix: str, file_type: str
) -> str:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{file_name_prefix}_{timestamp}.{file_type}"
    return os.path.abspath(os.path.join(target_directory, file_name))


def handle_content_saving(
    target_path: str,
    file_operation_mode: str,
    content: str,
) -> dict:
    try:
        save_to_file(path=target_path, mode=file_operation_mode, content=content)
        return {"error_status": False, "message": "Data saved successfully"}
    except Exception as error:
        logging.warning(f"Could not save local file: {error}")
        return {"error_status": True, "message": str(error)}


def save_to_file(path: str, mode: str, content: str) -> None:
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    if not content.endswith("\n"):
        content = f"{content}\n"
    with open(path, mode=mode) as f:
        f.write(content)


def deduct_csv_header(content: str) -> str:
    return "\n".join(list(content.split("\n"))[1:])


def dump_json_inline(content: str) -> str:
    parsed_content = json.loads(content)
    return json.dumps(parsed_content)
