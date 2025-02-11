import json
import logging
import os.path
from datetime import datetime
from io import TextIOWrapper
from typing import Any, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, field_validator

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    STRING_KIND,
    Selector,
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
    

!!! warning "Security considerations"

    The block has an ability to write to the file system. If you find this unintended in your system, 
    you can disable the block setting environmental variable `ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False`
    in the environment which host Workflows Execution Engine.
    
    If you want to **restrict** the directory which may be used to write data - set 
    environmental variable `WORKFLOW_BLOCKS_WRITE_DIRECTORY` to the absolute path of directory which you
    allow to be used.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Local File Sink",
            "version": "v1",
            "short_description": "Save data to a local file.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-file",
                "blockPriority": 3,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/local_file_sink@v1"]
    content: Selector(kind=[STRING_KIND]) = Field(
        description="Content of the file to save",
        examples=["$steps.csv_formatter.csv_content"],
    )
    file_type: Literal["csv", "json", "txt"] = Field(
        default="csv",
        description="Type of the file",
        examples=["csv"],
    )
    output_mode: Literal["append_log", "separate_files"] = Field(
        description="Decides how to organise the content of the file",
        examples=["append_log"],
        json_schema_extra={
            "values_metadata": {
                "append_log": {
                    "name": "Append Log",
                    "description": "Aggregates multiple documents in single file",
                },
                "separate_files": {
                    "name": "Separate File",
                    "description": "Outputs single document for each input datapoint",
                },
            }
        },
    )
    target_directory: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Target directory",
        examples=["some/location"],
    )
    file_name_prefix: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="workflow_output",
        description="File name prefix",
        examples=["my_file"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    max_entries_per_file: Union[int, Selector(kind=[STRING_KIND])] = Field(
        default=1024,
        description="Defines how many datapoints can be appended to a single file",
        examples=[1024],
        json_schema_extra={
            "relevant_for": {
                "output_mode": {
                    "values": ["append_log"],
                    "required": True,
                },
            }
        },
    )

    @field_validator("max_entries_per_file")
    @classmethod
    def ensure_max_entries_per_file_is_correct(cls, value: Any) -> Any:
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
        return ">=1.3.0,<2.0.0"


class LocalFileSinkBlockV1(WorkflowBlock):

    def __init__(
        self, allow_access_to_file_system: bool, allowed_write_directory: Optional[str]
    ):
        self._active_file_descriptor: Optional[TextIOWrapper] = None
        self._entries_in_file = 0
        self._allow_access_to_file_system = allow_access_to_file_system
        self._allowed_write_directory = allowed_write_directory

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["allow_access_to_file_system", "allowed_write_directory"]

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
        if not self._allow_access_to_file_system:
            raise RuntimeError(
                "`roboflow_core/local_file_sink@v1` block cannot run in this environment - "
                "local file system usage is forbidden - use self-hosted `inference` or "
                "Roboflow Dedicated Deployment."
            )
        self._verify_write_access_to_directory(target_directory=target_directory)
        if output_mode == "separate_files":
            return self._save_to_separate_file(
                content=content,
                file_type=file_type,
                target_directory=target_directory,
                file_name_prefix=file_name_prefix,
            )
        return self._append_to_file(
            content=content,
            file_type=file_type,
            target_directory=target_directory,
            file_name_prefix=file_name_prefix,
            max_entries_per_file=max_entries_per_file,
        )

    def _verify_write_access_to_directory(self, target_directory: str) -> None:
        if self._allowed_write_directory is None:
            return None
        if not path_is_within_specified_directory(
            path=target_directory,
            specified_directory=self._allowed_write_directory,
        ):
            raise ValueError(
                f"Requested file sink to save data in `{target_directory}` which is not sub-directory of "
                f"the location pointed to dump data from `roboflow_core/local_file_sink@v1` block. "
                f"Expected sub-directory of {self._allowed_write_directory}"
            )

    def _save_to_separate_file(
        self,
        content: str,
        file_type: Literal["csv", "json", "txt"],
        target_directory: str,
        file_name_prefix: str,
    ) -> BlockResult:
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

    def _append_to_file(
        self,
        content: str,
        file_type: Literal["csv", "json", "txt"],
        target_directory: str,
        file_name_prefix: str,
        max_entries_per_file: int,
    ) -> BlockResult:
        if file_type == "json":
            try:
                content = dump_json_inline(content=content)
            except Exception as error:
                logging.warning(f"Could not process JSON file in append mode: {error}")
                return {"error_status": True, "message": "Invalid JSON content"}
        if (
            self._active_file_descriptor is None
            or self._entries_in_file >= max_entries_per_file
        ):
            if self._active_file_descriptor is not None:
                self._active_file_descriptor.close()
                self._active_file_descriptor = None
                self._entries_in_file = 0
            try:
                self._active_file_descriptor = self._open_new_append_log_file(
                    file_type=file_type,
                    target_directory=target_directory,
                    file_name_prefix=file_name_prefix,
                )
                self._entries_in_file = 0
            except Exception as error:
                logging.warning(f"Could not create new sink file: {error}")
                return {
                    "error_status": True,
                    "message": "Could not create new sink file",
                }
        elif file_type == "csv":
            # deduct CSV headers only on append to existing sink
            content = deduct_csv_header(content=content)
        if not content.endswith("\n"):
            content = f"{content}\n"
        try:
            self._active_file_descriptor.write(content)
            self._active_file_descriptor.flush()
        except Exception as error:
            logging.warning(f"Could not append content to append log: {error}")
            return {
                "error_status": True,
                "message": "Could not append content to append log",
            }
        self._entries_in_file += 1
        return {"error_status": False, "message": "Data saved successfully"}

    def _open_new_append_log_file(
        self,
        file_type: Literal["csv", "json", "txt"],
        target_directory: str,
        file_name_prefix: str,
    ) -> TextIOWrapper:
        extension = file_type if file_type != "json" else "jsonl"
        file_path = generate_new_file_path(
            target_directory=target_directory,
            file_name_prefix=file_name_prefix,
            file_type=extension,
        )
        parent_dir = os.path.dirname(file_path)
        os.makedirs(parent_dir, exist_ok=True)
        return open(file_path, "w")

    def __del__(self):
        if self._active_file_descriptor is not None:
            self._active_file_descriptor.close()


def generate_new_file_path(
    target_directory: str, file_name_prefix: str, file_type: str
) -> str:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
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


def path_is_within_specified_directory(
    path: str,
    specified_directory: str,
) -> bool:
    absolute_path = normalize_directory_path(path=path)
    specified_directory = normalize_directory_path(path=specified_directory)
    return absolute_path.startswith(specified_directory)


def normalize_directory_path(path: str) -> str:
    absolute_path = os.path.abspath(path)
    if not absolute_path.endswith(os.sep):
        absolute_path = f"{absolute_path}{os.sep}"
    return absolute_path
