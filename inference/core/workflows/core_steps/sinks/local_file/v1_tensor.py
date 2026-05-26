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
Save workflow data as files on the local filesystem, supporting CSV, JSON, and text file formats with configurable output modes for aggregating multiple entries into single files or saving each entry separately, enabling persistent data storage, logging, and file-based data export.

## How This Block Works

This block writes string content from workflow steps to files on the local filesystem. The block:

1. Takes string content (from formatters, predictions, or other string-producing blocks) and file configuration as input
2. Validates filesystem access permissions (checks if local storage access is allowed based on environment configuration)
3. Verifies write permissions for the target directory (checks against allowed write directory restrictions if configured)
4. Selects the appropriate file saving strategy based on `output_mode`:
   - **Separate Files Mode**: Creates a new file for each input, generating unique filenames with timestamps
   - **Append Log Mode**: Appends content to an existing file (or creates a new one if needed), aggregating multiple entries
5. For **separate files mode**: Generates a unique file path using the target directory, file name prefix, file type, and a timestamp, then writes the content to the new file
6. For **append log mode**: 
   - Opens or creates a file based on the file name prefix and type
   - Applies format-specific handling for appending:
     - **CSV**: Removes the header row from subsequent appends (CSV content must include headers on first write)
     - **JSON**: Converts to JSONL (JSON Lines) format, parsing and re-serializing each JSON document to fit on a single line
     - **TXT**: Appends content directly with newlines
   - Tracks entry count and creates a new file when `max_entries_per_file` limit is reached
7. Creates parent directories if they don't exist
8. Writes content to the file (ensuring newline termination)
9. Returns error status and messages indicating save success or failure

The block supports two distinct storage strategies: separate files mode creates individual timestamped files for each input (useful for organizing outputs by execution), while append log mode aggregates multiple entries into continuous log files (useful for time-series data logging). The file path generation includes timestamps (format: `YYYY_MM_DD_HH_MM_SS_microseconds`) to ensure unique filenames and chronological organization. In append log mode, the block maintains file handles across executions and automatically handles file rotation when entry limits are reached.

## Requirements

**Local Filesystem Access**: This block requires write access to the local filesystem. Filesystem access can be controlled via environment variables:
- Set `ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False` to disable local file sink functionality (block will raise an error)
- Set `WORKFLOW_BLOCKS_WRITE_DIRECTORY` to an absolute path to restrict writes to a specific directory and its subdirectories only

**Note on Append Log Mode Format Handling**: 
- For CSV files in append mode, the content must include header rows on the first write; headers are automatically removed from subsequent appends
- For JSON files in append mode, files are saved with `.jsonl` extension in JSON Lines format (one JSON object per line)

## Common Use Cases

- **Data Logging and Audit Trails**: Save workflow execution data, detection results, or metrics to local log files (e.g., append CSV logs of detections, JSON logs of workflow outputs), enabling persistent logging and audit trails for production workflows
- **File-Based Data Export**: Export formatted workflow data to files for external processing (e.g., save CSV exports from CSV Formatter, JSON exports for downstream tools), enabling integration with file-based data processing pipelines
- **Time-Series Data Collection**: Aggregate workflow metrics over time into continuous log files (e.g., append CSV rows with timestamps, log detection counts per frame), creating persistent time-series datasets for analysis and reporting
- **Batch Result Storage**: Save individual results from batch processing workflows to separate files (e.g., save each image's detection results to separate JSON files), enabling organized storage of batch processing outputs with unique filenames
- **Data Archival**: Archive workflow outputs and results to local storage (e.g., save formatted reports, export analysis results), enabling long-term data retention and backup workflows
- **Integration with File-Based Systems**: Store workflow data in file formats compatible with external tools (e.g., save CSV for spreadsheet analysis, JSONL for data processing pipelines), enabling seamless data exchange with file-based systems

## Connecting to Other Blocks

This block receives string content from workflow steps and saves it to files:

- **After formatter blocks** (e.g., CSV Formatter) to save formatted data (CSV, JSON, or text) to files, enabling persistent storage of structured workflow outputs
- **After detection or analysis blocks** that output string-format data to save inference results, metrics, or analysis outputs to files for logging or archival
- **After data processing blocks** (e.g., Expression, Property Definition) that produce string outputs to save computed or transformed data to files
- **In logging workflows** to create persistent audit trails and logs of workflow executions, enabling record-keeping and debugging for production deployments
- **In batch processing workflows** where multiple data points need to be saved (either aggregated into log files or stored as separate files), enabling organized data collection and storage
- **Before external processing** where workflow data needs to be saved to files for consumption by external tools, scripts, or systems that read from filesystem storage
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
        description="String content to save as a file. This should be formatted data from other workflow blocks (e.g., CSV content from CSV Formatter, JSON strings, or plain text). The content format should match the specified file_type. For CSV files in append_log mode, content must include header rows on the first write.",
        examples=["$steps.csv_formatter.csv_content"],
    )
    file_type: Literal["csv", "json", "txt"] = Field(
        default="csv",
        description="Type of file to create: 'csv' (CSV format), 'json' (JSON format, or JSONL in append_log mode), or 'txt' (plain text). The content format should match this file type. In append_log mode, JSON files are saved as .jsonl (JSON Lines) format with one JSON object per line.",
        examples=["csv"],
    )
    output_mode: Literal["append_log", "separate_files"] = Field(
        description="File organization strategy: 'append_log' aggregates multiple content entries into a single file (useful for time-series logging, creates files that grow over time), or 'separate_files' creates a new file for each input (useful for organizing individual outputs, each file gets a unique timestamp-based filename). In append_log mode, the block handles format-specific appending (removes CSV headers, converts JSON to JSONL).",
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
        description="Directory path where files will be saved. Can be a relative or absolute path. Parent directories are created automatically if they don't exist. If WORKFLOW_BLOCKS_WRITE_DIRECTORY is set, this path must be a subdirectory of the allowed directory. Files are saved with filenames generated from file_name_prefix and timestamps.",
        examples=["some/location"],
    )
    file_name_prefix: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="workflow_output",
        description="Prefix used to generate filenames. Combined with a timestamp (format: YYYY_MM_DD_HH_MM_SS_microseconds) and file extension to create unique filenames like 'workflow_output_2024_10_18_14_09_57_622297.csv'. For append_log mode, new files are created when max_entries_per_file is reached, using this prefix with new timestamps.",
        examples=["my_file"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    max_entries_per_file: Union[int, Selector(kind=[STRING_KIND])] = Field(
        default=1024,
        description="Maximum number of entries (content appends) allowed per file in append_log mode. When this limit is reached, a new file is created with the same file_name_prefix and a new timestamp. Only applies when output_mode is 'append_log'. Must be at least 1. Use this to control file sizes and enable file rotation for long-running workflows.",
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
