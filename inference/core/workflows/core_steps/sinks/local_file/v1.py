import os.path
from datetime import datetime
from typing import Literal, Union, List, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import StepOutputSelector, STRING_KIND, \
    WorkflowParameterSelector, BOOLEAN_KIND
from inference.core.workflows.prototypes.block import WorkflowBlockManifest, WorkflowBlock, BlockResult


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Local File Sink",
            "version": "v1",
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "sink",
        }
    )
    type: Literal["roboflow_core/local_file_sink@v1"]
    content: StepOutputSelector(kind=[STRING_KIND]) = Field(
        description="Content of the file to save",
    )
    target_directory: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Target directory",
    )
    file_name_prefix: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        default="workflow_output",
        description="File name prefix",
    )
    file_extension: Union[WorkflowParameterSelector(kind=[STRING_KIND]), Literal["csv", "json", "txt"]] = Field(
        default="csv",
        description="File name prefix",
    )

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

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["allow_data_store_in_file_system"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        content: str,
        target_directory: str,
        file_name_prefix: str,
        file_extension: str,
    ) -> BlockResult:
        try:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            file_name = f"{file_name_prefix}_{timestamp}_.{file_extension}"
            target_path = os.path.abspath(os.path.join(target_directory, file_name))
            with open(target_path, "w") as f:
                f.write(content)
            return {"error_status": False, "message": "Data saved successfully"}
        except Exception as error:
            return {"error_status": True, "message": str(error)}
