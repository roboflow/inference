from typing import Dict, List, Optional, Type, Union
from pydantic import ConfigDict, Field
from typing_extensions import Literal

#TODO: Add to requirements.
import pylogix

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class PLCBlockManifest(WorkflowBlockManifest):
    """Manifest class for the PLC Communication Block.

    This specifies the parameters that the block needs:
    - plc_ip: The PLC IP address.
    - tags_to_read: A list of tag names to read from the PLC.
    - tags_to_write: A dictionary of tags and values to write to the PLC.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "name": "PLC Communication",
            "version": "v1",
            "short_description": "Block that reads/writes tags from/to a PLC using pylogix.",
            "long_description": "The PLCBlock allows reading and writing of tags from a PLC. "
                                "This can be used to integrate model results into a factory automation workflow.",
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )

    type: Literal["roboflow_core/plc_communication@v1"]

    plc_ip: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        description="IP address of the PLC",
        examples=["192.168.1.10"]
    )

    tags_to_read: Union[List[str], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])] = Field(
        default=[],
        description="List of PLC tags to read",
        examples=[["tag1", "tag2", "tag3"]]
    )

    tags_to_write: Union[Dict[str, Union[int, float, str]], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])] = Field(
        default={},
        description="Dictionary of PLC tags to write and their corresponding values",
        examples=[{"class_name": "car", "class_count": 5}]
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="plc_results",
                kind=[LIST_OF_VALUES_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class PLCBlockV1(WorkflowBlock):
    """A workflow block for PLC communication.

    This block:
    - Connects to a PLC using pylogix.
    - Reads specified tags from the PLC.
    - Writes specified values to the PLC.
    - Returns a dictionary containing read results and write confirmations.
    """

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PLCBlockManifest

    def run(
        self,
        plc_ip: str,
        tags_to_read: List[str],
        tags_to_write: Dict[str, Union[int, float, str]],
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> dict:
        """Connect to the PLC, read and write tags, and return the results.

        Args:
            plc_ip (str): The PLC IP address.
            tags_to_read (List[str]): Tag names to read from PLC.
            tags_to_write (Dict[str, Union[int, float, str]]): Key-value pairs of tags and values to write.
            image (Optional[WorkflowImageData]): Not required, included for framework compliance.
            metadata (Optional[VideoMetadata]): Not required, included for framework compliance.

        Returns:
            dict: A dictionary with 'plc_results' containing read and write results.
        """
        read_results = {}
        write_results = {}

        with pylogix.PLC() as comm:
            comm.IPAddress = plc_ip

            # Read tags
            for tag in tags_to_read:
                read_response = comm.Read(tag)
                if read_response.Status == "Success":
                    read_results[tag] = read_response.Value
                else:
                    read_results[tag] = f"ReadError: {read_response.Status}"

            # Write tags
            for tag, value in tags_to_write.items():
                write_response = comm.Write(tag, value)
                if write_response.Status == "Success":
                    write_results[tag] = "WriteSuccess"
                else:
                    write_results[tag] = f"WriteError: {write_response.Status}"

        return {
            "plc_results": {
                "read": read_results,
                "write": write_results
            }
        }
