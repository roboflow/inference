from typing import Dict, List, Optional, Type, Union

import pylogix
from pydantic import ConfigDict, Field
from typing_extensions import Literal

from inference.core.logger import logger
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
This **PLC Communication** block integrates a Roboflow Workflow with a PLC using Ethernet/IP communication.
It can:
- Read tags from a PLC if `mode='read'`.
- Write tags to a PLC if `mode='write'`.
- Perform both read and write in a single run if `mode='read_and_write'`.

**Parameters depending on mode:**
- If `mode='read'` or `mode='read_and_write'`, `tags_to_read` must be provided.
- If `mode='write'` or `mode='read_and_write'`, `tags_to_write` must be provided.

If a read or write operation fails, an error message is printed to the terminal, 
and the corresponding entry in the output dictionary is set to a generic "ReadFailure" or "WriteFailure" message.
"""


class PLCBlockManifest(WorkflowBlockManifest):
    """Manifest for a PLC communication block using Ethernet/IP.

    The block can be used in one of three modes:
    - 'read': Only reads specified tags.
    - 'write': Only writes specified tags.
    - 'read_and_write': Performs both reading and writing in one execution.

    `tags_to_read` and `tags_to_write` are applicable depending on the mode chosen.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "name": "PLC EthernetIP",
            "version": "v1",
            "short_description": "Generic PLC read/write block using pylogix over Ethernet/IP.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sinks",
        }
    )

    type: Literal["roboflow_core/sinks@v1"]

    plc_ip: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        description="IP address of the target PLC.", examples=["192.168.1.10"]
    )

    mode: Literal["read", "write", "read_and_write"] = Field(
        description="Mode of operation: 'read', 'write', or 'read_and_write'.",
        examples=["read", "write", "read_and_write"],
    )

    tags_to_read: Union[
        List[str], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default=[],
        description="List of PLC tag names to read. Applicable if mode='read' or mode='read_and_write'.",
        examples=[["camera_msg", "sku_number"]],
    )

    tags_to_write: Union[
        Dict[str, Union[int, float, str]],
        WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]),
    ] = Field(
        default={},
        description="Dictionary of tags and the values to write. Applicable if mode='write' or mode='read_and_write'.",
        examples=[{"camera_fault": True, "defect_count": 5}],
    )

    depends_on: Selector() = Field(
        description="Reference to the step output this block depends on.",
        examples=["$steps.some_previous_step"],
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
    """A PLC communication workflow block using Ethernet/IP and pylogix.

    Depending on the selected mode:
    - 'read': Reads specified tags.
    - 'write': Writes provided values to specified tags.
    - 'read_and_write': Reads and writes in one go.

    In case of failures, errors are printed to terminal and the corresponding tag entry in the output is set to "ReadFailure" or "WriteFailure".
    """

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PLCBlockManifest

    def _read_single_tag(self, comm, tag):
        try:
            response = comm.Read(tag)
            if response.Status == "Success":
                return response.Value
            logger.error(f"Error reading tag '%s': %s", tag, response.Status)
            return "ReadFailure"
        except Exception as e:
            logger.error(f"Unhandled error reading tag '%s': %s", tag, e)
            return "ReadFailure"

    def _write_single_tag(self, comm, tag, value):
        try:
            response = comm.Write(tag, value)
            if response.Status == "Success":
                return "WriteSuccess"
            logger.error(
                "Error writing tag '%s' with value '%s': %s",
                tag,
                value,
                response.Status,
            )
            return "WriteFailure"
        except Exception as e:
            logger.error(f"Unhandled error writing tag '%s': %s", tag, e)
            return "WriteFailure"

    def run(
        self,
        plc_ip: str,
        mode: str,
        tags_to_read: List[str],
        tags_to_write: Dict[str, Union[int, float, str]],
        depends_on: any,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> dict:
        """Run PLC read/write operations using pylogix over Ethernet/IP.

        Args:
            plc_ip (str): PLC IP address.
            mode (str): 'read', 'write', or 'read_and_write'.
            tags_to_read (List[str]): Tags to read if applicable.
            tags_to_write (Dict[str, Union[int, float, str]]): Tags to write if applicable.
            depends_on (any): The step output this block depends on.
            image (Optional[WorkflowImageData]): Not required for this block.
            metadata (Optional[VideoMetadata]): Not required for this block.

        Returns:
            dict: A dictionary with `plc_results` as a list containing one dictionary. That dictionary has 'read' and/or 'write' keys.
        """
        read_results = {}
        write_results = {}

        with pylogix.PLC() as comm:
            comm.IPAddress = plc_ip

            if mode in ["read", "read_and_write"]:
                read_results = {
                    tag: self._read_single_tag(comm, tag) for tag in tags_to_read
                }

            if mode in ["write", "read_and_write"]:
                write_results = {
                    tag: self._write_single_tag(comm, tag, value)
                    for tag, value in tags_to_write.items()
                }

        plc_output = {}
        if read_results:
            plc_output["read"] = read_results
        if write_results:
            plc_output["write"] = write_results

        return {"plc_results": [plc_output]}
