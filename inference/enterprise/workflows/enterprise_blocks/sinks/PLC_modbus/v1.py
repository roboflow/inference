from typing import Dict, List, Optional, Type, Union

from pydantic import ConfigDict, Field
from pymodbus.client import ModbusTcpClient as ModbusClient
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
This **Modbus TCP** block integrates a Roboflow Workflow with a PLC using Modbus TCP.
It can:
- Read registers from a PLC if `mode='read'`.
- Write registers to a PLC if `mode='write'`.
- Perform both read and write in a single run if `mode='read_and_write'`.

**Parameters depending on mode:**
- If `mode='read'` or `mode='read_and_write'`, `registers_to_read` must be provided as a list of register addresses.
- If `mode='write'` or `mode='read_and_write'`, `registers_to_write` must be provided as a dictionary mapping register addresses to values.

If a read or write operation fails, an error message is printed to the terminal, 
and the corresponding entry in the output dictionary is set to "ReadFailure" or "WriteFailure".
"""


class ModbusTCPBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "PLC ModbusTCP",
            "version": "v1",
            "short_description": "Generic Modbus TCP read/write block using pymodbus.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )

    type: Literal["roboflow_core/modbus_tcp@v1"]

    plc_ip: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        description="IP address of the target PLC.", examples=["10.0.1.31"]
    )
    plc_port: int = Field(
        default=502,
        description="Port number for Modbus TCP communication.",
        examples=[502],
    )
    mode: Literal["read", "write", "read_and_write"] = Field(
        description="Mode of operation: 'read', 'write', or 'read_and_write'.",
        examples=["read", "write", "read_and_write"],
    )
    registers_to_read: Union[
        List[int], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default=[],
        description="List of register addresses to read. Applicable if mode='read' or 'read_and_write'.",
        examples=[[1000, 1001]],
    )
    registers_to_write: Union[
        Dict[int, int], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default={},
        description="Dictionary mapping register addresses to values to write. Applicable if mode='write' or 'read_and_write'.",
        examples=[{1005: 25}],
    )
    depends_on: Selector() = Field(
        description="Reference to the step output this block depends on.",
        examples=["$steps.some_previous_step"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="modbus_results", kind=[LIST_OF_VALUES_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class ModbusTCPBlockV1(WorkflowBlock):
    """A Modbus TCP communication block using pymodbus.

    Supports:
    - 'read': Reads specified registers.
    - 'write': Writes values to specified registers.
    - 'read_and_write': Reads and writes in one execution.

    On failures, errors are printed and marked as "ReadFailure" or "WriteFailure".
    """

    def __init__(self):
        self.client: Optional[ModbusClient] = None

    def __del__(self):
        if self.client:
            try:
                self.client.close()
            except Exception as exc:
                logger.debug("Failed to release modbus client: %s", exc)

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ModbusTCPBlockManifest

    def run(
        self,
        plc_ip: str,
        plc_port: int,
        mode: str,
        registers_to_read: List[int],
        registers_to_write: Dict[int, int],
        depends_on: any,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> dict:
        read_results = {}
        write_results = {}

        if not self.client:
            self.client: ModbusClient = ModbusClient(plc_ip, port=plc_port)
            if not self.client.connect():
                print("Failed to connect to PLC")
                return {"modbus_results": [{"error": "ConnectionFailure"}]}

        # If mode involves reading
        if mode in ["read", "read_and_write"]:
            for address in registers_to_read:
                try:
                    response = self.client.read_holding_registers(address)
                    if not response.isError():
                        read_results[address] = (
                            response.registers[0] if response.registers else None
                        )
                    else:
                        print(f"Error reading register {address}: {response}")
                        read_results[address] = "ReadFailure"
                except Exception as e:
                    print(f"Exception reading register {address}: {e}")
                    read_results[address] = "ReadFailure"

        # If mode involves writing
        if mode in ["write", "read_and_write"]:
            for address, value in registers_to_write.items():
                try:
                    response = self.client.write_register(address, value)
                    if not response.isError():
                        write_results[address] = "WriteSuccess"
                    else:
                        print(
                            f"Error writing register {address} with value {value}: {response}"
                        )
                        write_results[address] = "WriteFailure"
                except Exception as e:
                    print(
                        f"Exception writing register {address} with value {value}: {e}"
                    )
                    write_results[address] = "WriteFailure"

        modbus_output = {}
        if read_results:
            modbus_output["read"] = read_results
        if write_results:
            modbus_output["write"] = write_results

        return {"modbus_results": [modbus_output]}
