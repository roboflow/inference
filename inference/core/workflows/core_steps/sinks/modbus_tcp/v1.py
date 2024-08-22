from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field
from pymodbus.client import ModbusTcpClient as ModbusClient

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_BOOLEAN_KIND,
    BATCH_OF_CLASSIFICATION_PREDICTION_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_STRING_KIND,
    BOOLEAN_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    ImageInputField,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

TYPE: str = "roboflow_core/modbus_tcp@v1"
SHORT_DESCRIPTION = "Read and Write to a PLC using Modbus TCP"
LONG_DESCRIPTION = """
Block to read and write to a PLC using Modbus TCP. This block can be used to read and write to
holding registers in a PLC. The block can be configured to read or write to the PLC based off 
predictions from a model.
"""


class ModbusTCPManifest(WorkflowBlockManifest):
    type: Literal[f"{TYPE}", "ModbusTCP"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Modbus TCP",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "communication",
        }
    )

    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.instance_segmentation_model.predictions"],
    )

    comm_type: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Literal["Read", "Write"]
    ] = Field(
        description="Read or Write to PLC", examples=["Read", "$inputs.comm_type"]
    )

    plc_ip: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="IP Address for the PLC. Must be in format 'xxx.xxx.xxx.xxx'.",
        examples=["192.168.0.12", "$inputs.plc_ip"],
    )

    plc_port: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Port for the PLC. Must be in format 'xxxx'.",
        default=502,
        examples=["502", "$inputs.plc_port"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="status", kind=[STRING_KIND]),
        ]

    # @classmethod
    # def describe_outputs(cls) -> List[OutputDefinition]:
    #     return [
    #         OutputDefinition(name="error_status", kind=[BATCH_OF_BOOLEAN_KIND]),
    #         OutputDefinition(name="message", kind=[BATCH_OF_STRING_KIND]),
    #     ]


class ModbusTCPBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ModbusTCPManifest]:
        return ModbusTCPManifest

    def write_to_holding_register(
        self, client: ModbusClient, address: int, value: int
    ) -> None:
        """
        Write to a holding register in a PLC.

        Args:
            client (ModbusClient): The Modbus TCP client.
            address (int): The address of the register to write.
            value (int): The value to write to the register.
        """
        response = client.write_register(address, value)
        if not response.isError():
            print("Successfully wrote value to register.")
        else:
            print(f"Error writing to register: {response}")

    def read_holding_registers(
        self, client: ModbusClient, address: int, count: int
    ) -> None:
        """
        Read holding registers from a PLC.

        Args:
            client (ModbusClient): The Modbus TCP client.
            address (int): The starting address of the register to read.
            count (int): The number of registers to read.
        """
        response = client.read_holding_registers(address, count)
        if not response.isError():
            print(f"Register Values: {response.registers}")
            return response.registers
        else:
            print(f"Error reading registers: {response}")
            return -1

    def run(
        self,
        predictions: Batch[sv.Detections],
        comm_type: str,
        plc_ip: str,
        plc_port: str,
        *args,
        **kwargs,
    ) -> BlockResult:
        try:
            plc_ip = plc_ip
            plc_port = plc_port

            # Setup Modbus Client
            modbus_client = ModbusClient(plc_ip, port=plc_port)

            print("-" * 50)
            print(f"PLC IP: {plc_ip}")

            # Get prediction data
            prediction_data = predictions.data
            # Grab 'claas_name' from prediction_data
            class_names = prediction_data["class_name"]

            # Check if read or write
            if comm_type.lower() == "write":
                # TODO: HARD CODED
                # Set PLC register value based on the label (Only supports 1 hand right now)
                red_value = 0
                yellow_value = 0
                green_value = 0
                print(f"Class Names: {class_names}")
                for class_name in class_names:
                    if class_name.lower() == "rock":
                        red_value = 1
                    elif class_name.lower() == "paper":
                        yellow_value = 1
                    elif class_name.lower() == "scissors":
                        green_value = 1

                print(f"Red: {red_value}")
                print(f"Yellow: {yellow_value}")
                print(f"Green: {green_value}")

                # Write to the PLC
                # Red
                self.write_to_holding_register(modbus_client, 0000, red_value)
                # Yellow
                self.write_to_holding_register(modbus_client, 1, yellow_value)
                # Green
                self.write_to_holding_register(modbus_client, 2, green_value)

                return_string = "Successfully wrote to PLC."

            elif comm_type.lower() == "read":
                # Read from the PLC
                # Loop through and read all 3 registers
                red_value = self.read_holding_registers(modbus_client, 0, 1)
                yellow_value = self.read_holding_registers(modbus_client, 1, 1)
                green_value = self.read_holding_registers(modbus_client, 2, 1)
                # Print the values
                print(f"Red: {red_value}")
                print(f"Yellow: {yellow_value}")
                print(f"Green: {green_value}")
                # Concat all values into a string
                combined_string = (
                    f"Red: {red_value}, Yellow: {yellow_value}, Green: {green_value}"
                )
                return_string = combined_string

            print("-" * 50)
            print(f"status: {return_string}")

            return {"status": return_string}

        except Exception as e:
            print("-" * 50)
            print(f"FAILED: {e}")

            return {"status": "False", "error": str(e)}
