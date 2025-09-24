import unittest
from unittest.mock import MagicMock, patch

from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1 import (
    ModbusTCPBlockV1,
)


class TestModbusTCPBlockV1(unittest.TestCase):
    @patch(
        "inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1.ModbusClient"
    )
    def test_successful_read_mode(self, MockModbusClient):
        # Arrange
        mock_client = MagicMock()
        MockModbusClient.return_value = mock_client
        mock_client.connect.return_value = True

        # Simulate successful read for each register
        def fake_read(address):
            response = MagicMock()
            response.isError.return_value = False
            response.registers = [123]  # Sample value
            return response

        mock_client.read_holding_registers.side_effect = fake_read

        block = ModbusTCPBlockV1()

        # Act
        result = block.run(
            plc_ip="10.0.1.31",
            plc_port=502,
            mode="read",
            registers_to_read=[1000, 1001],
            registers_to_write={},
            depends_on=None,
        )

        # Assert
        self.assertIn("modbus_results", result)
        results = result["modbus_results"][0]
        self.assertIn("read", results)
        self.assertEqual(results["read"][1000], 123)
        self.assertEqual(results["read"][1001], 123)

    @patch(
        "inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1.ModbusClient"
    )
    def test_successful_write_mode(self, MockModbusClient):
        # Arrange
        mock_client = MagicMock()
        MockModbusClient.return_value = mock_client
        mock_client.connect.return_value = True

        def fake_write_register(address, value):
            response = MagicMock()
            response.isError.return_value = False
            return response

        mock_client.write_register.side_effect = fake_write_register

        block = ModbusTCPBlockV1()

        # Act
        result = block.run(
            plc_ip="10.0.0.205",
            plc_port=502,
            mode="write",
            registers_to_read=[],
            registers_to_write={1005: 25},
            depends_on=None,
        )

        # Assert
        self.assertIn("modbus_results", result)
        results = result["modbus_results"][0]
        self.assertIn("write", results)
        self.assertEqual(results["write"][1005], "WriteSuccess")

    @patch(
        "inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1.ModbusClient"
    )
    def test_connection_failure(self, MockModbusClient):
        # Arrange
        mock_client = MagicMock()
        MockModbusClient.return_value = mock_client
        mock_client.connect.return_value = False

        block = ModbusTCPBlockV1()

        # Act
        result = block.run(
            plc_ip="10.0.1.31",
            plc_port=502,
            mode="read",
            registers_to_read=[1000],
            registers_to_write={},
            depends_on=None,
        )

        # Assert
        self.assertIn("modbus_results", result)
        results = result["modbus_results"][0]
        self.assertIn("error", results)
        self.assertEqual(results["error"], "ConnectionFailure")

    @patch(
        "inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1.ModbusClient"
    )
    def test_read_failure(self, MockModbusClient):
        # Arrange
        mock_client = MagicMock()
        MockModbusClient.return_value = mock_client
        mock_client.connect.return_value = True

        def fake_read(address):
            response = MagicMock()
            response.isError.return_value = True
            return response

        mock_client.read_holding_registers.side_effect = fake_read

        block = ModbusTCPBlockV1()

        # Act
        result = block.run(
            plc_ip="10.0.1.31",
            plc_port=502,
            mode="read",
            registers_to_read=[1000],
            registers_to_write={},
            depends_on=None,
        )

        # Assert
        self.assertIn("modbus_results", result)
        results = result["modbus_results"][0]
        self.assertIn("read", results)
        self.assertEqual(results["read"][1000], "ReadFailure")

    @patch(
        "inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1.ModbusClient"
    )
    def test_write_failure(self, MockModbusClient):
        # Arrange
        mock_client = MagicMock()
        MockModbusClient.return_value = mock_client
        mock_client.connect.return_value = True

        def fake_write_register(address, value):
            response = MagicMock()
            response.isError.return_value = True
            return response

        mock_client.write_register.side_effect = fake_write_register

        block = ModbusTCPBlockV1()

        # Act
        result = block.run(
            plc_ip="10.0.1.31",
            plc_port=502,
            mode="write",
            registers_to_read=[],
            registers_to_write={1005: 25},
            depends_on=None,
        )

        # Assert
        self.assertIn("modbus_results", result)
        results = result["modbus_results"][0]
        self.assertIn("write", results)
        self.assertEqual(results["write"][1005], "WriteFailure")


if __name__ == "__main__":
    unittest.main()
