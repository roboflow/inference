import unittest
import threading
import pytest
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1 import ModbusTCPBlockV1

@pytest.mark.timeout(5)
def test_successful_read_operation(fake_modbus_server):
    # given
    block = ModbusTCPBlockV1()
    expected_value = 123  # Example register value expected from the fake server

    # Configure the fake server to respond with the expected value for register 1000
    fake_modbus_server.set_register(1000, expected_value)
    fake_modbus_server.registers_to_expect_read = [1000]

    # Start the fake Modbus server in a separate thread
    server_thread = threading.Thread(target=fake_modbus_server.start)
    server_thread.start()

    # when
    result = block.run(
        plc_ip=fake_modbus_server.host,
        plc_port=fake_modbus_server.port,
        mode='read',
        tags_to_read=[1000],
        tags_to_write={},
        depends_on=None
    )

    server_thread.join(timeout=2)

    # then
    assert 'modbus_results' in result
    results = result['modbus_results'][0]
    assert 'read' in results
    read_results = results['read']
    assert read_results.get(1000) == expected_value
