"""
Tests for enterprise block access control enforcement.

This module tests that enterprise-only workflow blocks can only be used
by accounts with enterprise subscriptions, enforcing the enterprise-only policy.
"""
import pytest

from inference.core.workflows.errors import WorkflowCompilerError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from unittest.mock import patch, MagicMock


# All enterprise blocks and their type identifiers
ENTERPRISE_BLOCKS = [
    ("mqtt_writer_sink@v1", "MQTT Writer"),
    ("opc_writer_sink@v1", "OPC Writer"),
    ("plc_ethernet_ip@v1", "PLC Ethernet/IP"),
    ("modbus_tcp@v1", "Modbus TCP"),
    ("microsoft_sql_server_sink@v1", "Microsoft SQL Server"),
]


def create_workflow_with_enterprise_block(block_type: str) -> dict:
    """
    Create a minimal workflow definition that uses the specified enterprise block.

    This helper creates a workflow that attempts to use an enterprise block,
    which should fail for non-enterprise accounts.
    """
    # Minimal workflow configurations for each block type
    block_configs = {
        "mqtt_writer_sink@v1": {
            "type": "mqtt_writer_sink@v1",
            "name": "mqtt_writer",
            "host": "localhost",
            "port": 1883,
            "topic": "test/topic",
            "message": "test message",
        },
        "opc_writer_sink@v1": {
            "type": "opc_writer_sink@v1",
            "name": "opc_writer",
            "opc_server_url": "opc.tcp://localhost:4840",
            "opc_node_id": "ns=2;s=TestNode",
            "value_to_write": "test",
        },
        "plc_ethernet_ip@v1": {
            "type": "plc_ethernet_ip@v1",
            "name": "plc_writer",
            "plc_ip": "192.168.1.1",
            "tag_name": "TestTag",
            "tag_value": 100,
        },
        "modbus_tcp@v1": {
            "type": "modbus_tcp@v1",
            "name": "modbus",
            "plc_ip": "192.168.1.1",
            "plc_port": 502,
            "mode": "read",
            "registers_to_read": [1000],
            "registers_to_write": {},
        },
        "microsoft_sql_server_sink@v1": {
            "type": "microsoft_sql_server_sink@v1",
            "name": "sql_writer",
            "db_host": "localhost",
            "db_port": 1433,
            "db_name": "test_db",
            "db_username": "user",
            "db_password": "pass",
            "query": "INSERT INTO test VALUES (?)",
            "query_parameters": [],
        },
    }

    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            block_configs[block_type]
        ],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.*"}
        ],
    }


@pytest.mark.parametrize("block_type,block_name", ENTERPRISE_BLOCKS)
@patch("inference.core.env.LOAD_ENTERPRISE_BLOCKS", True)
def test_enterprise_block_rejected_for_non_enterprise_account(
    block_type: str,
    block_name: str,
):
    """
    Test that enterprise blocks are rejected when used by non-enterprise accounts.

    This test ensures that when LOAD_ENTERPRISE_BLOCKS=True (blocks are available),
    the workflow compiler still rejects workflows containing enterprise blocks
    if the API key does not have enterprise access.

    This is the critical enforcement mechanism for the enterprise-only policy.
    """
    # Given: A workflow using an enterprise block
    workflow_definition = create_workflow_with_enterprise_block(block_type)

    # And: An API key with non-enterprise plan
    with patch("inference.usage_tracking.plan_details.PlanDetails.get_api_key_plan") as mock_get_plan:
        mock_get_plan.return_value = {
            "is_enterprise": False,  # Non-enterprise account
            "is_pro": True,
            "is_trial": False,
        }

        # When/Then: Attempting to compile the workflow should raise an error
        with pytest.raises(
            WorkflowCompilerError,
            match=f".*{block_name}.*enterprise.*"  # Error should mention the block and enterprise requirement
        ):
            ExecutionEngine.init(
                workflow_definition=workflow_definition,
                init_parameters={
                    "workflows_core.api_key": "test_api_key_non_enterprise",
                },
            )


@pytest.mark.parametrize("block_type,block_name", ENTERPRISE_BLOCKS)
@patch("inference.core.env.LOAD_ENTERPRISE_BLOCKS", True)
def test_enterprise_block_allowed_for_enterprise_account(
    block_type: str,
    block_name: str,
):
    """
    Test that enterprise blocks are allowed when used by enterprise accounts.

    This test ensures that valid enterprise customers can successfully compile
    workflows containing enterprise blocks.
    """
    # Given: A workflow using an enterprise block
    workflow_definition = create_workflow_with_enterprise_block(block_type)

    # And: An API key with enterprise plan
    with patch("inference.usage_tracking.plan_details.PlanDetails.get_api_key_plan") as mock_get_plan:
        mock_get_plan.return_value = {
            "is_enterprise": True,  # Enterprise account
            "is_pro": True,
            "is_trial": False,
        }

        # When: Compiling the workflow with enterprise credentials
        # Then: Should succeed without raising an error
        try:
            execution_engine = ExecutionEngine.init(
                workflow_definition=workflow_definition,
                init_parameters={
                    "workflows_core.api_key": "test_api_key_enterprise",
                },
            )
            # Compilation succeeded - test passes
            assert execution_engine is not None
        except WorkflowCompilerError as e:
            # If we get a compiler error, it should NOT be about enterprise access
            assert "enterprise" not in str(e).lower(), (
                f"Enterprise account should be allowed to use {block_name}, "
                f"but got error: {e}"
            )


@patch("inference.core.env.LOAD_ENTERPRISE_BLOCKS", True)
def test_enterprise_block_rejected_without_api_key():
    """
    Test that enterprise blocks are rejected when no API key is provided.

    Without an API key, we cannot verify enterprise status, so access should be denied.
    """
    # Given: A workflow using any enterprise block (using MQTT as example)
    workflow_definition = create_workflow_with_enterprise_block("mqtt_writer_sink@v1")

    # When/Then: Attempting to compile without an API key should raise an error
    with pytest.raises(
        WorkflowCompilerError,
        match=".*enterprise.*api.*key.*"
    ):
        ExecutionEngine.init(
            workflow_definition=workflow_definition,
            init_parameters={
                "workflows_core.api_key": None,  # No API key
            },
        )


@patch("inference.core.env.LOAD_ENTERPRISE_BLOCKS", False)
def test_enterprise_blocks_not_loaded_when_flag_disabled():
    """
    Test that enterprise blocks are not available when LOAD_ENTERPRISE_BLOCKS=False.

    This validates the existing behavior where the flag controls block loading.
    """
    # Given: LOAD_ENTERPRISE_BLOCKS=False (via patch)
    # And: A workflow attempting to use an enterprise block
    workflow_definition = create_workflow_with_enterprise_block("mqtt_writer_sink@v1")

    # When/Then: Should fail with "unknown block type" error, not enterprise access error
    with pytest.raises(
        WorkflowCompilerError,
        match=".*unknown.*block.*type.*"  # Block type not found
    ):
        ExecutionEngine.init(
            workflow_definition=workflow_definition,
            init_parameters={
                "workflows_core.api_key": "test_api_key",
            },
        )
