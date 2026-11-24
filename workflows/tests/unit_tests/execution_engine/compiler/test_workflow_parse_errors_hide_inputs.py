import pytest
from roboflow_workflows.errors import WorkflowSyntaxError
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.introspection import blocks_loader

ENTERPRISE_PLUGIN = "roboflow_workflows.enterprise_blocks.loader"
# Short on purpose: pydantic truncates a long rendered input in the middle, so a
# secret is shown when it sits at either end of the step (first or last field).
SECRETS = ("Qz7u", "Xk9p")


@pytest.fixture
def enterprise_blocks(monkeypatch):
    # the plugin list is read at load time: clear the caches on both sides
    monkeypatch.setenv("WORKFLOWS_PLUGINS", ENTERPRISE_PLUGIN)
    blocks_loader.clear_caches()
    yield
    blocks_loader.clear_caches()


def compile_and_capture(step: dict) -> WorkflowSyntaxError:
    definition = {
        "version": "1.0",
        "inputs": [],
        "steps": [step],
        "outputs": [{"type": "JsonField", "name": "out", "selector": "$steps.step.*"}],
    }
    with pytest.raises(WorkflowSyntaxError) as error:
        ExecutionEngine.init(workflow_definition=definition, init_parameters={})
    return error.value


def assert_no_secret(error: WorkflowSyntaxError) -> None:
    rendered = [
        str(error),
        repr(error),
        str(error.inner_error),
        repr(error.inner_error),
        str(error.blocks_errors),
    ]
    for text in rendered:
        assert "input_value" not in text
        for secret in SECRETS:
            assert secret not in text


def test_credentials_in_an_opc_reader_url_are_not_echoed(enterprise_blocks):
    error = compile_and_capture(
        {
            # first, where a rendered input would show it
            "url": "opc.tcp://Qz7u:Xk9p@h",
            "type": "roboflow_enterprise/opc_reader@v1",
            "name": "step",
            "namespace": "2",
            "object_name": "Line1",
            "variable_name": "LineRunning",
        }
    )

    assert_no_secret(error)
    # the sanitized reason still reaches the caller
    [block_error] = error.blocks_errors
    assert block_error.block_type == "roboflow_enterprise/opc_reader@v1"
    assert "credentials in the url are not accepted" in block_error.property_details
    assert "credentials in the url are not accepted" in str(error.inner_error)


def test_literal_password_of_an_invalid_step_is_not_echoed(enterprise_blocks):
    # any validation error renders the step it failed on; the password is a literal
    # field of the same step
    error = compile_and_capture(
        {
            "user_name": "Qz7u",
            "type": "roboflow_enterprise/opc_writer_sink@v1",
            "name": "step",
            "url": "opc.tcp://host:4840/server",
            "namespace": "2",
            "object_name": "Line1",
            "variable_name": "LineRunning",
            "value": True,
            "value_type": "NotAType",
            # last, where a rendered input would show it
            "password": "Xk9p",
        }
    )

    assert_no_secret(error)
    [block_error] = error.blocks_errors
    assert block_error.property_name == "value_type"
