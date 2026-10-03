import os
import subprocess
import sys
import textwrap
import types

import pytest

from inference_server import configuration
from inference_server.workflows import host

ENTERPRISE_PLUGIN = "roboflow_workflows.enterprise_blocks.loader"

_CLEARED_VARIABLES = {
    "WORKFLOWS_PLUGINS",
    "LOAD_ENTERPRISE_BLOCKS",
    "GCP_SERVERLESS",
    "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST",
    "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS",
}

_FAKE_PAHO = """
import sys, types
for _name in ("paho", "paho.mqtt", "paho.mqtt.client"):
    sys.modules[_name] = types.ModuleType(_name)
sys.modules["paho.mqtt.client"].Client = object
"""


def _run(code, **env):
    child_env = {k: v for k, v in os.environ.items() if k not in _CLEARED_VARIABLES}
    child_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", _FAKE_PAHO + textwrap.dedent(code)],
        env=child_env,
        capture_output=True,
        text=True,
    )


def test_gcp_serverless_setting_reaches_the_platform_configuration(monkeypatch):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    assert host.build_workflows_configuration().platform.gcp_serverless is True

    monkeypatch.setattr(configuration, "GCP_SERVERLESS", False)
    assert host.build_workflows_configuration().platform.gcp_serverless is False


def test_block_refused_on_serverless_is_refused_by_the_engine_environment():
    result = _run(
        """
        import inference_server.workflows.host
        from roboflow_workflows.environment import GCP_SERVERLESS
        from roboflow_workflows.enterprise_blocks.sinks.mqtt_reader.v1 import (
            MQTTReaderBlockV1,
        )

        assert GCP_SERVERLESS is True
        outputs = MQTTReaderBlockV1().run(host="broker.example", port=1883, topic="t")
        assert outputs["error_status"] is True, outputs
        assert outputs["error_message"] == (
            "MQTT Reader is not available on the Roboflow hosted platform"
        ), outputs
        """,
        GCP_SERVERLESS="true",
    )

    assert result.returncode == 0, result.stderr


def test_block_is_not_refused_on_serverless_when_the_flag_is_off():
    result = _run(
        """
        import inference_server.workflows.host
        from roboflow_workflows.environment import GCP_SERVERLESS

        assert GCP_SERVERLESS is False
        """,
        GCP_SERVERLESS="false",
    )

    assert result.returncode == 0, result.stderr


def test_mqtt_policy_defaults_to_the_legacy_values():
    result = _run(
        """
        from inference_server import configuration as c

        assert c.MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST is True
        assert c.MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS is None
        """,
    )

    assert result.returncode == 0, result.stderr


def test_mqtt_policy_is_read_from_the_environment_in_operator_order():
    result = _run(
        """
        from inference_server import configuration as c

        assert c.MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST is False
        assert c.MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS == (
            "zeta.example:1884", "alpha.example"
        ), c.MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS
        """,
        MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST="false",
        MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS=" zeta.example:1884 , alpha.example ,,",
    )

    assert result.returncode == 0, result.stderr


def test_mqtt_policy_reaches_the_engine_configuration(monkeypatch):
    monkeypatch.setattr(
        configuration, "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", False
    )
    monkeypatch.setattr(
        configuration,
        "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS",
        ("zeta.example:1884", "alpha.example"),
    )

    engine = host.build_workflows_configuration().engine

    assert engine.allow_mqtt_blocks_user_provided_host is False
    assert engine.mqtt_blocks_whitelisted_hosts == (
        "zeta.example:1884",
        "alpha.example",
    )


def test_mqtt_policy_default_reaches_the_engine_configuration():
    engine = host.build_workflows_configuration().engine

    assert engine.allow_mqtt_blocks_user_provided_host is True
    assert engine.mqtt_blocks_whitelisted_hosts is None


def test_broker_outside_the_allowlist_is_rejected_the_way_legacy_rejects_it():
    result = _run(
        """
        import inference_server.workflows.host
        from roboflow_workflows.enterprise_blocks.sinks.mqtt_common import (
            ConfigurationError,
            resolve_broker_address,
        )

        assert resolve_broker_address("broker.example", 1883) == (
            "broker.example", 1883
        )
        try:
            resolve_broker_address("other.example", 1883)
        except ConfigurationError as error:
            assert str(error) == (
                "Broker 'other.example' port 1883 is not permitted on this "
                "deployment: the operator restricts the MQTT brokers Workflow "
                "blocks may connect to."
            ), str(error)
        else:
            raise AssertionError("broker outside the allowlist was permitted")
        """,
        MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS="broker.example",
    )

    assert result.returncode == 0, result.stderr


def test_any_broker_is_allowed_by_default_as_in_legacy():
    result = _run(
        """
        import inference_server.workflows.host
        from roboflow_workflows.enterprise_blocks.sinks.mqtt_common import (
            resolve_broker_address,
        )

        assert resolve_broker_address("other.example", 1883) == ("other.example", 1883)
        """,
    )

    assert result.returncode == 0, result.stderr


def test_user_provided_broker_is_replaced_by_the_operator_broker():
    result = _run(
        """
        import inference_server.workflows.host
        from roboflow_workflows.enterprise_blocks.sinks.mqtt_common import (
            resolve_broker_address,
        )

        assert resolve_broker_address("other.example", 9999) == (
            "operator.example", 1883
        )
        """,
        MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST="false",
        MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS="operator.example",
    )

    assert result.returncode == 0, result.stderr


@pytest.fixture
def enterprise_flag(monkeypatch):
    monkeypatch.setattr(configuration, "LOAD_ENTERPRISE_BLOCKS", True)
    monkeypatch.setenv("WORKFLOWS_PLUGINS", "")
    monkeypatch.delenv("WORKFLOWS_PLUGINS")


def test_load_enterprise_blocks_defaults_to_off():
    result = _run(
        """
        from inference_server import configuration as c

        assert c.LOAD_ENTERPRISE_BLOCKS is False
        """,
    )

    assert result.returncode == 0, result.stderr


def test_load_enterprise_blocks_is_read_from_the_environment():
    result = _run(
        """
        from inference_server import configuration as c

        assert c.LOAD_ENTERPRISE_BLOCKS is True
        """,
        LOAD_ENTERPRISE_BLOCKS="True",
    )

    assert result.returncode == 0, result.stderr


def test_enterprise_plugin_is_prepended_to_an_unset_plugin_list(enterprise_flag):
    host.expand_enterprise_blocks_plugin()

    assert os.environ["WORKFLOWS_PLUGINS"] == ENTERPRISE_PLUGIN


def test_enterprise_plugin_is_prepended_before_custom_plugins(
    enterprise_flag, monkeypatch
):
    monkeypatch.setenv("WORKFLOWS_PLUGINS", "custom_a,custom_b")

    host.expand_enterprise_blocks_plugin()

    assert os.environ["WORKFLOWS_PLUGINS"] == f"{ENTERPRISE_PLUGIN},custom_a,custom_b"


def test_enterprise_plugin_expansion_drops_empty_entries(enterprise_flag, monkeypatch):
    monkeypatch.setenv("WORKFLOWS_PLUGINS", "")
    host.expand_enterprise_blocks_plugin()
    assert os.environ["WORKFLOWS_PLUGINS"] == ENTERPRISE_PLUGIN

    monkeypatch.setenv("WORKFLOWS_PLUGINS", ",custom_a,")
    host.expand_enterprise_blocks_plugin()
    assert os.environ["WORKFLOWS_PLUGINS"] == f"{ENTERPRISE_PLUGIN},custom_a"


def test_enterprise_plugin_expansion_is_idempotent(enterprise_flag):
    host.expand_enterprise_blocks_plugin()
    host.expand_enterprise_blocks_plugin()

    assert os.environ["WORKFLOWS_PLUGINS"].split(",") == [ENTERPRISE_PLUGIN]


def test_enterprise_plugin_already_listed_keeps_its_position(
    enterprise_flag, monkeypatch
):
    monkeypatch.setenv("WORKFLOWS_PLUGINS", f"custom_a,{ENTERPRISE_PLUGIN}")

    host.expand_enterprise_blocks_plugin()

    assert os.environ["WORKFLOWS_PLUGINS"] == f"custom_a,{ENTERPRISE_PLUGIN}"


def test_plugin_list_is_untouched_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(configuration, "LOAD_ENTERPRISE_BLOCKS", False)
    monkeypatch.setenv("WORKFLOWS_PLUGINS", "custom_a")

    host.expand_enterprise_blocks_plugin()

    assert os.environ["WORKFLOWS_PLUGINS"] == "custom_a"


def test_enterprise_blocks_are_loaded_by_the_engine_when_the_flag_is_on(
    enterprise_flag, monkeypatch
):
    from roboflow_workflows.core_steps.transformations.dynamic_crop.v1 import (
        DynamicCropBlockV1,
    )
    from roboflow_workflows.execution_engine.introspection import blocks_loader

    fake_loader = types.ModuleType(ENTERPRISE_PLUGIN)
    fake_loader.load_blocks = lambda: [DynamicCropBlockV1]
    monkeypatch.setitem(sys.modules, ENTERPRISE_PLUGIN, fake_loader)

    host.expand_enterprise_blocks_plugin()
    host.require_enterprise_blocks_plugin()

    assert blocks_loader.get_plugin_modules() == [ENTERPRISE_PLUGIN]
    loaded = blocks_loader.load_workflow_blocks()
    assert [
        block.block_class for block in loaded if block.block_source == ENTERPRISE_PLUGIN
    ] == [DynamicCropBlockV1]


def test_enterprise_check_passes_when_the_loader_imports(enterprise_flag, monkeypatch):
    monkeypatch.setitem(sys.modules, ENTERPRISE_PLUGIN, types.ModuleType("loader"))

    host.require_enterprise_blocks_plugin()


def test_enterprise_check_names_the_extra_when_the_loader_cannot_be_imported(
    enterprise_flag, monkeypatch
):
    monkeypatch.setitem(sys.modules, ENTERPRISE_PLUGIN, None)

    with pytest.raises(RuntimeError) as error:
        host.require_enterprise_blocks_plugin()

    assert "LOAD_ENTERPRISE_BLOCKS" in str(error.value)
    assert "roboflow-workflows[enterprise]" in str(error.value)


def test_enterprise_check_is_skipped_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(configuration, "LOAD_ENTERPRISE_BLOCKS", False)
    monkeypatch.setitem(sys.modules, ENTERPRISE_PLUGIN, None)

    host.require_enterprise_blocks_plugin()


def test_server_fails_at_startup_when_the_enterprise_extra_is_missing():
    result = _run(
        """
        import sys
        sys.modules["roboflow_workflows.enterprise_blocks.loader"] = None
        import inference_server.workflows.host
        """,
        LOAD_ENTERPRISE_BLOCKS="true",
    )

    assert result.returncode != 0
    assert "roboflow-workflows[enterprise]" in result.stderr


def test_server_starts_and_lists_the_loader_first_when_the_flag_is_on():
    result = _run(
        """
        import os, sys, types
        fake = types.ModuleType("roboflow_workflows.enterprise_blocks.loader")
        fake.load_blocks = lambda: []
        sys.modules["roboflow_workflows.enterprise_blocks.loader"] = fake
        import inference_server.workflows.host
        assert os.environ["WORKFLOWS_PLUGINS"] == (
            "roboflow_workflows.enterprise_blocks.loader,custom_a"
        ), os.environ["WORKFLOWS_PLUGINS"]
        """,
        LOAD_ENTERPRISE_BLOCKS="true",
        WORKFLOWS_PLUGINS="custom_a",
    )

    assert result.returncode == 0, result.stderr


def test_lambda_setting_reaches_the_platform_configuration(monkeypatch):
    monkeypatch.setattr(configuration, "LAMBDA", True)
    assert host.build_workflows_configuration().platform.lambda_runtime is True

    monkeypatch.setattr(configuration, "LAMBDA", False)
    assert host.build_workflows_configuration().platform.lambda_runtime is False
