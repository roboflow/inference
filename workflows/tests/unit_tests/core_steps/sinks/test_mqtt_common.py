import logging
from unittest.mock import MagicMock

import pytest
from roboflow_workflows.enterprise_blocks.sinks import mqtt_common
from roboflow_workflows.enterprise_blocks.sinks.mqtt_common import (
    DEFAULT_MQTT_PORT,
    MQTT_KEEPALIVE_SECONDS,
    ConfigurationError,
    configure_tls,
    normalise_broker_address,
    resolve_broker_address,
    split_host_port,
)

ALLOW = "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST"
ALLOWLIST = "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS"


@pytest.fixture
def policy(monkeypatch):
    def configure(allow: bool = True, allowlist=None):
        monkeypatch.setattr(mqtt_common, ALLOW, allow)
        monkeypatch.setattr(mqtt_common, ALLOWLIST, allowlist)

    configure()
    return configure


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("broker.local", ("broker.local", None)),
        (" Broker.LOCAL ", ("broker.local", None)),
        ("broker.local:1883", ("broker.local", 1883)),
        ("Broker.LOCAL:8883 ", ("broker.local", 8883)),
        ("10.0.0.5", ("10.0.0.5", None)),
        ("[::1]", ("::1", None)),
        ("[::1]:8883", ("::1", 8883)),
        ("fe80::1", ("fe80::1", None)),
        ("broker.local:notaport", ("broker.local:notaport", None)),
    ],
)
def test_split_host_port(entry, expected):
    assert split_host_port(entry) == expected


@pytest.mark.parametrize(
    "host, port, expected",
    [
        ("Broker.LOCAL", 1883, "broker.local:1883"),
        ("[::1]", 8883, "::1:8883"),
        (" 10.0.0.5 ", "1883", "10.0.0.5:1883"),
    ],
)
def test_normalise_broker_address(host, port, expected):
    assert normalise_broker_address(host, port) == expected


def test_keepalive_is_shorter_than_paho_default():
    assert 0 < MQTT_KEEPALIVE_SECONDS < 60


def test_default_policy_passes_user_value_through_unchanged(policy):
    assert resolve_broker_address("Broker.LOCAL", 1883) == ("Broker.LOCAL", 1883)


@pytest.mark.parametrize(
    "host, port",
    [
        ("broker.local", 1883),
        ("Broker.LOCAL", 1883),
        ("[::1]", 8883),
        ("10.0.0.5", 1883),
        ("10.0.0.5", 8883),
    ],
)
def test_allowlist_accepts_listed_host_and_port_or_portless_host(policy, host, port):
    policy(allowlist=["broker.local:1883", "[::1]:8883", "10.0.0.5"])

    assert resolve_broker_address(host, port) == (host, port)


@pytest.mark.parametrize(
    "host, port",
    [
        ("other.local", 1883),
        ("broker.local", 8883),
        ("[::1]", 1883),
    ],
)
def test_allowlist_rejects_unlisted_address_without_revealing_the_allowlist(
    policy, host, port
):
    policy(allowlist=["broker.local:1883", "[::1]:8883", "10.0.0.5"])

    with pytest.raises(ConfigurationError) as error:
        resolve_broker_address(host, port)

    message = str(error.value)
    assert "not permitted" in message
    assert "10.0.0.5" not in message
    assert "broker.local" not in message or host == "broker.local"


def test_empty_allowlist_permits_nothing(policy):
    policy(allowlist=[])

    with pytest.raises(ConfigurationError):
        resolve_broker_address("broker.local", 1883)


def test_allowlist_ignores_blank_entries(policy):
    policy(allowlist=["  ", "broker.local", ""])

    assert resolve_broker_address("broker.local", 1883) == ("broker.local", 1883)


def test_user_host_not_allowed_uses_first_operator_entry(policy, caplog):
    policy(allow=False, allowlist=["Operator.Broker:8883", "other:1883"])

    with caplog.at_level(logging.WARNING, logger="inference"):
        resolved = resolve_broker_address("workflow.host", 1883)

    assert resolved == ("operator.broker", 8883)
    assert "replaced by the operator-configured broker" in caplog.text


def test_user_host_not_allowed_defaults_port(policy):
    policy(allow=False, allowlist=["operator.broker"])

    assert resolve_broker_address("workflow.host", 1) == (
        "operator.broker",
        DEFAULT_MQTT_PORT,
    )


def test_override_warning_can_be_silenced(policy, caplog):
    policy(allow=False, allowlist=["operator.broker"])

    with caplog.at_level(logging.WARNING, logger="inference"):
        resolve_broker_address("workflow.host", 1883, log_override=False)

    assert caplog.text == ""


def test_user_host_not_allowed_without_operator_broker_is_disabled(policy):
    policy(allow=False, allowlist=None)

    with pytest.raises(ConfigurationError, match="disabled"):
        resolve_broker_address("workflow.host", 1883)


class TestConfigureTLS:
    @pytest.mark.parametrize("ca_certificate_path", [None, "", "/ca.pem"])
    @pytest.mark.parametrize("allowed", [True, False])
    def test_disabled_tls_touches_nothing(self, ca_certificate_path, allowed):
        client = MagicMock()

        configure_tls(
            client,
            use_tls=False,
            ca_certificate_path=ca_certificate_path,
            allow_access_to_file_system=allowed,
        )

        assert client.method_calls == []

    @pytest.mark.parametrize("ca_certificate_path", [None, ""])
    def test_tls_without_ca_path_uses_system_trust_store(self, ca_certificate_path):
        client = MagicMock()

        configure_tls(
            client,
            use_tls=True,
            ca_certificate_path=ca_certificate_path,
            allow_access_to_file_system=False,
        )

        client.tls_set.assert_called_once_with()
        client.tls_insecure_set.assert_not_called()

    def test_ca_path_used_with_file_system_access(self):
        client = MagicMock()

        configure_tls(
            client,
            use_tls=True,
            ca_certificate_path="/etc/ssl/factory-ca.pem",
            allow_access_to_file_system=True,
        )

        client.tls_set.assert_called_once_with(ca_certs="/etc/ssl/factory-ca.pem")
        client.tls_insecure_set.assert_not_called()

    def test_ca_path_refused_without_file_system_access(self):
        client = MagicMock()

        with pytest.raises(ConfigurationError) as error:
            configure_tls(
                client,
                use_tls=True,
                ca_certificate_path="/etc/passwd",
                allow_access_to_file_system=False,
            )

        assert "ca_certificate_path" in str(error.value)
        assert "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE" in str(error.value)
        assert client.method_calls == []

    def test_unloadable_ca_bundle_reported(self):
        client = MagicMock()
        client.tls_set.side_effect = FileNotFoundError("no such file")

        with pytest.raises(
            ConfigurationError, match="could not load CA bundle"
        ) as error:
            configure_tls(
                client,
                use_tls=True,
                ca_certificate_path="/missing/ca.pem",
                allow_access_to_file_system=True,
            )

        assert "/missing/ca.pem" in str(error.value)
        assert "no such file" in str(error.value)

    def test_system_store_failure_reported(self):
        client = MagicMock()
        client.tls_set.side_effect = ValueError("bad context")

        with pytest.raises(ConfigurationError, match="TLS could not be configured"):
            configure_tls(
                client,
                use_tls=True,
                ca_certificate_path=None,
                allow_access_to_file_system=False,
            )
