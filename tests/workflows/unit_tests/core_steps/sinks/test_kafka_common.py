import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from inference.enterprise.workflows.enterprise_blocks.sinks import kafka_common
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_common import (
    ClientErrorTracker,
    ConfigurationError,
    build_connection_config,
    combine_messages,
    describe_client_error,
    normalise_bootstrap_server,
    probe_topic_metadata,
    resolve_bootstrap_servers,
)

ALLOW_USER_SERVERS = "KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS"
WHITELISTED_SERVERS = "KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS"
TOPIC = "line1.state"
_REPO_ROOT = str(Path(__file__).resolve().parents[5])


def policy(allow_user_provided: bool, allowlist: Optional[List[str]]):
    first = patch.object(kafka_common, ALLOW_USER_SERVERS, allow_user_provided)
    second = patch.object(kafka_common, WHITELISTED_SERVERS, allowlist)

    class Both:
        def __enter__(self):
            first.__enter__()
            second.__enter__()

        def __exit__(self, *args):
            second.__exit__(*args)
            first.__exit__(*args)

    return Both()


# --------------------------------------------------------------------------------------
# Bootstrap server normalisation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("broker-1:9092", "broker-1:9092"),
        ("  Broker-1.Example.COM:9093 ", "broker-1.example.com:9093"),
        ("broker-1", "broker-1:9092"),
        ("broker-1:", "broker-1:9092"),
        ("broker-1:09092", "broker-1:9092"),
        ("10.0.0.5", "10.0.0.5:9092"),
        ("[::1]:9092", "[::1]:9092"),
        ("[::1]", "[::1]:9092"),
        ("[0:0:0:0:0:0:0:1]:9093", "[::1]:9093"),
        ("[FE80::A]:9092", "[fe80::a]:9092"),
        # an IPv6 literal without brackets cannot carry a port
        ("::1", "[::1]:9092"),
        ("fe80::a", "[fe80::a]:9092"),
    ],
)
def test_normalise_bootstrap_server(entry: str, expected: str) -> None:
    assert normalise_bootstrap_server(entry) == expected


@pytest.mark.parametrize("entry", ["[::1", "[::1]9092", "SSL://Broker-1:9092"])
def test_normalise_bootstrap_server_leaves_malformed_entries_unmatched(
    entry: str,
) -> None:
    # not an error here: such an entry just cannot equal a well-formed allowlist entry
    assert normalise_bootstrap_server(entry) == entry.lower()


# --------------------------------------------------------------------------------------
# resolve_bootstrap_servers
# --------------------------------------------------------------------------------------


def test_module_defaults_keep_todays_behaviour() -> None:
    # unless the test environment configures the policy, nothing is restricted
    if WHITELISTED_SERVERS in os.environ or ALLOW_USER_SERVERS in os.environ:
        pytest.skip("Kafka bootstrap policy is configured in this environment")
    assert getattr(kafka_common, ALLOW_USER_SERVERS) is True
    assert getattr(kafka_common, WHITELISTED_SERVERS) is None


def test_resolve_passes_user_value_through_without_allowlist() -> None:
    with policy(allow_user_provided=True, allowlist=None):
        assert resolve_bootstrap_servers("Broker-1:9092, b2") == "Broker-1:9092, b2"


@pytest.mark.parametrize(
    "user_value",
    [
        "broker-1:9092",
        "broker-1:9092,broker-2:9092",
        "BROKER-1:9092 ,  Broker-2",
        "broker-2:9092,,broker-1",
        "[::1]:9093,[0:0:0:0:0:0:0:1]:9093",
    ],
)
def test_resolve_accepts_user_value_when_every_entry_is_listed(
    user_value: str,
) -> None:
    with policy(True, ["broker-1", "Broker-2:9092", "[::1]:9093"]):
        # returned unchanged: the allowlist filters, it does not rewrite
        assert resolve_bootstrap_servers(user_value) == user_value


@pytest.mark.parametrize(
    "user_value, rejected",
    [
        ("evil.example.com:9092", "evil.example.com:9092"),
        ("broker-1:9092,evil.example.com:9092", "evil.example.com:9092"),
        ("broker-1:9093", "broker-1:9093"),
        ("broker-1.evil.example.com:9092", "broker-1.evil.example.com:9092"),
        ("PLAINTEXT://broker-1:9092", "PLAINTEXT://broker-1:9092"),
        ("[::2]:9093", "[::2]:9093"),
    ],
)
def test_resolve_rejects_unlisted_entry_without_revealing_the_allowlist(
    user_value: str, rejected: str
) -> None:
    with policy(True, ["broker-1:9092", "secret-internal-host:9092", "[::1]:9093"]):
        with pytest.raises(ConfigurationError) as error:
            resolve_bootstrap_servers(user_value)

    assert repr(rejected) in str(error.value)
    assert "secret-internal-host" not in str(error.value)


def test_resolve_rejects_everything_when_the_allowlist_is_empty() -> None:
    # a variable that is set but holds no entries is an allowlist of nothing
    with policy(True, []):
        with pytest.raises(ConfigurationError):
            resolve_bootstrap_servers("broker-1:9092")


def test_resolve_rejects_a_value_without_entries_when_allowlist_is_set() -> None:
    with policy(True, ["broker-1:9092"]):
        with pytest.raises(ConfigurationError):
            resolve_bootstrap_servers(" , ")


def test_resolve_replaces_user_value_with_operator_servers_in_stable_order() -> None:
    allowlist = ["Kafka-2:9092", "kafka-1", "kafka-2:9092", "kafka-1:9092", " "]
    with policy(False, allowlist), patch.object(
        kafka_common.logger, "warning"
    ) as warning:
        first = resolve_bootstrap_servers("evil.example.com:9092")
        second = resolve_bootstrap_servers("anything", log_override=False)

    # exactly what the operator wrote, in the operator's order; only blank entries go
    assert first == second == "Kafka-2:9092,kafka-1,kafka-2:9092,kafka-1:9092"
    assert warning.call_count == 1
    logged = warning.call_args.args[0] % warning.call_args.args[1:]
    assert "evil.example.com" not in logged


@pytest.mark.parametrize("allowlist", [None, [], ["", "  "]])
def test_resolve_disables_the_blocks_without_operator_servers(
    allowlist: Optional[List[str]],
) -> None:
    with policy(False, allowlist):
        with pytest.raises(ConfigurationError) as error:
            resolve_bootstrap_servers("broker-1:9092")

    assert "disabled on this deployment" in str(error.value)


def test_resolve_accepts_a_comma_separated_string_as_allowlist() -> None:
    # env.py parses the variable into a list; a raw string must not be read per char
    with policy(True, "broker-1:9092, broker-2:9092"):
        assert resolve_bootstrap_servers("broker-2:9092") == "broker-2:9092"


def _import_env(extra_env: Dict[str, str]) -> Dict[str, Any]:
    # env.py computes the values at import time: read them in a fresh interpreter
    env = {**os.environ}
    env.pop(ALLOW_USER_SERVERS, None)
    env.pop(WHITELISTED_SERVERS, None)
    env.update(extra_env)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import json; from inference.core import env; "
        f"print('RESULT' + json.dumps([env.{ALLOW_USER_SERVERS}, "
        f"env.{WHITELISTED_SERVERS}]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    line = [l for l in result.stdout.splitlines() if l.startswith("RESULT")][-1]
    allow_user_provided, allowlist = json.loads(line[len("RESULT") :])
    return {"allow_user_provided": allow_user_provided, "allowlist": allowlist}


def test_env_parses_the_allowlist_in_order_without_empty_entries() -> None:
    values = _import_env(
        {
            ALLOW_USER_SERVERS: "False",
            WHITELISTED_SERVERS: " kafka-2:9092, ,kafka-1:9092 ,kafka-2:9092,",
        }
    )

    # trimmed, blank entries dropped, otherwise exactly what the operator wrote
    assert values == {
        "allow_user_provided": False,
        "allowlist": ["kafka-2:9092", "kafka-1:9092", "kafka-2:9092"],
    }


# --------------------------------------------------------------------------------------
# build_connection_config: file system access for ssl_ca_location
# --------------------------------------------------------------------------------------


def connection_config(**overrides) -> Dict[str, Any]:
    kwargs = {
        "provider": "Self-hosted",
        "bootstrap_servers": "broker-1:9092",
        "username": None,
        "password": None,
        "aws_region": None,
        "ssl_ca_location": None,
        "auth_failures": [],
        "allow_access_to_file_system": False,
    }
    kwargs.update(overrides)
    return build_connection_config(**kwargs)


def test_ssl_ca_location_requires_file_system_access() -> None:
    with pytest.raises(ConfigurationError) as error:
        connection_config(ssl_ca_location="/etc/passwd")

    assert "ssl_ca_location" in str(error.value)
    assert "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE" in str(error.value)


def test_ssl_ca_location_is_used_with_file_system_access() -> None:
    config = connection_config(
        ssl_ca_location="/ca.pem", allow_access_to_file_system=True
    )

    assert config["ssl.ca.location"] == "/ca.pem"


@pytest.mark.parametrize("allowed", [True, False])
@pytest.mark.parametrize("ssl_ca_location", [None, ""])
def test_file_system_flag_has_no_effect_without_ssl_ca_location(
    allowed: bool, ssl_ca_location: Optional[str]
) -> None:
    config = connection_config(
        ssl_ca_location=ssl_ca_location, allow_access_to_file_system=allowed
    )

    assert config == {"security.protocol": "PLAINTEXT"}


# --------------------------------------------------------------------------------------
# ClientErrorTracker
# --------------------------------------------------------------------------------------


class FakeClientError:
    def __init__(self, text: str, name: Optional[str] = None, fatal: bool = False):
        self._text = text
        self._name = name
        self._fatal = fatal

    def name(self) -> Optional[str]:
        return self._name

    def code(self) -> int:
        return -187

    def fatal(self) -> bool:
        return self._fatal

    def __str__(self) -> str:
        return self._text


class FakeClient:
    def __init__(self):
        self.calls: List[float] = []
        self.failure: Optional[Exception] = None
        self.topic_error: Optional[str] = None
        self.known_topics = {TOPIC}

    def list_topics(self, topic: str, timeout: float):
        self.calls.append(timeout)
        if self.failure is not None:
            raise self.failure
        if topic not in self.known_topics:
            return SimpleNamespace(topics={})
        return SimpleNamespace(topics={topic: SimpleNamespace(error=self.topic_error)})


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock():
    fake_clock = FakeClock()
    # only kafka_common's view of `time`; nothing sleeps in these tests
    with patch.object(kafka_common, "time", fake_clock):
        yield fake_clock


def test_describe_client_error_adds_the_name_or_code_when_missing() -> None:
    assert (
        describe_client_error(FakeClientError("brokers down", name="_ALL_DOWN"))
        == "_ALL_DOWN: brokers down"
    )
    # KafkaError's own text already carries the name
    assert (
        describe_client_error(FakeClientError("code=_ALL_DOWN,val=-187", "_ALL_DOWN"))
        == "code=_ALL_DOWN,val=-187"
    )
    assert describe_client_error(FakeClientError("down")) == "-187: down"
    assert describe_client_error(RuntimeError("boom")) == "boom"
    assert describe_client_error(object()).startswith("<object object")


def test_combine_messages() -> None:
    assert combine_messages(None, None) is None
    assert combine_messages("a", None) == "a"
    assert combine_messages(None, "b") == "b"
    assert combine_messages("a", "b") == "a; b"


def test_tracker_starts_clean_and_never_probes(clock: FakeClock) -> None:
    tracker, client = ClientErrorTracker("test"), FakeClient()
    clock.now += 3600

    assert tracker.check(client, TOPIC) is None
    assert client.calls == []


def down_client() -> FakeClient:
    """A real outage: the broker does not answer the recovery probe."""
    client = FakeClient()
    client.failure = RuntimeError("Local: Broker transport failure")
    return client


def test_tracker_keeps_the_problem_until_proof_of_life(clock: FakeClock) -> None:
    tracker, client = ClientErrorTracker("test"), down_client()

    tracker(FakeClientError("brokers down", name="_ALL_BROKERS_DOWN"))
    clock.now += 2.0
    first = tracker.check(client, TOPIC)
    second = tracker.check(client, TOPIC)

    assert first is not None and first == second
    assert first.fatal is False
    assert first.recorded_at == 1000.0
    assert "brokers down" in first.message("The record may be stale.")
    assert "first seen 2.0s ago" in first.message()
    assert first.message("The record may be stale.").endswith("may be stale.")
    # the probe's own failure never replaces the recorded problem
    assert "Broker transport failure" not in first.message()
    # one failed probe; the second check is inside the probe interval
    assert len(client.calls) == 1

    tracker.clear()

    assert tracker.check(client, TOPIC) is None
    assert len(client.calls) == 1


def test_tracker_first_probe_is_immediate_and_later_probes_are_spaced(
    clock: FakeClock,
) -> None:
    tracker, client = ClientErrorTracker("test"), down_client()
    interval = kafka_common.CLIENT_RECOVERY_PROBE_INTERVAL

    # zero elapsed time between the error and the first check: probed at once
    tracker(FakeClientError("brokers down"))
    assert tracker.check(client, TOPIC) is not None
    assert client.calls == [kafka_common.CLIENT_RECOVERY_PROBE_TIMEOUT]

    # the failed first probe starts the spacing: nothing until the interval elapsed
    assert tracker.check(client, TOPIC) is not None
    clock.now += interval - 0.1
    assert tracker.check(client, TOPIC) is not None
    assert len(client.calls) == 1

    # a later error of the same streak neither probes at once nor resets the spacing
    tracker(FakeClientError("still down"))
    assert tracker.check(client, TOPIC) is not None
    assert len(client.calls) == 1

    clock.now += 0.1
    assert tracker.check(client, TOPIC) is not None
    assert len(client.calls) == 2

    clock.now += interval - 0.1
    tracker(FakeClientError("still down"))
    assert tracker.check(client, TOPIC) is not None
    assert len(client.calls) == 2

    clock.now += 0.1
    problem = tracker.check(client, TOPIC)
    assert len(client.calls) == 3
    # failing probes change nothing about the recorded problem
    assert problem.description == "-187: still down"
    assert problem.since == 1000.0
    assert problem.recorded_at == 1000.0 + 2 * interval - 0.1


def test_tracker_transient_error_on_a_healthy_cluster_is_never_reported(
    clock: FakeClock,
) -> None:
    tracker, client = ClientErrorTracker("test"), FakeClient()

    # zero elapsed time: the immediate probe succeeds, the same check reports nothing
    tracker(FakeClientError("broker-2:9092: Disconnected"))
    assert tracker.check(client, TOPIC) is None
    assert len(client.calls) == 1

    # healthy again: no further probes
    clock.now += 3600
    assert tracker.check(client, TOPIC) is None
    assert len(client.calls) == 1


def test_tracker_new_streak_after_a_clear_is_probed_at_once(clock: FakeClock) -> None:
    tracker, client = ClientErrorTracker("test"), FakeClient()

    # cleared by a successful probe, then a new streak with zero elapsed time
    tracker(FakeClientError("blip 1"))
    assert tracker.check(client, TOPIC) is None
    tracker(FakeClientError("blip 2"))
    assert tracker.check(client, TOPIC) is None
    assert len(client.calls) == 2

    # cleared by proof of life right after a FAILED probe: the spacing that probe
    # started belongs to the old streak, the new streak is probed at once
    client.failure = RuntimeError("Local: Broker transport failure")
    tracker(FakeClientError("outage"))
    assert tracker.check(client, TOPIC) is not None
    assert len(client.calls) == 3
    tracker.clear()
    tracker(FakeClientError("outage again"))
    problem = tracker.check(client, TOPIC)
    assert len(client.calls) == 4
    assert problem is not None and "outage again" in problem.description
    assert problem.since == clock.now


def test_tracker_recovery_is_found_by_the_first_probe_after_the_interval(
    clock: FakeClock,
) -> None:
    tracker, client = ClientErrorTracker("test"), down_client()
    tracker(FakeClientError("brokers down"))
    assert tracker.check(client, TOPIC) is not None

    client.failure = None
    # the broker is back, but the block only finds out when the next probe is due
    clock.now += kafka_common.CLIENT_RECOVERY_PROBE_INTERVAL - 0.1
    assert tracker.check(client, TOPIC) is not None
    clock.now += 0.1
    assert tracker.check(client, TOPIC) is None
    assert len(client.calls) == 2


@pytest.mark.parametrize("failure", ["raises", "topic_error", "topic_missing"])
def test_probe_topic_metadata_failures(failure: str) -> None:
    client = FakeClient()
    assert probe_topic_metadata(client, TOPIC) is True
    if failure == "raises":
        client.failure = RuntimeError("timed out")
    elif failure == "topic_error":
        client.topic_error = "Broker: Leader not available"
    else:
        client.known_topics = set()

    assert probe_topic_metadata(client, TOPIC) is False


def test_tracker_error_during_the_probe_is_not_cleared_by_it(clock: FakeClock) -> None:
    tracker = ClientErrorTracker("test")
    tracker(FakeClientError("brokers down"))

    class ClientReportingWhileProbed(FakeClient):
        def list_topics(self, topic: str, timeout: float):
            tracker(FakeClientError("down again"))
            return super().list_topics(topic, timeout)

    problem = tracker.check(ClientReportingWhileProbed(), TOPIC)

    assert problem is not None and "down again" in problem.description


def test_tracker_proof_of_life_needs_an_error_free_stretch(clock: FakeClock) -> None:
    # the probe keeps failing, so only the caller's proof of life can clear
    tracker, client = ClientErrorTracker("test"), down_client()
    tracker(FakeClientError("brokers down"))

    before = tracker.generation
    tracker(FakeClientError("down again"))
    assert tracker.check(client, TOPIC, alive_since_generation=before) is not None

    before = tracker.generation
    assert tracker.check(client, TOPIC, alive_since_generation=before) is None


def test_tracker_fatal_error_is_never_cleared_or_probed(clock: FakeClock) -> None:
    tracker, client = ClientErrorTracker("test"), FakeClient()

    tracker(FakeClientError("fenced", fatal=True))
    # not at once, as a non-fatal streak would be
    assert tracker.check(client, TOPIC) is not None
    assert client.calls == []
    tracker(FakeClientError("brokers down"))
    clock.now += 3600
    tracker.clear()
    problem = tracker.check(client, TOPIC, alive_since_generation=tracker.generation)

    assert problem is not None and problem.fatal is True
    assert "fenced" in problem.description
    assert "unusable" in problem.message()
    assert "Restart the block (pipeline)" in problem.message()
    assert client.calls == []

    # only a new client starts clean
    tracker.reset()
    assert tracker.check(client, TOPIC) is None


def test_tracker_callback_never_raises(clock: FakeClock) -> None:
    tracker = ClientErrorTracker("test")

    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("no text")

        def name(self):
            raise RuntimeError("no name")

        def fatal(self):
            raise RuntimeError("no fatal")

    for error in (object(), None, "plain text", Hostile()):
        tracker(error)

    problem = tracker.problem
    assert problem is not None and problem.fatal is False
    assert problem.description == "Hostile"
