"""Unit tests of the Kafka owners, helpers and blocks.

Every Kafka client here is ``FakeDriver``: an in-memory stand-in for the
``confluent_kafka`` module, written from V1's tests and the calls V1 makes.
It is not confluent-kafka or librdkafka, and no test opens a socket. Token
signers are stand-ins with invented credentials; no AWS code runs. A fake
clock replaces wall time, so no assertion depends on timing.
"""

import copy
import json
import logging
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from roboflow_workflows import environment
from roboflow_workflows.v2.blocks.industrial import kafka
from roboflow_workflows.v2.blocks.industrial.kafka import (
    SCHEDULED_MESSAGE,
    DeliveryStats,
    KafkaConfigurationError,
    KafkaConsumerBlock,
    KafkaConsumerCursor,
    KafkaHostError,
    KafkaProducerSinkBlock,
    KafkaPublish,
    KafkaPublisher,
    KafkaRead,
    coerce_acks,
    derive_msk_region,
    encode_headers,
    encode_key,
    encode_value,
    kafka_connection,
    msk_token_callback,
    normalise_bootstrap_server,
    pop_auth_failure_message,
    resolve_bootstrap_servers,
)
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

BOOTSTRAP = "broker-1:9092,broker-2:9092"
TOPIC = "line1.state"
PASSWORD = "pw-SENTINEL-kafka-71c4"
OLD_PASSWORD = "pw-OLD-SENTINEL-kafka-5e2a"
MSK_BOOTSTRAP = "b-1.demo.abc123.c2.kafka.us-east-1.amazonaws.com:9098"
WAIT = 5


# --- in-memory stand-in for the confluent_kafka module -------------------------


class FakeClock:
    """Monotonic time that moves only when a fake client waits or sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TopicPartition:
    def __init__(self, topic: str, partition: int, offset: int = -1001) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset


class KafkaError:
    """Stand-in for ``confluent_kafka.KafkaError``."""

    _INVALID_ARG = -186

    def __init__(
        self,
        code: int,
        text: str = "Broker: Unknown topic or partition",
        *,
        name: Optional[str] = None,
        fatal: bool = False,
    ) -> None:
        self._code = code
        self._text = text
        self._name = name
        self._fatal = fatal

    def code(self) -> int:
        return self._code

    def name(self) -> Optional[str]:
        return self._name

    def fatal(self) -> bool:
        return self._fatal

    def __str__(self) -> str:
        return self._text


def brokers_down(fatal: bool = False) -> KafkaError:
    return KafkaError(
        -187, "2/2 brokers are down", name="_ALL_BROKERS_DOWN", fatal=fatal
    )


class Message:
    def __init__(
        self,
        topic: str,
        partition: int,
        offset: Optional[int],
        key: Any = None,
        value: Any = None,
        error: Any = None,
    ) -> None:
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._key = key
        self._value = value
        self._error = error

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> Optional[int]:
        return self._offset

    def key(self) -> Any:
        return self._key

    def value(self) -> Any:
        return self._value

    def error(self) -> Any:
        return self._error


class Broker:
    """Per-partition logs plus switches for failures; ``None`` = compacted away."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.logs: Dict[str, Dict[int, List[Optional[Tuple[Any, Any]]]]] = {}
        self.low: Dict[Tuple[str, int], int] = {}
        self.metadata_error: Optional[Exception] = None
        self.metadata_calls = 0
        self.creating_topic_replies = 0
        self.topic_error: Optional[KafkaError] = None
        self.construct_error: Optional[Exception] = None
        self.watermark_error: Optional[Exception] = None
        self.close_error: Optional[Exception] = None
        self.on_poll: Optional[Callable[[Any], None]] = None
        self.on_close: Optional[Callable[[Any], None]] = None
        # producer switches
        self.hang = False
        self.delivery_error: Optional[KafkaError] = None
        self.acks_zero = False
        self.queue_full_times = 0
        self.produce_error: Optional[Exception] = None
        self.records: List[Dict[str, Any]] = []
        self.consumers: List["FakeConsumer"] = []
        self.producers: List["FakeProducer"] = []

    def create_topic(self, topic: str = TOPIC, partitions: int = 1) -> None:
        self.logs[topic] = {partition: [] for partition in range(partitions)}

    def append(
        self, value: Any, *, key: Any = None, partition: int = 0, topic: str = TOPIC
    ) -> int:
        if topic not in self.logs:
            self.create_topic(topic)
        log = self.logs[topic][partition]
        log.append((_bytes(key), _bytes(value)))
        return len(log) - 1

    def metadata(self, topic: str) -> Any:
        self.metadata_calls += 1
        if self.metadata_error is not None:
            raise self.metadata_error
        if self.creating_topic_replies > 0:
            self.creating_topic_replies -= 1
            return _metadata(topic, partitions=[], error=KafkaError(3))
        if self.topic_error is not None:
            return _metadata(topic, partitions=[], error=self.topic_error)
        if topic not in self.logs:
            return _metadata(topic, partitions=[], error=KafkaError(3))
        return _metadata(topic, partitions=list(self.logs[topic]), error=None)


def _bytes(value: Any) -> Any:
    return value.encode("utf-8") if isinstance(value, str) else value


class _TopicMetadata:
    def __init__(self, partitions: List[int], error: Any) -> None:
        self.partitions = {partition: None for partition in partitions}
        self.error = error


class _ClusterMetadata:
    def __init__(self, topics: Dict[str, _TopicMetadata]) -> None:
        self.topics = topics


def _metadata(topic: str, *, partitions: List[int], error: Any) -> _ClusterMetadata:
    return _ClusterMetadata({topic: _TopicMetadata(partitions, error)})


class FakeConsumer:
    def __init__(self, broker: Broker, config: Dict[str, Any]) -> None:
        self.broker = broker
        self.config = config
        self.topic: Optional[str] = None
        self.positions: Dict[int, int] = {}
        self.assign_calls: List[List[Tuple[int, int]]] = []
        self.poll_timeouts: List[float] = []
        self.closes = 0

    def list_topics(self, topic: str, timeout: float) -> Any:
        return self.broker.metadata(topic)

    def get_watermark_offsets(self, partition: TopicPartition, timeout: float):
        if self.broker.watermark_error is not None:
            raise self.broker.watermark_error
        log = self.broker.logs[partition.topic][partition.partition]
        low = self.broker.low.get((partition.topic, partition.partition), 0)
        return low, len(log)

    def assign(self, partitions: List[TopicPartition]) -> None:
        self.assign_calls.append([(tp.partition, tp.offset) for tp in partitions])
        self.topic = partitions[0].topic
        self.positions = {tp.partition: tp.offset for tp in partitions}

    def poll(self, timeout: float) -> Optional[Message]:
        self.poll_timeouts.append(timeout)
        if self.broker.on_poll is not None:
            self.broker.on_poll(self)
        for partition in sorted(self.positions):
            log = self.broker.logs[self.topic][partition]
            position = self.positions[partition]
            while position < len(log):
                entry = log[position]
                position += 1
                if entry is None:
                    continue
                self.positions[partition] = position
                key, value = entry
                if isinstance(value, (Exception, KafkaError)):
                    return Message(
                        self.topic, partition, position - 1, key, error=value
                    )
                return Message(self.topic, partition, position - 1, key, value)
            self.positions[partition] = position
        self.broker.clock.sleep(timeout)
        return None

    def close(self) -> None:
        self.closes += 1
        if self.broker.on_close is not None:
            self.broker.on_close(self)
        if self.broker.close_error is not None:
            raise self.broker.close_error


class FakeProducer:
    def __init__(self, broker: Broker, config: Dict[str, Any]) -> None:
        self.broker = broker
        self.config = config
        self.pending: List[Tuple[Any, Message]] = []
        self.poll_timeouts: List[float] = []
        self.flush_timeouts: List[float] = []
        self.poll_gate: Optional[threading.Event] = None
        self.poll_entered = threading.Event()
        self._lock = threading.Lock()

    def list_topics(self, topic: str, timeout: float) -> Any:
        return self.broker.metadata(topic)

    def produce(self, topic, value=None, key=None, headers=None, on_delivery=None):
        if self.broker.queue_full_times > 0:
            self.broker.queue_full_times -= 1
            raise BufferError("Local: Queue full")
        if self.broker.produce_error is not None:
            raise self.broker.produce_error
        record = {"topic": topic, "value": value, "key": key, "headers": headers}
        with self._lock:
            self.broker.records.append(record)
            offset = len(self.broker.records) - 1
            self.pending.append((on_delivery, Message(topic, 0, offset)))

    def poll(self, timeout: float) -> int:
        self.poll_timeouts.append(timeout)
        if self.poll_gate is not None and timeout > 0:
            self.poll_entered.set()
            assert self.poll_gate.wait(WAIT)
        if self.broker.hang:
            self.broker.clock.sleep(timeout)
            return 0
        return self._serve()

    def flush(self, timeout: float) -> int:
        self.flush_timeouts.append(timeout)
        if not self.broker.hang:
            self._serve()
        return len(self.pending)

    def _serve(self) -> int:
        with self._lock:
            served, self.pending = self.pending, []
        for callback, message in served:
            if self.broker.acks_zero:
                message._offset = None
            callback(self.broker.delivery_error, message)
        return len(served)


class FakeDriver:
    """Module-like stand-in for ``confluent_kafka``."""

    TopicPartition = TopicPartition
    KafkaError = KafkaError

    def __init__(self, broker: Broker) -> None:
        self.broker = broker

    def Consumer(self, config: Dict[str, Any]) -> FakeConsumer:
        if self.broker.construct_error is not None:
            raise self.broker.construct_error
        consumer = FakeConsumer(self.broker, config)
        self.broker.consumers.append(consumer)
        return consumer

    def Producer(self, config: Dict[str, Any]) -> FakeProducer:
        if self.broker.construct_error is not None:
            raise self.broker.construct_error
        producer = FakeProducer(self.broker, config)
        self.broker.producers.append(producer)
        return producer


class StandInSigner:
    """MSK signer stand-in: invented token, optional failure; counts calls."""

    def __init__(self, error: Optional[Exception] = None) -> None:
        self.error = error
        self.regions: List[str] = []

    def __call__(self, region: str) -> Tuple[str, float]:
        self.regions.append(region)
        if self.error is not None:
            raise self.error
        return "token-INVENTED", 1_700_000_000_000


class NoCredentialsError(Exception):
    pass


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    monkeypatch.setattr(
        environment, "KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS", True
    )
    monkeypatch.setattr(
        environment, "KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS", None
    )
    monkeypatch.setattr(environment, "GCP_SERVERLESS", False)
    monkeypatch.setattr(environment, "LAMBDA", False)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def broker(clock) -> Broker:
    fake = Broker(clock)
    fake.create_topic()
    return fake


@pytest.fixture
def cursor(broker, clock) -> KafkaConsumerCursor:
    owner = KafkaConsumerCursor(
        driver=FakeDriver(broker), clock=clock, sleep=clock.sleep
    )
    yield owner
    owner.close()


@pytest.fixture
def publisher(broker, clock) -> KafkaPublisher:
    owner = KafkaPublisher(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
    yield owner
    owner.close(timeout=0)


def connection(**overrides: Any) -> kafka.KafkaConnection:
    settings = dict(
        topic=TOPIC,
        provider="Self-hosted",
        username=None,
        password=None,
        aws_region=None,
        ssl_ca_location=None,
        allow_access_to_file_system=False,
        log_override=True,
    )
    bootstrap = overrides.pop("bootstrap_servers", BOOTSTRAP)
    settings.update(overrides)
    return kafka_connection(bootstrap, **settings)


def read(cursor: KafkaConsumerCursor, **overrides: Any) -> Dict[str, Any]:
    request = dict(
        read_mode="latest",
        key_filter=None,
        pointer=None,
        poll_timeout=0.01,
        connect_timeout=5.0,
    )
    connection_overrides = overrides.pop("connection", {})
    request.update(overrides)
    return cursor.read(
        KafkaRead(connection=connection(**connection_overrides), **request)
    )


def publish(
    publisher: KafkaPublisher, *, wait: bool, **overrides: Any
) -> Tuple[bool, str]:
    request = dict(
        acks="all",
        value=b'{"state": "RUNNING"}',
        key=b"cam-1",
        headers=None,
        timeout=5.0,
    )
    connection_overrides = overrides.pop("connection", {})
    request.update(overrides)
    return publisher.publish(
        KafkaPublish(connection=connection(**connection_overrides), **request),
        wait=wait,
    )


def state(value: str) -> str:
    return json.dumps({"state": value})


# --- consumer: read modes -----------------------------------------------------


def test_latest_starts_at_the_newest_record_and_repeats_it_when_nothing_arrived(
    cursor, broker
) -> None:
    for name in ("STARTING", "RUNNING", "STOPPED"):
        broker.append(state(name), key="cam-1")

    first = read(cursor)
    repeated = read(cursor)
    broker.append(state("A"))
    broker.append(state("B"))
    newest = read(cursor)

    assert first == {
        "value": state("STOPPED"),
        "payload": {"state": "STOPPED"},
        "key": "cam-1",
        "is_new": True,
        "offset": 2,
        "partition": 0,
        "error_status": False,
        "error_message": None,
    }
    assert repeated == {**first, "is_new": False}
    assert (newest["value"], newest["offset"], newest["is_new"]) == (
        state("B"),
        4,
        True,
    )


def test_latest_on_an_empty_topic_returns_nothing_until_a_record_arrives(
    cursor, broker
) -> None:
    empty = read(cursor)
    broker.append("hello")
    arrived = read(cursor)

    assert empty["value"] is None and empty["offset"] is None
    assert empty["error_status"] is False and empty["is_new"] is False
    assert (arrived["value"], arrived["is_new"]) == ("hello", True)


def test_sequential_returns_every_record_once_in_order(cursor, broker) -> None:
    for name in ("old", "newest-at-start"):
        broker.append(name)

    first = read(cursor, read_mode="sequential")
    idle = read(cursor, read_mode="sequential")
    broker.append("b1")
    broker.append("b2")
    backlog = [read(cursor, read_mode="sequential") for _ in range(3)]

    assert (first["value"], first["is_new"]) == ("newest-at-start", True)
    assert (idle["value"], idle["is_new"]) == ("newest-at-start", False)
    assert [(row["value"], row["is_new"]) for row in backlog] == [
        ("b1", True),
        ("b2", True),
        ("b2", False),
    ]


def test_a_pointer_in_latest_mode_reads_that_record_then_newest_again(
    cursor, broker
) -> None:
    for value in ("r0", "r1", "r2"):
        broker.append(value)

    pointed = read(cursor, pointer=(0, 1))
    assigns_after_first = len(broker.consumers[0].assign_calls)
    cached = read(cursor, pointer=(0, 1))
    newest = read(cursor)

    assert (pointed["value"], pointed["offset"], pointed["is_new"]) == ("r1", 1, True)
    assert (cached["value"], cached["is_new"]) == ("r1", False)
    assert len(broker.consumers[0].assign_calls) == assigns_after_first + 1
    assert (newest["value"], newest["is_new"]) == ("r2", True)


def test_a_sequential_pointer_is_a_start_that_only_a_different_pointer_resets(
    cursor, broker
) -> None:
    for value in ("r0", "r1", "r2", "r3"):
        broker.append(value)

    values = [
        read(cursor, read_mode="sequential", pointer=(0, 0))["value"] for _ in range(3)
    ]
    reset = read(cursor, read_mode="sequential", pointer=(0, 2))
    after_reset = read(cursor, read_mode="sequential", pointer=(0, 2))

    assert values == ["r0", "r1", "r2"]
    assert (reset["value"], after_reset["value"]) == ("r2", "r3")


def test_a_pointer_on_another_partition_positions_the_others_at_their_end(
    cursor, broker
) -> None:
    broker.create_topic(partitions=2)
    for value in ("a0", "a1"):
        broker.append(value, partition=0)
    for value in ("b0", "b1", "b2"):
        broker.append(value, partition=1)

    pointed = read(cursor, pointer=(1, 1))
    broker.append("a2", partition=0)
    continued = read(cursor, read_mode="sequential")

    assert (pointed["value"], pointed["partition"], pointed["offset"]) == ("b1", 1, 1)
    assert broker.consumers[0].assign_calls[0] == [(1, 1), (0, 2)]
    # leaving pointer mode for sequential restarts at the newest record
    assert (continued["value"], continued["is_new"]) == ("b2", True)


@pytest.mark.parametrize(
    "prepare, pointer, message",
    [
        (
            lambda b: None,
            (0, 5),
            "Offset 5 is out of range for partition 0: the newest record is at offset 2.",
        ),
        (
            lambda b: b.low.update({(TOPIC, 0): 1}),
            (0, 0),
            "Offset 0 on partition 0 is no longer retained (earliest available offset is 1).",
        ),
        (
            lambda b: b.logs[TOPIC][0].__setitem__(1, None),
            (0, 1),
            "Offset 1 on partition 0 no longer exists (removed by compaction or "
            "deletion); the next record is at offset 2.",
        ),
        (
            lambda b: None,
            (3, 0),
            "Partition 3 does not exist on topic 'line1.state' (partitions: [0]).",
        ),
    ],
)
def test_an_invalid_pointer_reports_an_error_and_keeps_the_last_record(
    cursor, broker, prepare, pointer, message
) -> None:
    for value in ("r0", "r1", "r2"):
        broker.append(value)
    last = read(cursor)
    prepare(broker)

    failed = read(cursor, pointer=pointer)

    assert failed == {
        **last,
        "is_new": False,
        "error_status": True,
        "error_message": message,
    }


def test_key_filter_finds_the_newest_record_for_its_key_on_the_first_run(
    cursor, broker
) -> None:
    broker.append(state("cam-1-old"), key="cam-1")
    broker.append(state("cam-1-new"), key="cam-1")
    broker.append(state("other"), key="cam-2")
    broker.append(state("unkeyed"))

    first = read(cursor, key_filter="cam-1")
    broker.append(state("other-2"), key="cam-2")
    unchanged = read(cursor, key_filter="cam-1")

    assert (first["payload"], first["key"], first["is_new"]) == (
        {"state": "cam-1-new"},
        "cam-1",
        True,
    )
    assert (unchanged["payload"], unchanged["is_new"]) == (
        {"state": "cam-1-new"},
        False,
    )


def test_values_decode_as_utf8_tombstones_stay_none_and_payload_is_detached(
    cursor, broker
) -> None:
    broker.append(b"\xff\xfeok", key=b"\xffk")
    replaced = read(cursor)
    broker.append(None, key="deleted")
    tombstone = read(cursor)
    broker.append(json.dumps([1, 2]))
    listed = read(cursor)
    broker.append(json.dumps({"nested": {"n": 1}}))
    parsed = read(cursor)
    parsed["payload"]["nested"]["n"] = 99
    repeated = read(cursor)

    assert (replaced["value"], replaced["key"]) == ("��ok", "�k")
    assert (tombstone["value"], tombstone["payload"], tombstone["key"]) == (
        None,
        {},
        "deleted",
    )
    assert listed["payload"] == {}
    assert repeated["payload"] == {"nested": {"n": 1}}


def test_a_polled_error_is_reported_with_the_newest_good_record(cursor, broker) -> None:
    broker.append("good")
    read(cursor)
    broker.append(KafkaError(-191, "Broker: offset out of range"))
    broker.append("later")

    result = read(cursor)

    assert (result["value"], result["is_new"], result["error_status"]) == (
        "later",
        True,
        True,
    )
    assert (
        result["error_message"]
        == "Kafka error while polling: Broker: offset out of range"
    )


# --- consumer: configuration and connection -----------------------------------


def test_consumer_config_never_commits_and_uses_a_unique_group(broker, clock) -> None:
    broker.append("x")
    owners = [
        KafkaConsumerCursor(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
        for _ in range(2)
    ]
    for owner in owners:
        read(owner)

    configs = [consumer.config for consumer in broker.consumers]
    assert configs[0]["enable.auto.commit"] is False
    assert configs[0]["auto.offset.reset"] == "latest"
    assert configs[0]["log_level"] == 4
    assert configs[0]["security.protocol"] == "PLAINTEXT"
    assert configs[0]["bootstrap.servers"] == BOOTSTRAP
    assert configs[0]["group.id"].startswith("roboflow-inference-")
    assert configs[0]["group.id"] != configs[1]["group.id"]
    assert callable(configs[0]["error_cb"])
    for owner in owners:
        owner.close()


def test_scram_and_tls_settings_reach_the_client(cursor, broker) -> None:
    read(
        cursor,
        connection=dict(
            username="reader",
            password=Secret(PASSWORD, name="KAFKA_PASSWORD"),
            ssl_ca_location="/etc/ssl/ca.pem",
            allow_access_to_file_system=True,
        ),
    )

    config = broker.consumers[0].config
    assert {
        key: config[key]
        for key in (
            "security.protocol",
            "sasl.mechanisms",
            "sasl.username",
            "sasl.password",
            "ssl.ca.location",
        )
    } == {
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "SCRAM-SHA-512",
        "sasl.username": "reader",
        "sasl.password": PASSWORD,
        "ssl.ca.location": "/etc/ssl/ca.pem",
    }


def test_msk_signs_once_before_connecting_with_the_derived_region(
    broker, clock
) -> None:
    signer = StandInSigner()
    owner = KafkaConsumerCursor(
        driver=FakeDriver(broker),
        msk_token_signer=signer,
        clock=clock,
        sleep=clock.sleep,
    )

    read(owner, connection=dict(bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK"))
    config = broker.consumers[0].config

    assert (config["security.protocol"], config["sasl.mechanisms"]) == (
        "SASL_SSL",
        "OAUTHBEARER",
    )
    assert signer.regions == ["us-east-1"]
    assert config["oauth_cb"]("") == ("token-INVENTED", 1_700_000_000.0)
    owner.close()


def test_msk_without_credentials_fails_before_constructing_a_client(
    broker, clock
) -> None:
    owner = KafkaConsumerCursor(
        driver=FakeDriver(broker),
        msk_token_signer=StandInSigner(
            NoCredentialsError("Unable to locate credentials")
        ),
        clock=clock,
    )

    result = read(
        owner, connection=dict(bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    )

    assert result["error_message"] == (
        "AWS MSK IAM authentication failed: no AWS credentials were found on this "
        "machine. Check that AWS credentials are available on this machine and that "
        "the identity may connect to the cluster."
    )
    assert broker.consumers == []
    owner.close()


def test_a_missing_msk_signer_package_is_reported_without_importing_it(
    broker, clock, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, "aws_msk_iam_sasl_signer", None)
    owner = KafkaConsumerCursor(driver=FakeDriver(broker), clock=clock)

    result = read(
        owner, connection=dict(bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    )

    assert result["error_message"] == (
        "AWS MSK provider requires the aws-msk-iam-sasl-signer-python package."
    )
    owner.close()


def test_a_missing_driver_is_reported_at_first_read(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "confluent_kafka", None)
    owner = KafkaConsumerCursor()

    result = read(owner)

    assert result["error_message"] == (
        "Kafka Consumer requires the confluent-kafka package in the runtime."
    )
    owner.close()


def test_transient_topic_replies_are_ridden_out_and_terminal_ones_are_not(
    cursor, broker, clock
) -> None:
    broker.append("x")
    broker.creating_topic_replies = 2
    started = clock.now

    connected = read(cursor)

    assert connected["value"] == "x"
    assert broker.metadata_calls == 3
    assert clock.now - started >= 2 * kafka.TOPIC_METADATA_RETRY_INTERVAL


def test_a_non_transient_topic_error_and_an_unreachable_broker_are_reported(
    broker, clock
) -> None:
    broker.topic_error = KafkaError(29, "Broker: Topic authorization failed")
    denied = read(
        KafkaConsumerCursor(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
    )
    broker.topic_error = None
    broker.metadata_error = RuntimeError("Local: Broker transport failure")
    down = read(
        KafkaConsumerCursor(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
    )

    assert denied["error_message"] == (
        "Kafka topic 'line1.state' is not available (Broker: Topic authorization failed)."
    )
    assert down["error_message"] == (
        "Kafka broker not reachable (Local: Broker transport failure). Raise "
        "'connect_timeout' if the broker needs longer to connect."
    )
    assert [consumer.closes for consumer in broker.consumers] == [1, 1]


def test_a_failed_positioning_closes_its_partial_client_and_the_next_read_starts_clean(
    cursor, broker
) -> None:
    broker.append("x")
    broker.watermark_error = RuntimeError("Local: Timed out")
    failed = read(cursor)
    broker.watermark_error = None

    recovered = read(cursor)

    assert failed["error_message"].startswith(
        "Kafka broker not reachable (Local: Timed out)"
    )
    assert [consumer.closes for consumer in broker.consumers] == [1, 0]
    assert recovered["value"] == "x"


def test_changed_connection_settings_are_rejected_with_v1s_message(
    cursor, broker
) -> None:
    broker.append("x")
    read(cursor, connection=dict(username="a", password=Secret("one", name="P")))

    changed = read(
        cursor, connection=dict(username="a", password=Secret("two", name="P"))
    )
    same_value_new_secret = read(
        cursor, connection=dict(username="a", password=Secret("one", name="OTHER"))
    )

    assert changed["error_message"] == (
        "Kafka connection parameters (bootstrap servers, topic, provider or "
        "credentials) changed between runs; this block reads only from the "
        "connection configured on its first run."
    )
    assert same_value_new_secret["error_status"] is False
    assert len(broker.consumers) == 1


def test_credential_text_in_driver_errors_is_scrubbed_from_outputs_and_logs(
    cursor, broker, caplog
) -> None:
    broker.construct_error = RuntimeError(f"sasl.password={PASSWORD} rejected")

    with caplog.at_level(logging.DEBUG):
        result = read(
            cursor,
            connection=dict(username="u", password=Secret(PASSWORD, name="KAFKA_PW")),
        )

    assert result["error_message"] == (
        "Kafka broker not reachable (sasl.password=<secret KAFKA_PW> rejected). Raise "
        "'connect_timeout' if the broker needs longer to connect."
    )
    assert PASSWORD not in caplog.text
    assert "<secret KAFKA_PW>" in caplog.text


# --- consumer: client problems and token callbacks ----------------------------


def test_a_client_problem_is_probed_at_once_then_reported_until_proven_healthy(
    cursor, broker, clock
) -> None:
    broker.append("x")
    read(cursor)
    error_cb = broker.consumers[0].config["error_cb"]

    error_cb(brokers_down())
    broker.metadata_error = RuntimeError("down")
    calls_before = broker.metadata_calls
    first = read(cursor)
    second = read(cursor)
    clock.sleep(kafka.CLIENT_RECOVERY_PROBE_INTERVAL)
    broker.metadata_error = None
    healed = read(cursor)

    assert first["error_status"] is True and first["value"] == "x"
    assert first["error_message"] == (
        "The Kafka client reported a broker problem, first seen 0.0s ago: "
        "_ALL_BROKERS_DOWN: 2/2 brokers are down. This is reported on every run "
        "until the connection is proven healthy again. The returned record is the "
        "last one read and may be stale."
    )
    assert second["error_status"] is True
    assert broker.metadata_calls == calls_before + 2
    assert healed["error_status"] is False


def test_a_blip_answered_by_the_first_probe_is_never_reported(cursor, broker) -> None:
    read(cursor)
    broker.consumers[0].config["error_cb"](brokers_down())

    assert read(cursor)["error_status"] is False


def test_a_new_record_is_proof_of_life_but_a_fatal_error_never_clears(
    cursor, broker
) -> None:
    broker.append("x")
    read(cursor)
    error_cb = broker.consumers[0].config["error_cb"]
    broker.metadata_error = RuntimeError("down")
    error_cb(brokers_down())
    read(cursor)
    broker.append("alive")
    alive = read(cursor)
    error_cb(brokers_down(fatal=True))
    broker.append("after-fatal")
    fatal = read(cursor)

    assert (alive["value"], alive["error_status"]) == ("alive", False)
    assert fatal["error_status"] is True
    assert "reported a fatal error" in fatal["error_message"]
    assert "the host must replace its Kafka owner" in fatal["error_message"]
    assert fatal["error_message"].endswith(
        "The returned record is new, but the client reported the problem during "
        "the same run."
    )


def test_a_token_refresh_failing_on_the_client_thread_is_reported_once(
    broker, clock
) -> None:
    signer = StandInSigner()
    owner = KafkaConsumerCursor(
        driver=FakeDriver(broker),
        msk_token_signer=signer,
        clock=clock,
        sleep=clock.sleep,
    )
    broker.append("x")
    read(owner, connection=dict(bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK"))

    def refresh_on_client_thread(consumer: FakeConsumer) -> None:
        thread = threading.Thread(
            target=_call_ignoring_errors, args=(consumer.config["oauth_cb"],)
        )
        thread.start()
        thread.join(WAIT)

    signer.error = RuntimeError("ExpiredToken")
    broker.on_poll = refresh_on_client_thread
    failed = read(
        owner, connection=dict(bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    )
    signer.error = None
    recovered = read(
        owner, connection=dict(bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    )

    assert failed["value"] == "x" and failed["error_status"] is True
    assert failed["error_message"].startswith(
        "AWS MSK IAM authentication failed: ExpiredToken."
    )
    assert recovered["error_status"] is False
    owner.close()


def _call_ignoring_errors(callback: Callable[[str], Any]) -> None:
    try:
        callback("")
    except Exception:
        pass


def test_a_refresh_succeeding_between_check_and_pop_is_not_an_error() -> None:
    signer = StandInSigner()

    class RacingFailures(list):
        def pop(self, *args: Any) -> Any:
            callback("")  # the refresh succeeds right before the pop
            return super().pop(*args)

    failures = RacingFailures([RuntimeError("stale")])
    callback = msk_token_callback("us-east-1", failures=failures, signer=signer)

    assert pop_auth_failure_message(failures) is None
    assert pop_auth_failure_message([]) is None


# --- consumer: ownership --------------------------------------------------------


def test_independent_cursors_keep_independent_positions(broker, clock) -> None:
    broker.append("r0")
    one, two = (
        KafkaConsumerCursor(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
        for _ in range(2)
    )
    read(one, read_mode="sequential")
    read(two, read_mode="sequential")
    broker.append("r1")
    broker.append("r2")

    assert [read(one, read_mode="sequential")["value"] for _ in range(2)] == [
        "r1",
        "r2",
    ]
    assert read(two, read_mode="sequential")["value"] == "r1"
    one.close()
    two.close()


def test_one_cursor_shared_by_two_readers_splits_a_sequential_stream(
    cursor, broker
) -> None:
    broker.append("r0")
    read(cursor, read_mode="sequential")
    for value in ("r1", "r2", "r3", "r4"):
        broker.append(value)

    reader_a = [read(cursor, read_mode="sequential")["value"] for _ in range(2)]
    reader_b = [read(cursor, read_mode="sequential")["value"] for _ in range(2)]

    assert (reader_a, reader_b) == (["r1", "r2"], ["r3", "r4"])


def test_close_waits_for_a_running_read_closes_once_and_rejects_later_reads(
    cursor, broker
) -> None:
    broker.append("x")
    read(cursor)
    entered, gate, closed = threading.Event(), threading.Event(), threading.Event()

    def block_in_poll(consumer: FakeConsumer) -> None:
        entered.set()
        assert gate.wait(WAIT)

    broker.on_poll = block_in_poll
    reader = threading.Thread(target=read, args=(cursor,))
    closer = threading.Thread(target=lambda: (cursor.close(), closed.set()))
    reader.start()
    assert entered.wait(WAIT)
    closer.start()

    assert not closed.wait(0.05)
    assert broker.consumers[0].closes == 0
    broker.on_poll = None
    gate.set()
    reader.join(WAIT)
    closer.join(WAIT)
    cursor.close()

    assert closed.is_set() and cursor.closed
    assert broker.consumers[0].closes == 1
    with pytest.raises(KafkaHostError, match="KafkaConsumerCursor is closed"):
        read(cursor)


def test_a_failing_consumer_close_is_logged_scrubbed(broker, clock, caplog) -> None:
    owner = KafkaConsumerCursor(
        driver=FakeDriver(broker), clock=clock, sleep=clock.sleep
    )
    read(owner, connection=dict(username="u", password=Secret(PASSWORD, name="PW")))
    broker.close_error = RuntimeError(f"close failed for {PASSWORD}")

    with caplog.at_level(logging.ERROR):
        owner.close()

    assert "Failed to close Kafka consumer: close failed for <secret PW>" in caplog.text
    assert PASSWORD not in caplog.text


def test_consumer_error_callbacks_stay_scrubbed_during_and_after_close(
    broker, clock, caplog
) -> None:
    owner = KafkaConsumerCursor(
        driver=FakeDriver(broker), clock=clock, sleep=clock.sleep
    )
    read(owner, connection=dict(username="u", password=Secret(PASSWORD, name="PW")))
    error_cb = broker.consumers[0].config["error_cb"]
    broker.on_close = lambda consumer: error_cb(
        KafkaError(-195, f"SASL {PASSWORD} closing")
    )

    with caplog.at_level(logging.ERROR):
        owner.close()
        error_cb(KafkaError(-187, f"late {PASSWORD}", fatal=True))

    assert "SASL <secret PW> closing" in caplog.text
    assert "late <secret PW>" in caplog.text
    assert PASSWORD not in caplog.text


def test_a_failed_attempts_late_errors_keep_its_own_scrubber_and_problems(
    cursor, broker, caplog
) -> None:
    old = dict(username="u", password=Secret(OLD_PASSWORD, name="OLD_PW"))
    new = dict(username="u", password=Secret(PASSWORD, name="NEW_PW"))
    broker.append("x")
    broker.watermark_error = RuntimeError("Local: Timed out")
    broker.on_close = lambda consumer: consumer.config["error_cb"](
        KafkaError(-195, f"closing {OLD_PASSWORD}")
    )
    with caplog.at_level(logging.ERROR):
        read(cursor, connection=old)
    broker.watermark_error = None
    broker.on_close = None
    read(cursor, connection=new)
    old_cb, new_cb = (consumer.config["error_cb"] for consumer in broker.consumers)

    with caplog.at_level(logging.ERROR):
        old_cb(KafkaError(-187, f"late {OLD_PASSWORD}", fatal=True))
        after_old = read(cursor, connection=new)
        new_cb(KafkaError(-195, f"now {PASSWORD}"))
        broker.metadata_error = RuntimeError("down")
        after_new = read(cursor, connection=new)

    assert old_cb is not new_cb
    assert (after_old["value"], after_old["error_status"]) == ("x", False)
    assert "now <secret NEW_PW>" in after_new["error_message"]
    assert "closing <secret OLD_PW>" in caplog.text
    assert "late <secret OLD_PW>" in caplog.text
    assert "now <secret NEW_PW>" in caplog.text
    assert OLD_PASSWORD not in caplog.text and PASSWORD not in caplog.text


def test_a_new_cursor_does_not_inherit_old_client_or_auth_problems(
    broker, clock
) -> None:
    broker.append("x")
    old = KafkaConsumerCursor(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
    read(old)
    broker.metadata_error = RuntimeError("down")
    broker.consumers[0].config["error_cb"](brokers_down(fatal=True))
    assert read(old)["error_status"] is True
    old.close()
    broker.metadata_error = None

    new = KafkaConsumerCursor(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
    fresh = read(new)

    assert (fresh["value"], fresh["is_new"], fresh["error_status"]) == (
        "x",
        True,
        False,
    )
    new.close()


# --- producer -----------------------------------------------------------------


def test_fire_and_forget_reports_scheduling_and_later_reports_stay_observable(
    publisher, broker, caplog
) -> None:
    broker.hang = True

    scheduled = publish(publisher, wait=False)
    pending = publisher.stats()
    broker.hang = False
    broker.delivery_error = KafkaError(2, "Broker: Message too large")
    with caplog.at_level(logging.ERROR):
        drained = publisher.flush(1.0)

    assert scheduled == (False, SCHEDULED_MESSAGE)
    assert (pending.queued, pending.pending, pending.delivered) == (1, 1, 0)
    assert (drained.failed, drained.pending) == (1, 0)
    assert [(f.topic, f.message, f.background) for f in publisher.failures()] == [
        (TOPIC, "Broker: Message too large", True)
    ]
    assert (
        "delivery to topic 'line1.state' failed: Broker: Message too large"
        in caplog.text
    )


@pytest.mark.parametrize("failure_site", ["partition", "offset", "error_text"])
def test_delivery_callback_failures_log_safely_and_allow_later_settlement(
    publisher, broker, caplog, failure_site
) -> None:
    broker.hang = True
    scheduled = publish(
        publisher,
        wait=False,
        connection={"username": "test", "password": Secret(PASSWORD, name="META")},
    )
    callback, message = broker.producers[0].pending.pop()

    class BrokenMessage:
        def partition(self):
            if failure_site == "partition":
                raise RuntimeError("metadata failure " + PASSWORD)

            return message.partition()

        def offset(self):
            raise RuntimeError("metadata failure " + PASSWORD)

    class BrokenError:
        def __str__(self):
            raise RuntimeError("error text failure " + PASSWORD)

    error = BrokenError() if failure_site == "error_text" else None
    with caplog.at_level(logging.ERROR):
        callback(error, BrokenMessage())

    assert scheduled == (False, SCHEDULED_MESSAGE)
    assert callback.report is None
    pending = publisher.stats()
    assert (pending.queued, pending.pending, pending.delivered, pending.failed) == (
        1,
        1,
        0,
        0,
    )
    assert not publisher.failures()
    # Format the complete record: getMessage() alone omits exception tracebacks.
    formatted = "\n".join(logging.Formatter().format(r) for r in caplog.records)
    assert "Kafka Producer: recording a delivery report failed" in formatted
    assert PASSWORD not in formatted
    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is None
    assert caplog.records[0].exc_text is None
    assert caplog.records[0].stack_info is None

    callback(None, message)
    delivered = publisher.stats()
    assert (
        delivered.queued, delivered.pending, delivered.delivered, delivered.failed
    ) == (1, 0, 1, 0)
    assert (callback.report.partition, callback.report.offset) == (0, 0)
    callback(KafkaError(2, "duplicate " + PASSWORD), None)
    assert publisher.stats().late_reports == 1
    assert publisher.stats().delivered == 1
    assert not publisher.failures()
    assert len(caplog.records) == 1
    assert len(broker.records) == 1


def test_confirmed_publish_reports_the_offset_acks_zero_and_rejection(
    publisher, broker
) -> None:
    delivered = publish(publisher, wait=True)
    broker.acks_zero = True
    unacknowledged = publish(publisher, wait=True)
    broker.acks_zero = False
    broker.delivery_error = KafkaError(2, "Broker: Message too large")
    rejected = publish(publisher, wait=True)

    assert delivered == (False, "Message delivered to partition 0 at offset 0")
    assert unacknowledged == (
        False,
        "Message sent to partition 0 without acknowledgement (acks=0)",
    )
    assert rejected == (True, "Kafka rejected the message: Broker: Message too large")
    stats = publisher.stats()
    assert (stats.queued, stats.delivered, stats.failed) == (3, 2, 1)


def test_an_unconfirmed_wait_reports_unknown_and_never_produces_twice(
    publisher, broker, clock
) -> None:
    broker.hang = True

    outcome = publish(publisher, wait=True, timeout=1.0)
    broker.hang = False
    stats = publisher.flush(1.0)

    assert outcome == (
        True,
        "Delivery not acknowledged within 1.0s (1 record(s) still queued); delivery "
        "status unknown and the message may still be delivered.",
    )
    assert len(broker.records) == 1
    assert (stats.delivered, stats.pending) == (1, 0)


def test_a_confirmed_wait_returns_when_another_thread_serves_its_report(
    publisher, broker
) -> None:
    publish(publisher, wait=True)
    producer = broker.producers[0]
    producer.poll_gate = threading.Event()
    broker.hang = True  # poll(0) right after produce serves nothing
    result: List[Tuple[bool, str]] = []
    waiter = threading.Thread(
        target=lambda: result.append(publish(publisher, wait=True))
    )
    waiter.start()
    assert producer.poll_entered.wait(WAIT)  # the waiter is inside its own poll

    broker.hang = False
    producer._serve()  # another thread serves the waiter's report
    producer.poll_gate.set()
    waiter.join(WAIT)

    assert result == [(False, "Message delivered to partition 0 at offset 1")]
    assert len(broker.records) == 2


class PausingMessage(Message):
    """Delivery metadata whose ``partition()`` blocks until released."""

    def __init__(
        self, message: Message, *, entered: threading.Event, release: threading.Event
    ) -> None:
        super().__init__(message.topic(), message.partition(), message.offset())
        self.entered = entered
        self.release = release

    def partition(self) -> int:
        self.entered.set()
        assert self.release.wait(WAIT)
        return super().partition()


class PausingError(KafkaError):
    """Delivery error whose text blocks until released."""

    def __init__(self, *, entered: threading.Event, release: threading.Event) -> None:
        super().__init__(2, "Broker: Message too large")
        self.entered = entered
        self.release = release

    def __str__(self) -> str:
        self.entered.set()
        assert self.release.wait(WAIT)
        return super().__str__()


@pytest.mark.parametrize(
    "fails, expected",
    [
        (False, (False, "Message delivered to partition 0 at offset 1")),
        (True, (True, "Kafka rejected the message: Broker: Message too large")),
    ],
)
def test_a_confirmed_wait_never_sees_its_report_before_the_details_are_complete(
    publisher, broker, fails, expected
) -> None:
    publish(publisher, wait=True)
    producer = broker.producers[0]
    in_details, release = threading.Event(), threading.Event()
    served, polled_again = threading.Event(), threading.Event()

    def poll(timeout: float) -> int:
        # The waiter's polls serve nothing: another thread serves its report.
        # The first returns while that report is half-read; later ones wait
        # (bounded) until it is complete.
        if timeout == 0:
            return 0
        if not producer.poll_entered.is_set():
            producer.poll_entered.set()
            assert in_details.wait(WAIT)
        else:
            polled_again.set()
            assert served.wait(WAIT)
        return 0

    producer.poll = poll
    result: List[Tuple[bool, str]] = []
    waiter = threading.Thread(
        target=lambda: result.append(publish(publisher, wait=True))
    )
    waiter.start()
    assert producer.poll_entered.wait(WAIT)
    callback, message = producer.pending.pop()
    if fails:
        report = (PausingError(entered=in_details, release=release), message)
    else:
        report = (None, PausingMessage(message, entered=in_details, release=release))
    server = threading.Thread(target=lambda: (callback(*report), served.set()))
    server.start()

    _wait_until(lambda: polled_again.is_set() or bool(result))
    returned_early = list(result)
    during: List[DeliveryStats] = []
    reader = threading.Thread(target=lambda: during.append(publisher.stats()))
    reader.start()
    reader.join(WAIT)
    release.set()
    server.join(WAIT)
    waiter.join(WAIT)

    assert returned_early == []
    # The ledger lock is not held while the driver's report is read.
    assert [stats.pending for stats in during] == [1]
    assert result == [expected]
    stats = publisher.stats()
    assert (stats.queued, stats.failed, stats.pending) == (2, int(fails), 0)
    assert stats.delivered == 2 - int(fails)
    assert len(broker.records) == 2


@pytest.mark.parametrize("wait", [True, False])
def test_a_polling_failure_after_queueing_reports_unknown_delivery_and_no_retry(
    publisher, broker, wait
) -> None:
    publish(publisher, wait=False)
    producer = broker.producers[0]
    serve = producer.poll

    def failing_poll(timeout: float) -> int:
        raise RuntimeError("Local: Fatal poll failure")

    producer.poll = failing_poll
    outcome = publish(publisher, wait=wait)
    pending = publisher.stats().pending
    producer.poll = serve
    drained = publisher.flush(1.0)

    assert outcome == (
        True,
        "Message queued, but serving delivery reports failed (Local: Fatal poll "
        "failure); delivery status unknown and the message may still be delivered, "
        "so publishing it again may duplicate it.",
    )
    assert pending == 1
    assert (drained.queued, drained.delivered, drained.pending) == (2, 2, 0)
    assert len(broker.records) == 2


def test_a_full_queue_is_polled_and_retried_once_without_duplicates(
    publisher, broker
) -> None:
    broker.queue_full_times = 1
    retried = publish(publisher, wait=False)
    broker.queue_full_times = 2
    full = publish(publisher, wait=False)

    assert retried == (False, SCHEDULED_MESSAGE)
    assert full == (
        True,
        "Kafka producer's local queue is full; the broker is slower than the stream. "
        "Lower the frame rate or use fire_and_forget=false.",
    )
    assert len(broker.records) == 1
    stats = publisher.stats()
    assert (stats.queued, stats.delivered, stats.pending) == (1, 1, 0)


def test_producer_config_carries_acks_and_rejects_a_changed_identity(
    publisher, broker
) -> None:
    publish(publisher, wait=False, acks="1")
    changed = publish(publisher, wait=False, acks="all")

    config = broker.producers[0].config
    assert (config["acks"], config["log_level"], config["security.protocol"]) == (
        "1",
        4,
        "PLAINTEXT",
    )
    assert "enable.auto.commit" not in config
    assert changed == (
        True,
        "Kafka connection parameters (bootstrap servers, topic, provider, credentials "
        "or acks) changed between runs; this block publishes only to the connection "
        "configured on its first run.",
    )
    assert len(broker.producers) == 1


def test_invalid_client_configuration_and_unreachable_brokers_are_distinguished(
    broker, clock
) -> None:
    broker.construct_error = Exception(KafkaError(-186, 'Invalid value "x" for "acks"'))
    invalid = publish(
        KafkaPublisher(driver=FakeDriver(broker), clock=clock), wait=False
    )
    broker.construct_error = None
    broker.metadata_error = RuntimeError("Local: All broker connections are down")
    owner = KafkaPublisher(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
    down = publish(owner, wait=False)
    broker.metadata_error = None
    recovered = publish(owner, wait=False)

    assert invalid == (
        True,
        'Invalid Kafka client configuration: Invalid value "x" for "acks"',
    )
    assert down == (
        True,
        "Kafka broker not reachable (Local: All broker connections are down). Raise "
        "'timeout' if the broker needs longer to connect.",
    )
    assert recovered == (False, SCHEDULED_MESSAGE)
    stats = owner.stats()
    assert (stats.clients_created, stats.clients_released) == (2, 1)
    owner.close(timeout=0)


def test_producer_client_problems_note_queued_records_and_clear_on_delivery(
    publisher, broker
) -> None:
    publish(publisher, wait=False)
    error_cb = broker.producers[0].config["error_cb"]
    broker.metadata_error = RuntimeError("down")
    broker.hang = True
    error_cb(brokers_down())

    queued = publish(publisher, wait=False)
    broker.hang = False
    broker.metadata_error = None
    delivered = publish(publisher, wait=True)

    assert queued[0] is True
    assert queued[1].endswith(
        "The record is queued in the client, which retries delivery for up to five "
        "minutes, but the broker is currently unreachable."
    )
    assert delivered == (False, "Message delivered to partition 0 at offset 2")


def test_credential_text_is_scrubbed_from_producer_outputs_failures_and_logs(
    publisher, broker, caplog
) -> None:
    secret = Secret(PASSWORD, name="KAFKA_PW")
    login = dict(username="writer", password=secret)
    with caplog.at_level(logging.DEBUG):
        publish(publisher, wait=False, connection=login)
        broker.produce_error = RuntimeError(f"auth {PASSWORD} refused")
        refused = publish(publisher, wait=False, connection=login)
        broker.produce_error = None
        broker.delivery_error = KafkaError(1, f"SASL {PASSWORD} rejected")
        publish(publisher, wait=False, connection=login)

    assert refused == (
        True,
        "Failed to publish message: auth <secret KAFKA_PW> refused",
    )
    assert publisher.failures()[0].message == "SASL <secret KAFKA_PW> rejected"
    assert PASSWORD not in caplog.text


def test_producer_callbacks_keep_their_scrubbers_after_a_failed_attempt_and_close(
    broker, clock, caplog
) -> None:
    old = dict(username="u", password=Secret(OLD_PASSWORD, name="OLD_PW"))
    new = dict(username="u", password=Secret(PASSWORD, name="NEW_PW"))
    owner = KafkaPublisher(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
    broker.metadata_error = RuntimeError("Local: All broker connections are down")
    publish(owner, wait=False, connection=old)
    broker.metadata_error = None
    publish(owner, wait=False, connection=new)
    old_cb, new_cb = (producer.config["error_cb"] for producer in broker.producers)

    with caplog.at_level(logging.ERROR):
        old_cb(KafkaError(-187, f"late {OLD_PASSWORD}", fatal=True))
        healthy = publish(owner, wait=True, connection=new)
        broker.hang = True
        publish(owner, wait=False, connection=new)
        stuck_callback, stuck_message = broker.producers[1].pending[0]
        closed = owner.close(timeout=0)
        new_cb(KafkaError(-187, f"after close {PASSWORD}", fatal=True))
        stuck_callback(KafkaError(1, f"SASL {PASSWORD} rejected"), stuck_message)

    assert old_cb is not new_cb
    assert healthy == (False, "Message delivered to partition 0 at offset 1")
    assert (closed.unknown_at_close, owner.stats().late_reports) == (1, 1)
    assert owner.failures() == ()
    assert "late <secret OLD_PW>" in caplog.text
    assert "after close <secret NEW_PW>" in caplog.text
    assert OLD_PASSWORD not in caplog.text and PASSWORD not in caplog.text


def test_close_waits_for_a_running_publish_and_rejects_new_admission(
    publisher, broker
) -> None:
    publish(publisher, wait=False)
    producer = broker.producers[0]
    producer.poll_gate = threading.Event()
    broker.hang = True  # the confirmed record stays pending until the gate opens
    results: List[Any] = []
    publisher_thread = threading.Thread(
        target=lambda: results.append(publish(publisher, wait=True))
    )
    closed = threading.Event()
    stats: List[DeliveryStats] = []
    closer = threading.Thread(
        target=lambda: (stats.append(publisher.close(timeout=1.0)), closed.set())
    )

    publisher_thread.start()
    assert producer.poll_entered.wait(WAIT)
    closer.start()
    _wait_until(lambda: publisher.closed)
    with pytest.raises(KafkaHostError, match="KafkaPublisher is closed"):
        publish(publisher, wait=False)
    assert not closed.wait(0.05)
    assert producer.flush_timeouts == []
    broker.hang = False
    producer.poll_gate.set()
    publisher_thread.join(WAIT)
    closer.join(WAIT)

    assert results == [(False, "Message delivered to partition 0 at offset 1")]
    assert producer.flush_timeouts == [1.0]
    assert stats[0] == DeliveryStats(
        queued=2,
        delivered=2,
        failed=0,
        pending=0,
        unknown_at_close=0,
        late_reports=0,
        clients_created=1,
        clients_released=1,
    )
    assert publisher.close(timeout=9) is stats[0]
    assert publisher.flush(1.0) == stats[0]


def test_close_settles_reports_served_while_closing_and_marks_the_rest_unknown(
    broker, clock
) -> None:
    owner = KafkaPublisher(driver=FakeDriver(broker), clock=clock, sleep=clock.sleep)
    broker.hang = True
    publish(owner, wait=False)
    publish(owner, wait=False)
    producer = broker.producers[0]
    stuck = producer.pending[1]
    producer.pending = producer.pending[:1]
    broker.hang = False
    broker.delivery_error = KafkaError(1, "Broker: Not enough replicas")

    stats = owner.close(timeout=1.0)
    stuck[0](None, stuck[1])  # a report arriving after close

    assert (stats.failed, stats.delivered, stats.unknown_at_close) == (1, 0, 1)
    assert owner.stats().late_reports == 1
    assert owner.stats().delivered == 0
    assert owner.failures()[0].message == "Broker: Not enough replicas"


def test_flush_during_a_draining_close_is_provisional_and_close_returns_a_snapshot(
    publisher, broker
) -> None:
    broker.hang = True
    publish(publisher, wait=False)
    producer = broker.producers[0]
    in_flush, release = threading.Event(), threading.Event()

    def held_flush(timeout: float) -> int:
        in_flush.set()
        assert release.wait(WAIT)
        return len(producer.pending)

    producer.flush = held_flush
    closed: List[DeliveryStats] = []
    closer = threading.Thread(
        target=lambda: closed.append(publisher.close(timeout=1.0))
    )
    closer.start()
    assert in_flush.wait(WAIT)

    during = publisher.flush(1.0)
    release.set()
    closer.join(WAIT)
    callback, message = producer.pending[0]
    callback(None, message)  # a report after close

    assert (during.pending, during.unknown_at_close) == (1, 0)
    assert (closed[0].pending, closed[0].unknown_at_close) == (0, 1)
    assert publisher.close(timeout=0) is closed[0]
    assert closed[0].late_reports == 0
    assert (publisher.stats().late_reports, publisher.stats().delivered) == (1, 0)


def test_flush_is_draining_only_and_kept_failures_must_be_positive(
    publisher, broker
) -> None:
    publish(publisher, wait=False)
    publisher.flush(1.0)

    assert not publisher.closed
    assert publish(publisher, wait=False) == (False, SCHEDULED_MESSAGE)
    with pytest.raises(ValueError, match="kept_failures"):
        KafkaPublisher(kept_failures=0)


def test_a_missing_producer_driver_is_reported_at_first_publish(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "confluent_kafka", None)
    owner = KafkaPublisher()

    outcome = publish(owner, wait=False)

    assert outcome == (
        True,
        "Kafka Producer requires the confluent-kafka package in the runtime.",
    )
    assert owner.close(timeout=0).clients_created == 0


# --- helpers ------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry, expected",
    [
        (" Broker-1:09092 ", "broker-1:9092"),
        ("broker-1", "broker-1:9092"),
        ("broker-1:", "broker-1:9092"),
        ("[::1]:9093", "[::1]:9093"),
        ("::1", "[::1]:9092"),
        ("SSL://Host:9092", "ssl://host:9092"),
    ],
)
def test_bootstrap_entries_normalise_like_v1(entry: str, expected: str) -> None:
    assert normalise_bootstrap_server(entry) == expected


def test_operator_policy_replaces_restricts_or_disables_bootstrap_servers(
    monkeypatch, caplog
) -> None:
    allow = "KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS"
    listed = "KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS"
    monkeypatch.setattr(environment, listed, ["internal-b:9092", "internal-a:9092"])
    monkeypatch.setattr(environment, allow, False)
    with caplog.at_level(logging.WARNING):
        replaced = resolve_bootstrap_servers("user:9092", log_override=True)
    quiet = len(caplog.records)
    resolve_bootstrap_servers("user:9092", log_override=False)

    monkeypatch.setattr(environment, allow, True)
    permitted = resolve_bootstrap_servers(
        "INTERNAL-A, internal-b:9092", log_override=True
    )
    with pytest.raises(KafkaConfigurationError) as refused:
        resolve_bootstrap_servers("evil:9092", log_override=True)

    monkeypatch.setattr(environment, allow, False)
    monkeypatch.setattr(environment, listed, None)
    with pytest.raises(KafkaConfigurationError, match="disabled on this deployment"):
        resolve_bootstrap_servers("user:9092", log_override=True)

    assert replaced == "internal-b:9092,internal-a:9092"
    assert len(caplog.records) == quiet == 1
    assert permitted == "INTERNAL-A, internal-b:9092"
    assert "internal" not in str(refused.value)


def test_connection_validation_follows_v1_rules() -> None:
    with pytest.raises(
        KafkaConfigurationError, match="ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE"
    ):
        connection(ssl_ca_location="/etc/ca.pem")
    with pytest.raises(KafkaConfigurationError, match="Set both username and password"):
        connection(username="only-user")
    with pytest.raises(
        KafkaConfigurationError, match="Could not derive the AWS region"
    ):
        connection(provider="AWS MSK")
    with pytest.raises(KafkaConfigurationError, match="Unknown provider 'Confluent'"):
        connection(provider="Confluent")
    with pytest.raises(
        KafkaConfigurationError, match="bootstrap_servers and topic must be set"
    ):
        connection(topic="  ")

    explicit = connection(provider="AWS MSK", aws_region=" eu-west-1 ")
    assert explicit.msk_region == "eu-west-1"
    assert (
        derive_msk_region("x:1, b.kafka-serverless.ap-south-1.amazonaws.com.cn:9098")
        == "ap-south-1"
    )
    assert "pw" not in repr(connection(username="u", password=Secret("pw", name="P")))


def test_values_keys_headers_and_acks_encode_like_v1() -> None:
    assert encode_value({"a": 1}) == b'{"a": 1}'
    assert encode_value([1]) == b"[1]"
    assert encode_value("żółw") == "żółw".encode("utf-8")
    assert encode_key(None) is None and encode_key("cam-1") == b"cam-1"
    assert encode_headers({"a": "x", "b": None, "c": b"raw"}) == (
        ("a", b"x"),
        ("b", None),
        ("c", b"raw"),
    )
    assert [coerce_acks(value) for value in ("all", 0, 1.0, " 1 ")] == [
        "all",
        "0",
        "1",
        "1",
    ]
    for bad, call in [
        ("\ud800", encode_value),
        ({"x": object()}, encode_value),
        (3.5, encode_value),
        ("\ud800", encode_key),
        (["not", "a", "dict"], encode_headers),
        (True, coerce_acks),
        ("2", coerce_acks),
    ]:
        with pytest.raises(KafkaConfigurationError):
            call(bad)


# --- blocks without the compiler ----------------------------------------------


class RecordingConsumerPort:
    """Fake port: records requests; it is not a Kafka client."""

    def __init__(self) -> None:
        self.requests: List[KafkaRead] = []
        self.repeated: List[str] = []

    def read(self, request: KafkaRead) -> Dict[str, Any]:
        self.requests.append(request)
        return {"value": "v", "error_status": False}

    def repeat_last(self, *, error_message: str) -> Dict[str, Any]:
        self.repeated.append(error_message)
        return {"value": "last", "error_status": True, "error_message": error_message}


class RecordingPublisherPort:
    """Fake port: records requests; it is not a Kafka client."""

    def __init__(self) -> None:
        self.requests: List[Tuple[KafkaPublish, bool]] = []

    def publish(self, request: KafkaPublish, *, wait: bool) -> Tuple[bool, str]:
        self.requests.append((request, wait))
        return False, SCHEDULED_MESSAGE


def consumer_run(block: KafkaConsumerBlock, **overrides: Any) -> Dict[str, Any]:
    params = dict(bootstrap_servers=BOOTSTRAP, topic=TOPIC)
    params.update(overrides)
    return block.run(**params)


def producer_run(block: KafkaProducerSinkBlock, **overrides: Any) -> Dict[str, Any]:
    params = dict(
        bootstrap_servers=BOOTSTRAP, topic=TOPIC, message={"state": "RUNNING"}
    )
    params.update(overrides)
    return block.run(**params)


def test_the_consumer_block_needs_a_cursor_except_on_the_hosted_platform(
    monkeypatch,
) -> None:
    with pytest.raises(KafkaHostError, match='resources\\["kafka_consumer"\\]'):
        consumer_run(KafkaConsumerBlock())
    monkeypatch.setattr(environment, "LAMBDA", True)

    hosted = consumer_run(KafkaConsumerBlock())

    assert (
        hosted["error_message"]
        == "Kafka Consumer is not available on the Roboflow hosted platform."
    )
    assert hosted["error_status"] is True and hosted["value"] is None


def test_the_consumer_block_builds_an_immutable_request_and_reports_config_errors() -> (
    None
):
    port = RecordingConsumerPort()
    block = KafkaConsumerBlock(kafka_consumer=port)

    consumer_run(
        block,
        read_mode="sequential",
        offset=7,
        partition=2,
        key_filter="cam-1",
        poll_timeout=0,
    )
    failed = consumer_run(block, username="only-user", password=None)

    request = port.requests[0]
    assert (
        request.read_mode,
        request.pointer,
        request.key_filter,
        request.poll_timeout,
    ) == (
        "sequential",
        (2, 7),
        "cam-1",
        0.0,
    )
    assert failed["value"] == "last"
    assert port.repeated == [
        "Set both username and password for SASL authentication, or neither for an "
        "unauthenticated plaintext listener."
    ]


def test_the_consumer_block_fails_the_step_for_an_unavailable_secret() -> None:
    port = RecordingConsumerPort()

    with pytest.raises(SecretUnavailableError):
        consumer_run(
            KafkaConsumerBlock(kafka_consumer=port),
            username="u",
            password=Secret.unavailable("KAFKA_PW"),
        )
    assert port.requests == []


def test_a_closed_cursor_fails_the_step_before_reporting_invalid_parameters(
    cursor, broker
) -> None:
    read(cursor)
    cursor.close()
    block = KafkaConsumerBlock(kafka_consumer=cursor)

    for topic in ("", TOPIC):
        with pytest.raises(KafkaHostError, match="KafkaConsumerCursor is closed"):
            consumer_run(block, topic=topic)
    assert [consumer.closes for consumer in broker.consumers] == [1]


def test_the_producer_block_skips_everything_when_disabled() -> None:
    disabled = producer_run(
        KafkaProducerSinkBlock(disable_sinks=True), password=Secret.unavailable("PW")
    )

    assert disabled == {
        "error_status": False,
        "message": "Sink was disabled by workflow execution policy",
    }


def test_the_producer_block_needs_a_publisher_and_treats_empty_credentials_as_none() -> (
    None
):
    with pytest.raises(KafkaHostError, match='resources\\["kafka_publisher"\\]'):
        producer_run(KafkaProducerSinkBlock())
    port = RecordingPublisherPort()
    block = KafkaProducerSinkBlock(kafka_publisher=port)

    producer_run(
        block, username="", password="", key="cam-1", headers={"s": "v"}, acks=1
    )
    producer_run(block, fire_and_forget=False)

    request, wait = port.requests[0]
    assert (request.connection.username, request.connection.password) == (None, None)
    assert (request.acks, request.key, request.headers, request.value, wait) == (
        "1",
        b"cam-1",
        (("s", b"v"),),
        b'{"state": "RUNNING"}',
        False,
    )
    assert port.requests[1][1] is True


def test_the_producer_block_reports_config_errors_scrubbed(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        environment, "KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS", ["ok:9092"]
    )
    port = RecordingPublisherPort()
    block = KafkaProducerSinkBlock(kafka_publisher=port)

    with caplog.at_level(logging.ERROR):
        failed = producer_run(block, bootstrap_servers="ok:9092", message="bad \ud800")
        refused = producer_run(
            block,
            bootstrap_servers=f"{PASSWORD}:9092",
            username="u",
            password=Secret(PASSWORD, name="PW"),
        )

    assert failed == {
        "error_status": True,
        "message": "message contains characters that cannot be encoded as UTF-8.",
    }
    assert refused == {
        "error_status": True,
        "message": "Bootstrap server '<secret PW>:9092' is not permitted on this "
        "deployment: the operator restricts the Kafka servers Workflow blocks may "
        "connect to.",
    }
    assert PASSWORD not in caplog.text
    assert port.requests == []


def test_block_declarations_match_v1() -> None:
    consumer_params = KafkaConsumerBlock.Params(
        bootstrap_servers=BOOTSTRAP, topic=TOPIC
    )
    ff_params = KafkaProducerSinkBlock.Params(
        bootstrap_servers=BOOTSTRAP, topic=TOPIC, message="m"
    )
    confirmed = KafkaProducerSinkBlock.Params(
        bootstrap_servers=BOOTSTRAP, topic=TOPIC, message="m", fire_and_forget=False
    )

    assert [
        op.value for op in KafkaConsumerBlock.discover_work_operations(consumer_params)
    ] == [
        "external_request",
        "temporal_buffering",
    ]
    assert [
        r.code for r in KafkaConsumerBlock.discover_restrictions(consumer_params)
    ] == [
        "unavailable_on_hosted_platform",
        "connection_and_state_rebuilt_per_request",
    ]
    assert [
        r.code for r in KafkaProducerSinkBlock.discover_restrictions(ff_params)
    ] == [
        "unavailable_on_hosted_platform",
        "fire_and_forget_hides_persistence_failures",
    ]
    assert [
        r.code for r in KafkaProducerSinkBlock.discover_restrictions(confirmed)
    ] == ["unavailable_on_hosted_platform"]
    assert KafkaConsumerBlock.discover_dependent_resources(consumer_params) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"offset": -1},
        {"partition": -1},
        {"poll_timeout": -0.1},
        {"connect_timeout": 0},
        {"connect_timeout": float("inf")},
        {"poll_timeout": True},
        {"read_mode": "earliest"},
        {"provider": "Confluent"},
    ],
)
def test_invalid_literal_consumer_params_are_rejected(
    overrides: Dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        KafkaConsumerBlock.Params(bootstrap_servers=BOOTSTRAP, topic=TOPIC, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"timeout": 0},
        {"acks": "2"},
        {"acks": 2},
        {"fire_and_forget": "yes"},
        {"headers": {"a": 1}},
    ],
)
def test_invalid_literal_producer_params_are_rejected(
    overrides: Dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        KafkaProducerSinkBlock.Params(
            bootstrap_servers=BOOTSTRAP, topic=TOPIC, message="m", **overrides
        )


def test_importing_the_module_imports_no_optional_driver() -> None:
    assert (
        "confluent_kafka" not in sys.modules or sys.modules["confluent_kafka"] is None
    )
    assert (
        "aws_msk_iam_sasl_signer" not in sys.modules
        or sys.modules["aws_msk_iam_sasl_signer"] is None
    )


def _wait_until(predicate: Callable[[], bool]) -> None:
    done = threading.Event()
    for _ in range(5000):
        if predicate():
            return
        done.wait(0.001)
    raise AssertionError("condition not reached")


@pytest.mark.parametrize("boundary", ["parser", "copy"])
def test_json_recursion_limit_preserves_raw_record_and_next_read(
    cursor, broker, boundary
):
    if boundary == "parser":
        raw = "[" * 10000 + "0" + "]" * 10000
        with pytest.raises(RecursionError):
            json.loads(raw)
    else:
        # JSON accepts this object; deepcopy needs more stack frames per level.
        depth = sys.getrecursionlimit() * 2 // 3
        raw = '{"nested":' + "[" * depth + "0" + "]" * depth + "}"
        parsed = json.loads(raw)
        with pytest.raises(RecursionError):
            copy.deepcopy(parsed)

    broker.append("prior", key="prior")
    broker.append(raw, key="deep")
    options = {"read_mode": "sequential", "pointer": (0, 1)}
    first = read(cursor, **options)
    assert first == {
        "value": raw,
        "payload": {},
        "key": "deep",
        "is_new": True,
        "offset": 1,
        "partition": 0,
        "error_status": False,
        "error_message": None,
    }
    assert broker.consumers[0].positions == {0: 2}
    assert read(cursor, **options) == {**first, "is_new": False}
    broker.append('{"next":"łódź/東京","null":null}', key="next")
    broker.append("last", key="last")
    next_row = read(cursor, **options)
    assert (next_row["offset"], next_row["key"], next_row["payload"]) == (
        2,
        "next",
        {"next": "łódź/東京", "null": None},
    )
    assert broker.consumers[0].positions == {0: 3}
    assert read(cursor, **options)["value"] == "last"


@pytest.mark.parametrize(
    "raw, payload",
    [
        (
            '{"nested":[{"unicode":"łódź/東京","null":null}]}',
            {"nested": [{"unicode": "łódź/東京", "null": None}]},
        ),
        ("[1,[2]]", {}),
        ("null", {}),
        ('"hello"', {}),
        ("0", {}),
        ("", {}),
        ("{broken", {}),
    ],
)
def test_optional_json_view_keeps_shallow_semantics(cursor, broker, raw, payload):
    broker.append(raw)
    result = read(cursor)
    assert result["value"] == raw and result["payload"] == payload
    assert result["is_new"] and not result["error_status"]
    result["payload"].clear()
    assert read(cursor)["payload"] == payload


@pytest.mark.parametrize("operation", ["loads", "deepcopy"])
def test_optional_json_view_does_not_hide_unexpected_errors(
    cursor, broker, monkeypatch, operation
):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected host failure")

    broker.append('{"ok":true}')
    module = kafka.json if operation == "loads" else kafka.copy
    monkeypatch.setattr(module, operation, fail)
    with pytest.raises(RuntimeError, match="unexpected host failure"):
        read(cursor)
