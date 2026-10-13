"""Kafka blocks through the real compiler and sessions, with host-owned owners.

The blocks are registered in a test-local catalogue until root adds them to
the shared one. The owners are the real ``KafkaConsumerCursor`` and
``KafkaPublisher``; only their driver is ``FakeDriver``, an in-memory stand-in
for the ``confluent_kafka`` module. No test opens a socket::

    inputs / secrets store ─▶ v2/kafka_consumer@v1     ─▶ KafkaConsumerCursor ─┐
                            ─▶ v2/kafka_producer_sink@v1 ─▶ KafkaPublisher    ─┴▶ FakeDriver
                               (borrowed; the test, as host, closes both)
"""

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from roboflow_workflows import environment
from roboflow_workflows.v2.blocks.industrial.kafka import (
    SCHEDULED_MESSAGE,
    KafkaConsumerBlock,
    KafkaConsumerCursor,
    KafkaProducerSinkBlock,
    KafkaPublisher,
)
from roboflow_workflows.v2.blocks.secrets import (
    SECRET_RECORDING_CODEC,
    EnvironmentSecretsStoreBlock,
    StaticSecrets,
)
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.introspection.workload import discover_workload

TOPIC = "line1.state"
PASSWORD = "pw-SENTINEL-kafka-5d20"
CATALOGUE = Catalogue(
    [KafkaConsumerBlock, KafkaProducerSinkBlock, EnvironmentSecretsStoreBlock],
    kinds=[SECRET_KIND],
    codecs=[SECRET_RECORDING_CODEC],
)
CONSUMER_OUTPUTS = ("value", "key", "is_new", "offset", "partition", "error_status")


# --- in-memory stand-in for the confluent_kafka module -------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TopicPartition:
    def __init__(self, topic: str, partition: int, offset: int = -1001) -> None:
        self.topic, self.partition, self.offset = topic, partition, offset


class Message:
    def __init__(self, topic, partition, offset, key=None, value=None) -> None:
        self._fields = (topic, partition, offset, key, value)

    def topic(self):
        return self._fields[0]

    def partition(self):
        return self._fields[1]

    def offset(self):
        return self._fields[2]

    def key(self):
        return self._fields[3]

    def value(self):
        return self._fields[4]

    def error(self):
        return None


class _Topic:
    def __init__(self, partitions: List[int]) -> None:
        self.partitions = {partition: None for partition in partitions}
        self.error = None


class _Metadata:
    def __init__(self, topic: str, partitions: List[int]) -> None:
        self.topics = {topic: _Topic(partitions)}


class FakeDriver:
    """Module-like stand-in: one topic's partition logs, consumers and producers."""

    TopicPartition = TopicPartition

    class KafkaError:
        _INVALID_ARG = -186

    def __init__(self, clock: FakeClock, partitions: int = 1) -> None:
        self.clock = clock
        self.logs: Dict[int, List[Tuple[bytes, bytes]]] = {
            partition: [] for partition in range(partitions)
        }
        self.construct_error: Optional[Exception] = None
        self.hang = False
        self.delivery_error: Any = None
        self.consumers: List["FakeConsumer"] = []
        self.producers: List["FakeProducer"] = []

    def append(self, value: str, *, key: str = "cam-1", partition: int = 0) -> None:
        self.logs[partition].append((key.encode(), value.encode()))

    def Consumer(self, config: Dict[str, Any]) -> "FakeConsumer":
        if self.construct_error is not None:
            raise self.construct_error
        consumer = FakeConsumer(self, config)
        self.consumers.append(consumer)
        return consumer

    def Producer(self, config: Dict[str, Any]) -> "FakeProducer":
        producer = FakeProducer(self, config)
        self.producers.append(producer)
        return producer


class FakeConsumer:
    def __init__(self, driver: FakeDriver, config: Dict[str, Any]) -> None:
        self.driver, self.config = driver, config
        self.positions: Dict[int, int] = {}
        self.closes = 0
        self.on_close: Optional[Callable[[], None]] = None

    def list_topics(self, topic, timeout):
        return _Metadata(topic, list(self.driver.logs))

    def get_watermark_offsets(self, partition, timeout):
        return 0, len(self.driver.logs[partition.partition])

    def assign(self, partitions):
        self.positions = {tp.partition: tp.offset for tp in partitions}

    def poll(self, timeout):
        for partition in sorted(self.positions):
            position = self.positions[partition]
            log = self.driver.logs[partition]
            if position < len(log):
                self.positions[partition] = position + 1
                key, value = log[position]
                return Message(TOPIC, partition, position, key, value)
        self.driver.clock.sleep(timeout)
        return None

    def close(self):
        self.closes += 1
        if self.on_close is not None:
            self.on_close()


class FakeProducer:
    def __init__(self, driver: FakeDriver, config: Dict[str, Any]) -> None:
        self.driver, self.config = driver, config
        self.records: List[bytes] = []
        self.pending: List[Tuple[Any, Message]] = []
        self.flushes: List[float] = []
        self._lock = threading.Lock()  # librdkafka's queue is thread-safe too

    def list_topics(self, topic, timeout):
        return _Metadata(topic, [0])

    def produce(self, topic, value=None, key=None, headers=None, on_delivery=None):
        with self._lock:
            self.records.append(value)
            message = Message(topic, 0, len(self.records) - 1)
            self.pending.append((on_delivery, message))

    def poll(self, timeout):
        if self.driver.hang:
            self.driver.clock.sleep(timeout)
            return 0
        with self._lock:
            served, self.pending = self.pending, []
        for callback, message in served:
            callback(self.driver.delivery_error, message)
        return len(served)

    def flush(self, timeout):
        self.flushes.append(timeout)
        self.poll(0)
        return len(self.pending)


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
def driver(clock) -> FakeDriver:
    return FakeDriver(clock, partitions=2)


def new_cursor(driver: FakeDriver) -> KafkaConsumerCursor:
    return KafkaConsumerCursor(
        driver=driver, clock=driver.clock, sleep=driver.clock.sleep
    )


def new_publisher(driver: FakeDriver) -> KafkaPublisher:
    return KafkaPublisher(driver=driver, clock=driver.clock, sleep=driver.clock.sleep)


def compile_consumer(*extra_steps: Dict[str, Any], **overrides: Any):
    step = {
        "type": "v2/kafka_consumer@v1",
        "name": "kafka",
        "bootstrap_servers": "broker:9092",
        "topic": TOPIC,
        "offset": "$inputs.offset",
        "partition": "$inputs.partition",
    }
    step.update(overrides)
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "offset", "default_value": None},
            {"type": "WorkflowParameter", "name": "partition", "default_value": 0},
        ],
        "steps": [*extra_steps, step],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.kafka.{name}"}
            for name in (*CONSUMER_OUTPUTS, "error_message")
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def compile_producer(**overrides: Any):
    step = {
        "type": "v2/kafka_producer_sink@v1",
        "name": "sink",
        "bootstrap_servers": "broker:9092",
        "topic": TOPIC,
        "message": "$inputs.message",
        "key": "cam-1",
        "fire_and_forget": "$inputs.fire_and_forget",
    }
    step.update(overrides)
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "message"},
            {
                "type": "WorkflowParameter",
                "name": "fire_and_forget",
                "default_value": True,
            },
        ],
        "steps": [step],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.sink.{name}"}
            for name in ("error_status", "message")
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def row(session, **inputs: Any) -> Dict[str, Any]:
    result = session.run(inputs).rows(serialize=True)[0]

    return result


def brief(result: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(result[name] for name in CONSUMER_OUTPUTS)


def _chain_text(error: Optional[BaseException]) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


# --- consumer -------------------------------------------------------------------


def test_one_session_walks_newest_pointer_and_back_to_newest_across_partitions(
    driver,
) -> None:
    for value in ("a0", "a1"):
        driver.append(value, partition=0)
    for value in ("b0", "b1", "b2"):
        driver.append(value, partition=1)
    cursor = new_cursor(driver)

    with compile_consumer().create_session(
        resources={"kafka_consumer": cursor}
    ) as session:
        newest = brief(row(session))
        pointed = brief(row(session, offset=1, partition=1))
        cached = brief(row(session, offset=1, partition=1))
        back = brief(row(session))
        idle = brief(row(session))

    # newest = the record the broker delivered last when positioning: b2
    assert newest == ("b2", "cam-1", True, 2, 1, False)
    assert pointed == ("b1", "cam-1", True, 1, 1, False)
    assert cached == ("b1", "cam-1", False, 1, 1, False)
    assert back == ("b2", "cam-1", True, 2, 1, False)
    assert idle == ("b2", "cam-1", False, 2, 1, False)
    assert driver.consumers[0].closes == 0
    cursor.close()
    assert driver.consumers[0].closes == 1


def test_a_reset_to_a_sequential_plan_continues_from_the_cursor_position(
    driver,
) -> None:
    driver.append("r0")
    cursor = new_cursor(driver)
    latest = compile_consumer()
    sequential = compile_consumer(read_mode="sequential")
    session = latest.create_session(resources={"kafka_consumer": cursor})
    try:
        first = row(session)["value"]
        for value in ("r1", "r2", "r3"):
            driver.append(value)
        receipt = session.apply_update(session.prepare_update(sequential, reset=True))
        switched = brief(row(session))
        for value in ("r4", "r5"):
            driver.append(value)
        stream = [brief(row(session)) for _ in range(3)]
    finally:
        session.close()

    assert receipt is not None
    assert first == "r0"
    # The cursor was positioned in latest mode, so sequential restarts at the
    # newest record (V1's mode switch), then follows the stream one by one.
    assert switched == ("r3", "cam-1", True, 3, 0, False)
    assert stream == [
        ("r4", "cam-1", True, 4, 0, False),
        ("r5", "cam-1", True, 5, 0, False),
        ("r5", "cam-1", False, 5, 0, False),
    ]
    assert len(driver.consumers) == 1 and driver.consumers[0].closes == 0
    cursor.close()


def test_two_sessions_sharing_a_cursor_split_a_sequential_stream(driver) -> None:
    driver.append("r0")
    shared = new_cursor(driver)
    own = new_cursor(driver)
    plan = compile_consumer(read_mode="sequential")
    first = plan.create_session(resources={"kafka_consumer": shared})
    second = plan.create_session(resources={"kafka_consumer": shared})
    independent = plan.create_session(resources={"kafka_consumer": own})
    try:
        for session in (first, independent):
            row(session)
        for value in ("r1", "r2", "r3", "r4"):
            driver.append(value)
        shared_values = [
            row(session)["value"] for session in (first, second, first, second)
        ]
        own_values = [row(independent)["value"] for _ in range(4)]
    finally:
        for session in (first, second, independent):
            session.close()

    assert shared_values == ["r1", "r2", "r3", "r4"]
    assert own_values == ["r1", "r2", "r3", "r4"]
    shared.close()
    own.close()


def test_a_borrowed_cursor_survives_session_close_and_reset_until_the_host_closes_it(
    driver,
) -> None:
    driver.append("r0")
    cursor = new_cursor(driver)
    plan = compile_consumer(read_mode="sequential")
    first = plan.create_session(resources={"kafka_consumer": cursor})
    second = plan.create_session(resources={"kafka_consumer": cursor})
    try:
        assert row(first)["value"] == "r0"
        first.apply_update(first.prepare_update(plan, reset=True))
        driver.append("r1")
        assert row(first)["value"] == "r1"
        first.close()
        driver.append("r2")
        after_close = row(second)
    finally:
        first.close()
        second.close()

    assert (after_close["value"], after_close["is_new"]) == ("r2", True)
    assert not cursor.closed and driver.consumers[0].closes == 0
    cursor.close()
    cursor.close()
    assert driver.consumers[0].closes == 1


def test_rebinding_with_a_new_cursor_on_reset_starts_fresh(driver) -> None:
    driver.append("r0")
    old = new_cursor(driver)
    new = new_cursor(driver)
    plan = compile_consumer()
    session = plan.create_session(resources={"kafka_consumer": old})
    try:
        before = row(session)
        session.apply_update(
            session.prepare_update(plan, resources={"kafka_consumer": new}, reset=True)
        )
        old.close()  # the host closes the old owner after its runs settled
        after = row(session)
    finally:
        session.close()

    assert (before["value"], before["is_new"]) == ("r0", True)
    assert (after["value"], after["is_new"]) == (
        "r0",
        True,
    )  # new owner, no last record
    assert [consumer.closes for consumer in driver.consumers] == [1, 0]
    new.close()


def test_a_secrets_store_password_reaches_the_client_and_errors_are_scrubbed(
    driver,
) -> None:
    store = {
        "type": "v2/environment_secrets_store@v1",
        "name": "secrets",
        "variables_storing_secrets": ["KAFKA_PASSWORD"],
    }
    plan = compile_consumer(
        store, username="reader", password="$steps.secrets.kafka_password"
    )
    driver.construct_error = RuntimeError(f"SASL login with {PASSWORD} failed")
    cursor = new_cursor(driver)

    with plan.create_session(
        resources={
            "kafka_consumer": cursor,
            "secrets": StaticSecrets({"KAFKA_PASSWORD": PASSWORD}),
        }
    ) as session:
        failed = row(session)
        driver.construct_error = None
        driver.append("r0")
        connected = row(session)

    assert failed["error_message"] == (
        "Kafka broker not reachable (SASL login with <secret KAFKA_PASSWORD> failed). "
        "Raise 'connect_timeout' if the broker needs longer to connect."
    )
    assert connected["value"] == "r0"
    assert driver.consumers[0].config["sasl.password"] == PASSWORD
    cursor.close()


def test_a_closed_cursor_keeps_its_callback_scrubbed_and_fails_even_invalid_steps(
    driver, caplog
) -> None:
    store = {
        "type": "v2/environment_secrets_store@v1",
        "name": "secrets",
        "variables_storing_secrets": ["KAFKA_PASSWORD"],
    }
    plan = compile_consumer(
        store, username="reader", password="$steps.secrets.kafka_password"
    )
    secrets = StaticSecrets({"KAFKA_PASSWORD": PASSWORD})
    cursor = new_cursor(driver)
    driver.append("r0")
    with plan.create_session(
        resources={"kafka_consumer": cursor, "secrets": secrets}
    ) as session:
        connected = row(session)
    consumer = driver.consumers[0]
    consumer.on_close = lambda: consumer.config["error_cb"](
        RuntimeError(f"SASL session for {PASSWORD} closed")
    )

    with caplog.at_level(logging.ERROR):
        cursor.close()  # the host closes the owner after its sessions stopped
    invalid = compile_consumer(username="reader-without-password")
    with invalid.create_session(resources={"kafka_consumer": cursor}) as session:
        with pytest.raises(StepExecutionError) as raised:
            session.run({})

    assert connected["value"] == "r0"
    assert "SASL session for <secret KAFKA_PASSWORD> closed" in caplog.text
    assert PASSWORD not in caplog.text
    assert "KafkaConsumerCursor is closed" in _chain_text(raised.value)
    assert [consumer.closes for consumer in driver.consumers] == [1]


def test_a_missing_cursor_or_an_unavailable_secret_fails_the_step(driver) -> None:
    with compile_consumer().create_session() as session:
        with pytest.raises(StepExecutionError) as missing:
            session.run({})
    secret_plan = _consumer_with_secret_input()
    cursor = new_cursor(driver)
    with secret_plan.create_session(resources={"kafka_consumer": cursor}) as session:
        with pytest.raises(StepExecutionError) as unavailable:
            session.run({"password": Secret.unavailable("KAFKA_PASSWORD")})

    assert "KafkaHostError" in _chain_text(missing.value)
    assert 'resources["kafka_consumer"]' in _chain_text(missing.value)
    assert "SecretUnavailableError" in _chain_text(unavailable.value)
    assert driver.consumers == []
    cursor.close()


def _consumer_with_secret_input():
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "password", "kind": ["secret"]}
        ],
        "steps": [
            {
                "type": "v2/kafka_consumer@v1",
                "name": "kafka",
                "bootstrap_servers": "broker:9092",
                "topic": TOPIC,
                "username": "reader",
                "password": "$inputs.password",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "value", "selector": "$steps.kafka.value"}
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def test_negative_offsets_fail_compilation_or_the_step(driver) -> None:
    with pytest.raises(ParamsValidationError):
        compile_consumer(offset=-1)
    cursor = new_cursor(driver)

    with compile_consumer().create_session(
        resources={"kafka_consumer": cursor}
    ) as session:
        with pytest.raises(StepExecutionError):
            session.run({"offset": -3})

    assert driver.consumers == []
    cursor.close()


# --- producer -------------------------------------------------------------------


def test_a_compiled_producer_schedules_confirms_and_keeps_background_failures(
    driver,
) -> None:
    publisher = new_publisher(driver)

    with compile_producer().create_session(
        resources={"kafka_publisher": publisher}
    ) as session:
        scheduled = row(session, message={"state": "RUNNING"})
        confirmed = row(session, message="plain", fire_and_forget=False)
        driver.hang = True
        row(session, message={"state": "LATE"})

    driver.hang = False
    driver.delivery_error = "Broker: Not enough replicas"
    stats = publisher.close(timeout=1.0)

    assert scheduled == {"error_status": False, "message": SCHEDULED_MESSAGE}
    assert confirmed == {
        "error_status": False,
        "message": "Message delivered to partition 0 at offset 1",
    }
    assert driver.producers[0].records == [
        json.dumps({"state": "RUNNING"}).encode(),
        b"plain",
        json.dumps({"state": "LATE"}).encode(),
    ]
    assert (stats.delivered, stats.failed, stats.unknown_at_close) == (2, 1, 0)
    assert publisher.failures()[0].message == "Broker: Not enough replicas"


def test_a_reset_with_an_old_generation_delivery_pending_settles_through_the_publisher(
    driver,
) -> None:
    publisher = new_publisher(driver)
    plan = compile_producer()
    session = plan.create_session(resources={"kafka_publisher": publisher})
    try:
        driver.hang = True
        old = row(session, message="old-generation")
        pending = publisher.stats().pending
        session.apply_update(session.prepare_update(plan, reset=True))
        driver.hang = False
        new = row(session, message="new-generation")
    finally:
        session.close()
    stats = publisher.close(timeout=1.0)

    assert old == new == {"error_status": False, "message": SCHEDULED_MESSAGE}
    assert pending == 1
    assert (stats.queued, stats.delivered, stats.unknown_at_close) == (2, 2, 0)
    assert driver.producers[0].flushes == [1.0]


def test_a_publisher_closed_before_its_sessions_stop_fails_the_step(driver) -> None:
    publisher = new_publisher(driver)
    plan = compile_producer()

    with plan.create_session(resources={"kafka_publisher": publisher}) as session:
        row(session, message="first")
        publisher.close(timeout=1.0)
        with pytest.raises(StepExecutionError) as raised:
            session.run({"message": "too late"})

    assert "KafkaPublisher is closed" in _chain_text(raised.value)
    assert driver.producers[0].records == [b"first"]


def test_a_publisher_shared_by_two_sessions_counts_both_and_closes_once(driver) -> None:
    publisher = new_publisher(driver)
    plan = compile_producer()
    first = plan.create_session(resources={"kafka_publisher": publisher})
    second = plan.create_session(resources={"kafka_publisher": publisher})
    try:
        row(first, message="one")
        first.close()
        row(second, message="two", fire_and_forget=False)
    finally:
        first.close()
        second.close()

    assert not publisher.closed
    stats = publisher.close(timeout=1.0)
    assert (stats.queued, stats.delivered, stats.clients_created) == (2, 2, 1)
    assert len(driver.producers) == 1


def test_disabled_sinks_need_no_publisher(driver) -> None:
    with compile_producer().create_session(
        resources={"disable_sinks": True}
    ) as session:
        result = row(session, message="ignored")

    assert result == {
        "error_status": False,
        "message": "Sink was disabled by workflow execution policy",
    }
    assert driver.producers == []


def test_selected_fire_and_forget_leaves_restrictions_incomplete() -> None:
    restrictions = (
        discover_workload(compile_producer())
        .step("$steps.sink")
        .describe()["restrictions"]
    )
    consumer = (
        discover_workload(compile_consumer())
        .step("$steps.kafka")
        .describe()["restrictions"]
    )

    assert restrictions["complete"] is False
    assert [item["code"] for item in restrictions["items"]] == [
        "unavailable_on_hosted_platform"
    ]
    assert consumer["complete"] is True
    assert sorted(item["code"] for item in consumer["items"]) == [
        "connection_and_state_rebuilt_per_request",
        "unavailable_on_hosted_platform",
    ]


def test_concurrent_sessions_publish_through_one_publisher_without_loss(driver) -> None:
    publisher = new_publisher(driver)
    plan = compile_producer()
    errors: List[BaseException] = []

    def publish_many(tag: str) -> None:
        try:
            with plan.create_session(
                resources={"kafka_publisher": publisher}
            ) as session:
                for index in range(20):
                    row(
                        session,
                        message=f"{tag}-{index}",
                        fire_and_forget=index % 2 == 0,
                    )
        except BaseException as error:  # reported below
            errors.append(error)

    threads = [threading.Thread(target=publish_many, args=(tag,)) for tag in "ab"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    stats = publisher.close(timeout=1.0)

    assert errors == []
    assert (stats.queued, stats.delivered, stats.pending, stats.unknown_at_close) == (
        40,
        40,
        0,
        0,
    )
    assert len(driver.producers[0].records) == 40
