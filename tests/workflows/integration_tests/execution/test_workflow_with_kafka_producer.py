"""Set KAFKA_TEST_BOOTSTRAP_SERVERS (e.g. localhost:9092) to a disposable broker with
topic auto-creation enabled: `docker run -d -p 9092:9092 apache/kafka:3.8.0`."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pytest

from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_consumer.v1 import (
    KafkaConsumerBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_producer.v1 import (
    KafkaProducerSinkBlockV1,
)

BOOTSTRAP = os.environ.get("KAFKA_TEST_BOOTSTRAP_SERVERS")

pytestmark = pytest.mark.skipif(
    not BOOTSTRAP,
    reason="Set KAFKA_TEST_BOOTSTRAP_SERVERS to a disposable Kafka broker",
)


# --------------------------------------------------------------------------------------
# Broker helpers
# --------------------------------------------------------------------------------------


def new_topic() -> str:
    return f"inference-producer-test-{uuid4().hex}"


def read_records(
    topic: str, expected: int, timeout: float = 10.0
) -> List[Tuple[Optional[bytes], bytes, Any]]:
    """Read `expected` records from offset 0 of partition 0 with a raw consumer."""
    from confluent_kafka import Consumer, TopicPartition

    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"producer-test-{uuid4().hex}",
            "enable.auto.commit": False,
        }
    )
    records: List[Tuple[Optional[bytes], bytes, Any]] = []
    try:
        consumer.assign([TopicPartition(topic, 0, 0)])
        deadline = time.monotonic() + timeout
        while len(records) < expected and time.monotonic() < deadline:
            message = consumer.poll(0.5)
            if message is None or message.error() is not None:
                continue
            records.append((message.key(), message.value(), message.headers()))
    finally:
        consumer.close()
    return records


def producer() -> KafkaProducerSinkBlockV1:
    return KafkaProducerSinkBlockV1(None, None)


def publish(block: KafkaProducerSinkBlockV1, topic: str, **overrides) -> Dict[str, Any]:
    kwargs = {
        "bootstrap_servers": BOOTSTRAP,
        "topic": topic,
        "message": {"state": "RUNNING"},
        "fire_and_forget": False,
        "timeout": 10.0,
    }
    kwargs.update(overrides)
    return block.run(**kwargs)


# --------------------------------------------------------------------------------------
# Block-level tests against the real broker
# --------------------------------------------------------------------------------------


def test_confirmed_delivery_round_trips_key_value_and_headers() -> None:
    # given
    topic = new_topic()
    block = producer()

    try:
        # when
        result = publish(
            block,
            topic,
            message={"state": "RUNNING", "n": 1},
            key="cam-1",
            headers={"source": "inference", "schema": "v1"},
        )

        # then
        assert result["error_status"] is False, result["message"]
        assert result["message"] == "Message delivered to partition 0 at offset 0"
        records = read_records(topic, expected=1)
        assert records == [
            (
                b"cam-1",
                json.dumps({"state": "RUNNING", "n": 1}).encode("utf-8"),
                [("source", b"inference"), ("schema", b"v1")],
            )
        ]
    finally:
        block.close()


def test_fire_and_forget_publishes_in_the_background() -> None:
    topic = new_topic()
    block = producer()

    try:
        results = [
            publish(block, topic, message=f"m{i}", fire_and_forget=True)
            for i in range(3)
        ]
        block.close()  # flushes whatever is still queued

        assert all(r["error_status"] is False for r in results)
        assert all(r["message"] == "Message scheduled for delivery" for r in results)
        values = [value for _, value, _ in read_records(topic, expected=3)]
        assert values == [b"m0", b"m1", b"m2"]
    finally:
        block.close()


def test_one_producer_instance_serves_many_runs_in_order() -> None:
    topic = new_topic()
    block = producer()

    try:
        for i in range(5):
            result = publish(block, topic, message={"n": i}, key="cam-1")
            assert result["error_status"] is False, result["message"]
            assert result["message"].endswith(f"at offset {i}")

        values = [json.loads(value) for _, value, _ in read_records(topic, expected=5)]
        assert values == [{"n": i} for i in range(5)]
    finally:
        block.close()


def test_changed_connection_is_rejected_without_publishing() -> None:
    topic = new_topic()
    block = producer()

    try:
        publish(block, topic, message="first")
        result = publish(block, topic, message="second", acks="1")

        assert result["error_status"] is True
        assert "changed between runs" in result["message"]
        values = [value for _, value, _ in read_records(topic, expected=1, timeout=3)]
        assert values == [b"first"]
    finally:
        block.close()


# --------------------------------------------------------------------------------------
# Workflow-level tests through the Execution Engine
# --------------------------------------------------------------------------------------


@pytest.fixture
def enterprise_blocks(monkeypatch):
    from inference.core.env import ENTERPRISE_BLOCKS_PLUGIN
    from inference.core.workflows.execution_engine.introspection import blocks_loader
    from inference.core.workflows.execution_engine.v1.compiler.core import (
        COMPILATION_CACHE,
    )

    # enterprise blocks load through the generic plugin mechanism (see env.py)
    plugins = blocks_loader.get_plugin_modules()
    if ENTERPRISE_BLOCKS_PLUGIN not in plugins:
        plugins.append(ENTERPRISE_BLOCKS_PLUGIN)
    monkeypatch.setenv(blocks_loader.WORKFLOWS_PLUGINS_ENV, ",".join(plugins))
    blocks_loader.load_core_workflow_blocks.cache_clear()
    yield
    blocks_loader.load_core_workflow_blocks.cache_clear()
    with COMPILATION_CACHE._cache_lock:
        COMPILATION_CACHE._cache.clear()
        COMPILATION_CACHE._keys_buffer.clear()


def producer_workflow(topic: str) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "kafka_bootstrap"},
            {"type": "WorkflowParameter", "name": "camera_id"},
            {"type": "WorkflowParameter", "name": "payload"},
        ],
        "steps": [
            {
                "type": "roboflow_enterprise/kafka_producer_sink@v1",
                "name": "publish",
                "bootstrap_servers": "$inputs.kafka_bootstrap",
                "topic": topic,
                "message": "$inputs.payload",
                "key": "$inputs.camera_id",
                "headers": {"source": "inference"},
                "fire_and_forget": False,
                "timeout": 10.0,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "error_status",
                "selector": "$steps.publish.error_status",
            },
            {
                "type": "JsonField",
                "name": "message",
                "selector": "$steps.publish.message",
            },
        ],
    }


def test_workflow_publishes_resolved_inputs(enterprise_blocks) -> None:
    # given
    topic = new_topic()
    engine = ExecutionEngine.init(workflow_definition=producer_workflow(topic))

    # when
    result = engine.run(
        runtime_parameters={
            "kafka_bootstrap": BOOTSTRAP,
            "camera_id": "cam-7",
            "payload": {"state": "RUNNING", "count": 3},
        }
    )

    # then
    assert result == [
        {
            "error_status": False,
            "message": "Message delivered to partition 0 at offset 0",
        }
    ]
    records = read_records(topic, expected=1)
    assert records == [
        (
            b"cam-7",
            json.dumps({"state": "RUNNING", "count": 3}).encode("utf-8"),
            [("source", b"inference")],
        )
    ]


def consumer_workflow(topic: str) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "kafka_bootstrap"}],
        "steps": [
            {
                "type": "roboflow_enterprise/kafka_consumer@v1",
                "name": "state",
                "bootstrap_servers": "$inputs.kafka_bootstrap",
                "topic": topic,
                "read_mode": "sequential",
                "connect_timeout": 10.0,
                "poll_timeout": 0.5,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "payload",
                "selector": "$steps.state.payload",
            },
            {"type": "JsonField", "name": "key", "selector": "$steps.state.key"},
            {"type": "JsonField", "name": "is_new", "selector": "$steps.state.is_new"},
        ],
    }


def test_producer_and_consumer_blocks_interoperate(enterprise_blocks) -> None:
    # given: the producer workflow publishes three records
    topic = new_topic()
    publisher = ExecutionEngine.init(workflow_definition=producer_workflow(topic))
    for n in range(3):
        result = publisher.run(
            runtime_parameters={
                "kafka_bootstrap": BOOTSTRAP,
                "camera_id": "cam-1",
                "payload": {"n": n},
            }
        )
        assert result[0]["error_status"] is False, result[0]["message"]

    # when: one consumer workflow (one long-lived block instance) reads sequentially
    reader = ExecutionEngine.init(workflow_definition=consumer_workflow(topic))
    first = reader.run(runtime_parameters={"kafka_bootstrap": BOOTSTRAP})[0]
    publisher.run(
        runtime_parameters={
            "kafka_bootstrap": BOOTSTRAP,
            "camera_id": "cam-1",
            "payload": {"n": 3},
        }
    )
    publisher.run(
        runtime_parameters={
            "kafka_bootstrap": BOOTSTRAP,
            "camera_id": "cam-1",
            "payload": {"n": 4},
        }
    )
    following = []
    deadline = time.monotonic() + 10
    while len(following) < 2 and time.monotonic() < deadline:
        step = reader.run(runtime_parameters={"kafka_bootstrap": BOOTSTRAP})[0]
        if step["is_new"]:
            following.append(step["payload"])
        else:
            time.sleep(0.1)

    # then: sequential mode starts at the newest existing record and delivers the rest
    assert first == {"payload": {"n": 2}, "key": "cam-1", "is_new": True}
    assert following == [{"n": 3}, {"n": 4}]


def test_consumer_block_reads_what_the_producer_block_published() -> None:
    """Block-level round trip without the engine: latest mode sees the newest record."""
    topic = new_topic()
    publisher = producer()
    reader = KafkaConsumerBlockV1()

    try:
        for state in ("A", "B", "C"):
            publish(publisher, topic, message={"state": state}, key="cam-1")
        result = reader.run(
            bootstrap_servers=BOOTSTRAP,
            topic=topic,
            connect_timeout=10.0,
            poll_timeout=0.5,
        )

        assert result["error_status"] is False, result["error_message"]
        assert result["payload"] == {"state": "C"}
        assert result["key"] == "cam-1"
        assert result["offset"] == 2
    finally:
        publisher.close()
        reader.close()
