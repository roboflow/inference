"""Set KAFKA_TEST_BOOTSTRAP_SERVERS (e.g. localhost:9092) to a disposable broker with
topic auto-creation enabled: `docker run -d -p 9092:9092 apache/kafka:3.8.0`."""

import json
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import numpy as np
import pytest

from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    IMAGE_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_consumer.v1 import (
    KafkaConsumerBlockV1,
)

BOOTSTRAP = os.environ.get("KAFKA_TEST_BOOTSTRAP_SERVERS")

pytestmark = pytest.mark.skipif(
    not BOOTSTRAP,
    reason="Set KAFKA_TEST_BOOTSTRAP_SERVERS to a disposable Kafka broker",
)


# --------------------------------------------------------------------------------------
# Broker helpers
# --------------------------------------------------------------------------------------


class Publisher:
    """Thin producer wrapper returning the (partition, offset) pointer of each record."""

    def __init__(self, topic: str):
        from confluent_kafka import Producer

        self.topic = topic
        self._producer = Producer({"bootstrap.servers": BOOTSTRAP})

    def publish(self, value: Any, key: Optional[str] = None) -> Tuple[int, int]:
        pointer: List[Tuple[int, int]] = []

        def on_delivery(error, message):
            assert error is None, error
            pointer.append((message.partition(), message.offset()))

        if not isinstance(value, (str, bytes)):
            value = json.dumps(value)
        self._producer.produce(
            self.topic, value=value, key=key, on_delivery=on_delivery
        )
        self._producer.flush(10)
        assert pointer, "delivery report missing"
        return pointer[0]


@pytest.fixture
def publisher() -> Publisher:
    return Publisher(topic=f"inference-test-{uuid4().hex}")


def consumer_step(topic: str, **overrides) -> Dict[str, Any]:
    step = {
        "type": "roboflow_enterprise/kafka_consumer@v1",
        "name": "state",
        "bootstrap_servers": "$inputs.kafka_bootstrap",
        "topic": topic,
        "connect_timeout": 10.0,
    }
    step.update(overrides)
    return step


def run_kafka_block(block: KafkaConsumerBlockV1, topic: str, **overrides):
    kwargs = {
        "bootstrap_servers": BOOTSTRAP,
        "topic": topic,
        "connect_timeout": 10.0,
        "poll_timeout": 0.5,
    }
    kwargs.update(overrides)
    return block.run(**kwargs)


def run_until_new(block: KafkaConsumerBlockV1, topic: str, attempts: int = 20, **kw):
    """Poll a few times: the broker may take a moment to serve a just-produced record."""
    result = None
    for _ in range(attempts):
        result = run_kafka_block(block, topic, **kw)
        if result["is_new"]:
            return result
        time.sleep(0.1)
    return result


# --------------------------------------------------------------------------------------
# Block-level tests against the real broker
# --------------------------------------------------------------------------------------


def test_first_run_reads_newest_existing_record(publisher: Publisher) -> None:
    # given
    for value in ("A", "B"):
        publisher.publish({"state": value}, key="cam-1")
    last_pointer = publisher.publish({"state": "C"}, key="cam-1")
    block = KafkaConsumerBlockV1()

    try:
        # when
        result = run_kafka_block(block, publisher.topic)

        # then
        assert result["error_status"] is False, result["error_message"]
        assert result["payload"] == {"state": "C"}
        assert result["key"] == "cam-1"
        assert result["is_new"] is True
        assert (result["partition"], result["offset"]) == last_pointer
    finally:
        block.close()


def test_latest_mode_follows_new_records_and_key_filter(publisher: Publisher) -> None:
    # given
    publisher.publish({"state": "seed"}, key="cam-1")
    block = KafkaConsumerBlockV1()

    try:
        first = run_kafka_block(block, publisher.topic, key_filter="cam-1")
        repeat = run_kafka_block(block, publisher.topic, key_filter="cam-1")
        publisher.publish({"state": "other-camera"}, key="cam-2")
        publisher.publish({"state": "RUNNING"}, key="cam-1")

        # when
        result = run_until_new(block, publisher.topic, key_filter="cam-1")

        # then
        assert first["payload"] == {"state": "seed"} and first["is_new"] is True
        assert repeat["payload"] == {"state": "seed"} and repeat["is_new"] is False
        assert result["payload"] == {"state": "RUNNING"}
        assert result["key"] == "cam-1"
    finally:
        block.close()


def test_pointer_from_delivery_report_reads_exact_record(publisher: Publisher) -> None:
    # given
    pointers = [publisher.publish({"state": value}) for value in ("A", "B", "C", "D")]
    target_partition, target_offset = pointers[1]
    block = KafkaConsumerBlockV1()

    try:
        # when
        pointed = run_kafka_block(
            block, publisher.topic, offset=target_offset, partition=target_partition
        )
        cached = run_kafka_block(
            block, publisher.topic, offset=target_offset, partition=target_partition
        )
        out_of_range = run_kafka_block(
            block, publisher.topic, offset=999, partition=target_partition
        )

        # then
        assert pointed["error_status"] is False, pointed["error_message"]
        assert pointed["payload"] == {"state": "B"}
        assert (pointed["partition"], pointed["offset"]) == pointers[1]
        assert pointed["is_new"] is True
        assert cached["payload"] == {"state": "B"} and cached["is_new"] is False
        assert out_of_range["error_status"] is True
        assert "out of range" in out_of_range["error_message"]
        assert out_of_range["payload"] == {"state": "B"}
    finally:
        block.close()


def test_leaving_pointer_mode_returns_newest_existing_record(
    publisher: Publisher,
) -> None:
    # given
    pointers = [publisher.publish({"state": value}) for value in ("A", "B", "C")]
    block = KafkaConsumerBlockV1()

    try:
        run_kafka_block(block, publisher.topic, offset=pointers[0][1])
        publisher.publish({"state": "during-pointer-mode"})
        back_to_latest = run_kafka_block(block, publisher.topic)
        publisher.publish({"state": "after"})

        # when
        result = run_until_new(block, publisher.topic)

        # then
        assert back_to_latest["payload"] == {"state": "during-pointer-mode"}
        assert back_to_latest["is_new"] is True
        assert result["payload"] == {"state": "after"}
    finally:
        block.close()


def test_sequential_mode_delivers_every_record_in_order(publisher: Publisher) -> None:
    # given
    publisher.publish({"n": 0})
    block = KafkaConsumerBlockV1()

    try:
        first = run_kafka_block(block, publisher.topic, read_mode="sequential")
        for n in (1, 2, 3):
            publisher.publish({"n": n})

        # when
        delivered = []
        for _ in range(3):
            result = run_until_new(block, publisher.topic, read_mode="sequential")
            delivered.append(result["payload"])
        idle = run_kafka_block(block, publisher.topic, read_mode="sequential")

        # then
        assert first["payload"] == {"n": 0}
        assert delivered == [{"n": 1}, {"n": 2}, {"n": 3}]
        assert idle["payload"] == {"n": 3} and idle["is_new"] is False
    finally:
        block.close()


def test_sequential_mode_pointer_starts_reading_there(publisher: Publisher) -> None:
    pointers = [publisher.publish({"state": value}) for value in ("A", "B", "C", "D")]
    partition, offset = pointers[1]
    block = KafkaConsumerBlockV1()

    try:
        started = run_kafka_block(
            block,
            publisher.topic,
            read_mode="sequential",
            offset=offset,
            partition=partition,
        )
        following = [
            run_until_new(
                block,
                publisher.topic,
                read_mode="sequential",
                offset=offset,
                partition=partition,
            )["payload"]
            for _ in range(2)
        ]

        assert started["payload"] == {"state": "B"}
        assert following == [{"state": "C"}, {"state": "D"}]
    finally:
        block.close()


def test_two_instances_both_receive_every_record(publisher: Publisher) -> None:
    # given
    publisher.publish({"state": "seed"})
    camera_a, camera_b = KafkaConsumerBlockV1(), KafkaConsumerBlockV1()

    try:
        run_kafka_block(camera_a, publisher.topic)
        run_kafka_block(camera_b, publisher.topic)
        publisher.publish({"state": "RUNNING"})

        # when
        result_a = run_until_new(camera_a, publisher.topic)
        result_b = run_until_new(camera_b, publisher.topic)

        # then
        assert result_a["payload"] == {"state": "RUNNING"} and result_a["is_new"]
        assert result_b["payload"] == {"state": "RUNNING"} and result_b["is_new"]
    finally:
        camera_a.close()
        camera_b.close()


# --------------------------------------------------------------------------------------
# Workflow-level tests through the Execution Engine
# --------------------------------------------------------------------------------------


class DetectorStubManifest(WorkflowBlockManifest):
    type: Literal["test/detector_stub@v1"]
    image: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="predictions", kind=[STRING_KIND])]


class DetectorStubBlock(WorkflowBlock):
    """Stands in for a model step so the test needs no weights or API key."""

    @classmethod
    def get_manifest(cls):
        return DetectorStubManifest

    def run(self, image: WorkflowImageData):
        return {"predictions": f"detections for {image.numpy_image.shape}"}


class DictionaryStubManifest(WorkflowBlockManifest):
    type: Literal["test/dictionary_stub@v1"]
    data: Selector(kind=[DICTIONARY_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="received", kind=[DICTIONARY_KIND])]


class DictionaryStubBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls):
        return DictionaryStubManifest

    def run(self, data: Dict[str, Any]):
        return {"received": {"echo": data}}


@pytest.fixture
def enterprise_blocks_with_stubs(monkeypatch):
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
    original_load = blocks_loader.load_blocks
    monkeypatch.setattr(
        blocks_loader,
        "load_blocks",
        lambda: original_load() + [DetectorStubBlock, DictionaryStubBlock],
    )
    blocks_loader.load_core_workflow_blocks.cache_clear()
    yield
    blocks_loader.load_core_workflow_blocks.cache_clear()
    with COMPILATION_CACHE._cache_lock:
        COMPILATION_CACHE._cache.clear()
        COMPILATION_CACHE._keys_buffer.clear()


def gating_workflow(topic: str) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "kafka_bootstrap"},
        ],
        "steps": [
            consumer_step(topic),
            {
                "type": "roboflow_core/continue_if@v1",
                "name": "gate",
                "condition_statement": {
                    "type": "StatementGroup",
                    "statements": [
                        {
                            "type": "UnaryStatement",
                            "operand": {
                                "type": "DynamicOperand",
                                "operand_name": "raw",
                                "operations": [
                                    {
                                        "type": "StringMatches",
                                        "regex": '"state"\\s*:\\s*"RUNNING"',
                                    }
                                ],
                            },
                            "operator": {"type": "(Boolean) is True"},
                        }
                    ],
                },
                "evaluation_parameters": {"raw": "$steps.state.value"},
                "next_steps": ["$steps.model"],
            },
            {
                "type": "test/detector_stub@v1",
                "name": "model",
                "image": "$inputs.image",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "state", "selector": "$steps.state.payload"},
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.model.predictions",
            },
        ],
    }


def run_workflow(definition: Dict[str, Any]) -> Dict[str, Any]:
    engine = ExecutionEngine.init(workflow_definition=definition)
    results = engine.run(
        runtime_parameters={
            "image": np.zeros((32, 48, 3), dtype=np.uint8),
            "kafka_bootstrap": BOOTSTRAP,
        }
    )
    assert len(results) == 1
    return results[0]


def test_gating_workflow_runs_model_only_while_running(
    publisher: Publisher, enterprise_blocks_with_stubs
) -> None:
    # given
    publisher.publish({"state": "RUNNING"})

    # when
    running = run_workflow(gating_workflow(publisher.topic))
    publisher.publish({"state": "PAUSED"})
    paused = run_workflow(gating_workflow(publisher.topic))

    # then
    assert running["state"] == {"state": "RUNNING"}
    assert running["predictions"] == "detections for (32, 48, 3)"
    assert paused["state"] == {"state": "PAUSED"}
    assert paused.get("predictions") is None


def test_payload_flows_into_dictionary_consuming_block(
    publisher: Publisher, enterprise_blocks_with_stubs
) -> None:
    # given
    publisher.publish({"state": "RUNNING", "sku": "A-17"})
    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "kafka_bootstrap"}],
        "steps": [
            consumer_step(publisher.topic),
            {
                "type": "test/dictionary_stub@v1",
                "name": "sink",
                "data": "$steps.state.payload",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "received",
                "selector": "$steps.sink.received",
            },
            {"type": "JsonField", "name": "is_new", "selector": "$steps.state.is_new"},
        ],
    }

    # when
    engine = ExecutionEngine.init(workflow_definition=definition)
    result = engine.run(runtime_parameters={"kafka_bootstrap": BOOTSTRAP})

    # then
    assert result == [
        {"received": {"echo": {"state": "RUNNING", "sku": "A-17"}}, "is_new": True}
    ]


def test_pointer_workflow_reads_record_named_by_inputs(
    publisher: Publisher, enterprise_blocks_with_stubs
) -> None:
    # given
    pointers = [publisher.publish({"state": value}) for value in ("A", "B", "C")]
    partition, offset = pointers[0]
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "kafka_bootstrap"},
            {"type": "WorkflowParameter", "name": "state_offset"},
            {"type": "WorkflowParameter", "name": "state_partition"},
        ],
        "steps": [
            consumer_step(
                publisher.topic,
                offset="$inputs.state_offset",
                partition="$inputs.state_partition",
            )
        ],
        "outputs": [
            {"type": "JsonField", "name": "state", "selector": "$steps.state.payload"},
            {"type": "JsonField", "name": "offset", "selector": "$steps.state.offset"},
        ],
    }

    # when
    engine = ExecutionEngine.init(workflow_definition=definition)
    result = engine.run(
        runtime_parameters={
            "kafka_bootstrap": BOOTSTRAP,
            "state_offset": offset,
            "state_partition": partition,
        }
    )

    # then
    assert result == [{"state": {"state": "A"}, "offset": offset}]
