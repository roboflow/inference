import json
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.enterprise_blocks.sinks import kafka_common
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_common import (
    derive_msk_region,
    msk_token_callback,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_consumer import v1
from inference.enterprise.workflows.enterprise_blocks.sinks.kafka_consumer.v1 import (
    BlockManifest,
    KafkaConsumerBlockV1,
)

BOOTSTRAP = "broker-1:9092,broker-2:9092"
TOPIC = "line1.state"
OUTPUT_KEYS = [
    "value",
    "payload",
    "key",
    "is_new",
    "offset",
    "partition",
    "error_status",
    "error_message",
]


# --------------------------------------------------------------------------------------
# Fake broker / consumer standing in for confluent_kafka
# --------------------------------------------------------------------------------------


class FakeTopicPartition:
    def __init__(self, topic: str, partition: int, offset: int = -1001):
        self.topic = topic
        self.partition = partition
        self.offset = offset


class FakeMessage:
    def __init__(self, partition: int, offset: int, key: Any, value: Any, error=None):
        self._partition = partition
        self._offset = offset
        self._key = key
        self._value = value
        self._error = error

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def key(self):
        return self._key

    def value(self):
        return self._value

    def error(self):
        return self._error


class FakeBroker:
    """Per-partition append-only logs. A `None` slot is a record removed by compaction."""

    def __init__(self):
        self.logs: Dict[str, Dict[int, List[Optional[Tuple[Any, Any]]]]] = {}
        self.low_watermarks: Dict[Tuple[str, int], int] = {}
        self.fail_connect: Optional[Exception] = None
        self.watermark_delay: float = 0.0
        self.consumers: List["FakeConsumer"] = []

    def create_topic(self, topic: str, partitions: int = 1) -> None:
        self.logs[topic] = {p: [] for p in range(partitions)}

    def produce(
        self, topic: str, value: Any, key: Any = None, partition: int = 0
    ) -> int:
        if topic not in self.logs:
            self.create_topic(topic)
        if isinstance(value, str):
            value = value.encode("utf-8")
        if isinstance(key, str):
            key = key.encode("utf-8")
        log = self.logs[topic][partition]
        log.append((key, value))
        return len(log) - 1

    def compact_away(self, topic: str, partition: int, offset: int) -> None:
        self.logs[topic][partition][offset] = None

    def set_low_watermark(self, topic: str, partition: int, low: int) -> None:
        self.low_watermarks[(topic, partition)] = low


class FakeConsumer:
    def __init__(self, broker: FakeBroker, config: Dict[str, Any]):
        self.broker = broker
        self.config = config
        self.positions: Dict[int, int] = {}
        self.topic: Optional[str] = None
        self.assign_calls: List[List[Tuple[int, int]]] = []
        self.poll_calls = 0
        self.poll_timeouts: List[float] = []
        self.watermark_timeouts: List[float] = []
        self.closed = False
        broker.consumers.append(self)

    def list_topics(self, topic: str, timeout: float):
        if self.broker.fail_connect is not None:
            raise self.broker.fail_connect
        if topic not in self.broker.logs:
            return SimpleNamespace(
                topics={topic: SimpleNamespace(partitions={}, error="Unknown topic")}
            )
        return SimpleNamespace(
            topics={
                topic: SimpleNamespace(
                    partitions={p: None for p in self.broker.logs[topic]}, error=None
                )
            }
        )

    def get_watermark_offsets(self, tp: FakeTopicPartition, timeout: float):
        self.watermark_timeouts.append(timeout)
        if self.broker.watermark_delay:
            time.sleep(self.broker.watermark_delay)
        log = self.broker.logs[tp.topic][tp.partition]
        low = self.broker.low_watermarks.get((tp.topic, tp.partition), 0)
        return low, len(log)

    def assign(self, partitions: List[FakeTopicPartition]) -> None:
        self.assign_calls.append([(tp.partition, tp.offset) for tp in partitions])
        self.topic = partitions[0].topic
        self.positions = {tp.partition: tp.offset for tp in partitions}

    def poll(self, timeout: float):
        self.poll_calls += 1
        self.poll_timeouts.append(timeout)
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
                if isinstance(value, Exception):
                    return FakeMessage(partition, position - 1, key, None, error=value)
                return FakeMessage(partition, position - 1, key, value)
            self.positions[partition] = position
        return None

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def broker():
    fake_broker = FakeBroker()
    fake_module = SimpleNamespace(
        Consumer=lambda config: FakeConsumer(fake_broker, config),
        TopicPartition=FakeTopicPartition,
        OFFSET_END=-1,
    )
    with patch.object(v1, "confluent_kafka", fake_module):
        yield fake_broker


def run(block: KafkaConsumerBlockV1, **overrides) -> Dict[str, Any]:
    kwargs = {"bootstrap_servers": BOOTSTRAP, "topic": TOPIC}
    kwargs.update(overrides)
    return block.run(**kwargs)


def state(value: str) -> str:
    return json.dumps({"state": value})


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def manifest(**overrides) -> BlockManifest:
    data = {
        "type": "roboflow_enterprise/kafka_consumer@v1",
        "name": "kafka",
        "bootstrap_servers": BOOTSTRAP,
        "topic": TOPIC,
    }
    data.update(overrides)
    return BlockManifest.model_validate(data)


def test_manifest_defaults() -> None:
    result = manifest()

    assert result.provider == "Self-hosted"
    assert result.username is None and result.password is None
    assert result.aws_region is None
    assert result.key_filter is None
    assert result.offset is None and result.partition == 0
    assert result.poll_timeout == 0.01 and result.connect_timeout == 5.0


@pytest.mark.parametrize("provider", ["Self-hosted", "AWS MSK"])
def test_manifest_accepts_known_providers(provider: str) -> None:
    assert manifest(provider=provider).provider == provider


@pytest.mark.parametrize("read_mode", ["latest", "sequential"])
def test_manifest_accepts_known_read_modes(read_mode: str) -> None:
    assert manifest(read_mode=read_mode).read_mode == read_mode


def test_manifest_defaults_to_latest_read_mode_and_rejects_unknown() -> None:
    assert manifest().read_mode == "latest"
    with pytest.raises(ValidationError):
        manifest(read_mode="from-beginning")


def test_manifest_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        manifest(provider="Confluent Cloud")


@pytest.mark.parametrize(
    "field, value",
    [
        ("offset", -1),
        ("offset", True),
        ("offset", 1.5),
        ("offset", "7"),
        ("partition", -3),
        ("partition", 2.0),
        ("poll_timeout", "fast"),
        ("poll_timeout", float("inf")),
        ("poll_timeout", -0.1),
        ("connect_timeout", 0),
        ("connect_timeout", True),
    ],
)
def test_manifest_rejects_invalid_numeric_values(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        manifest(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("offset", "$inputs.offset"),
        ("partition", "$steps.a.partition"),
        ("poll_timeout", "$inputs.poll"),
        ("connect_timeout", "$inputs.connect"),
    ],
)
def test_manifest_lets_selectors_bypass_numeric_validation(
    field: str, value: str
) -> None:
    assert getattr(manifest(**{field: value}), field) == value


def test_manifest_conditional_fields_metadata() -> None:
    properties = BlockManifest.model_json_schema()["properties"]

    for field in ("username", "password"):
        assert properties[field]["relevant_for"] == {
            "provider": {"values": ["Self-hosted"], "required": False}
        }
    assert properties["aws_region"]["relevant_for"] == {
        "provider": {"values": ["AWS MSK"], "required": False}
    }
    assert properties["password"]["private"] is True
    for field in ("ssl_ca_location", "poll_timeout", "connect_timeout"):
        assert properties[field]["additional_section"] is True
    for field in ("bootstrap_servers", "topic", "provider", "read_mode"):
        assert properties[field]["always_visible"] is True


def test_manifest_describe_outputs_order() -> None:
    assert [o.name for o in BlockManifest.describe_outputs()] == OUTPUT_KEYS


def test_manifest_restrictions_are_soft_and_runtime_keyed() -> None:
    restrictions = [r.to_dict() for r in BlockManifest.get_restrictions()]

    assert len(restrictions) == 1
    assert restrictions[0]["severity"] == "soft"
    assert set(restrictions[0]["applies_to_runtimes"]) == {
        "self_hosted_cpu",
        "self_hosted_gpu",
        "dedicated_deployment",
    }


def test_block_declares_no_init_parameters() -> None:
    assert KafkaConsumerBlockV1.get_init_parameters() == []
    assert KafkaConsumerBlockV1.get_manifest() is BlockManifest


# --------------------------------------------------------------------------------------
# Latest mode
# --------------------------------------------------------------------------------------


def test_first_run_returns_newest_existing_record(broker: FakeBroker) -> None:
    # given
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, state(value), key="cam-1")
    block = KafkaConsumerBlockV1()

    # when
    result = run(block)

    # then
    assert set(result) == set(OUTPUT_KEYS)
    assert result["value"] == state("C")
    assert result["payload"] == {"state": "C"}
    assert result["key"] == "cam-1"
    assert result["is_new"] is True
    assert result["offset"] == 2 and result["partition"] == 0
    assert result["error_status"] is False and result["error_message"] is None
    assert broker.consumers[0].assign_calls[0] == [(0, 2)]


def test_first_run_on_empty_topic_returns_empty_outputs(broker: FakeBroker) -> None:
    # given
    broker.create_topic(TOPIC)
    block = KafkaConsumerBlockV1()

    # when
    result = run(block)

    # then
    assert result == {
        "value": None,
        "payload": {},
        "key": None,
        "is_new": False,
        "offset": None,
        "partition": None,
        "error_status": False,
        "error_message": None,
    }
    assert broker.consumers[0].assign_calls[0] == [(0, 0)]


def test_first_poll_waits_for_the_fetch_when_a_record_exists(
    broker: FakeBroker,
) -> None:
    # given
    broker.produce(TOPIC, "A")
    block = KafkaConsumerBlockV1()

    # when
    run(block, poll_timeout=0.01, connect_timeout=7.0)
    run(block, poll_timeout=0.01, connect_timeout=7.0)

    # then
    timeouts = broker.consumers[0].poll_timeouts
    assert timeouts[0] == 0.5  # bounded slice of connect_timeout, not poll_timeout
    assert all(t in (0.01, 0) for t in timeouts[1:])


def test_first_poll_does_not_wait_on_an_empty_topic(broker: FakeBroker) -> None:
    broker.create_topic(TOPIC)

    run(KafkaConsumerBlockV1(), poll_timeout=0.01, connect_timeout=7.0)

    assert broker.consumers[0].poll_timeouts[0] == 0.01


def test_leaving_pointer_mode_waits_for_the_newest_record_once(
    broker: FakeBroker,
) -> None:
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()

    run(block, offset=1, connect_timeout=7.0)
    back = run(block, poll_timeout=0.01, connect_timeout=7.0)
    again = run(block, poll_timeout=0.01, connect_timeout=7.0)

    consumer = broker.consumers[0]
    # repositioning at the newest record waits in bounded slices, then settles
    assert back["value"] == "C" and back["is_new"] is True
    assert again["is_new"] is False
    assert consumer.poll_timeouts[-1] == 0.01


def test_repeated_run_without_new_record_repeats_last_with_is_new_false(
    broker: FakeBroker,
) -> None:
    # given
    broker.produce(TOPIC, state("RUNNING"))
    block = KafkaConsumerBlockV1()
    first = run(block)

    # when
    second = run(block)

    # then
    assert first["is_new"] is True
    assert second["is_new"] is False
    assert second["value"] == first["value"] == state("RUNNING")
    assert second["payload"] == {"state": "RUNNING"}
    assert len(broker.consumers) == 1


def test_latest_mode_keeps_only_newest_of_several_records(broker: FakeBroker) -> None:
    # given
    broker.produce(TOPIC, "old")
    block = KafkaConsumerBlockV1()
    run(block)
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)

    # when
    result = run(block)

    # then
    assert result["value"] == "C"
    assert result["offset"] == 3
    assert result["is_new"] is True


def test_backlog_over_the_cap_skips_to_the_newest_record(broker: FakeBroker) -> None:
    # given a block that has read the topic once
    broker.produce(TOPIC, "seed")
    block = KafkaConsumerBlockV1()
    run(block)
    # and far more records than one run may drain arrived since
    for i in range(2500):
        broker.produce(TOPIC, f"m{i}")

    # when
    result = run(block)
    follow_up = run(block)

    # then: the newest record is reported, not the 1000th of the backlog
    assert result["value"] == "m2499"
    assert result["offset"] == 2500
    assert result["is_new"] is True
    assert result["error_status"] is False
    assert broker.consumers[0].assign_calls[-1] == [(0, 2500)]
    assert follow_up["value"] == "m2499" and follow_up["is_new"] is False


def test_backlog_under_the_cap_is_drained_normally(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "seed")
    block = KafkaConsumerBlockV1()
    run(block)
    for i in range(999):
        broker.produce(TOPIC, f"m{i}")

    result = run(block)

    assert result["value"] == "m998"
    assert len(broker.consumers[0].assign_calls) == 1  # no reposition needed


def test_key_filter_ignores_other_keys(broker: FakeBroker) -> None:
    # given
    broker.produce(TOPIC, "seed", key="cam-1")
    block = KafkaConsumerBlockV1()
    run(block, key_filter="cam-1")
    broker.produce(TOPIC, "for-cam-1", key="cam-1")
    broker.produce(TOPIC, "for-cam-2", key="cam-2")

    # when
    result = run(block, key_filter="cam-1")

    # then
    assert result["value"] == "for-cam-1"
    assert result["key"] == "cam-1"


def test_key_filter_first_run_finds_newest_record_for_that_key(
    broker: FakeBroker,
) -> None:
    # given the topic's newest record belongs to another camera
    broker.produce(TOPIC, state("old"), key="cam-1")
    broker.produce(TOPIC, state("RUNNING"), key="cam-1")
    for _ in range(5):
        broker.produce(TOPIC, state("PAUSED"), key="cam-2")

    # when
    result = run(KafkaConsumerBlockV1(), key_filter="cam-1")

    # then
    assert result["payload"] == {"state": "RUNNING"}
    assert result["key"] == "cam-1"
    assert result["offset"] == 1
    assert result["is_new"] is True


def test_key_filter_first_run_scan_window_is_bounded(broker: FakeBroker) -> None:
    broker.produce(TOPIC, state("too-old"), key="cam-1")
    for _ in range(1000):
        broker.produce(TOPIC, state("PAUSED"), key="cam-2")

    result = run(KafkaConsumerBlockV1(), key_filter="cam-1")

    assert result["value"] is None and result["is_new"] is False
    assert result["error_status"] is False
    assert broker.consumers[0].assign_calls[0] == [(0, 1)]


def test_first_run_without_filter_reads_only_the_newest_record(
    broker: FakeBroker,
) -> None:
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)

    run(KafkaConsumerBlockV1())

    assert broker.consumers[0].assign_calls[0] == [(0, 2)]


def test_first_run_consumes_every_partition_seed_record(broker: FakeBroker) -> None:
    # given three partitions each holding a record
    broker.create_topic(TOPIC, partitions=3)
    for partition, value in enumerate(("P0", "P1", "P2")):
        broker.produce(TOPIC, value, partition=partition)
    block = KafkaConsumerBlockV1()

    # when
    first = run(block)
    second = run(block)

    # then: all three seeds were consumed in the first run, so the second run
    # cannot surface an older seed as "new"
    assert first["is_new"] is True
    assert second["is_new"] is False
    assert broker.consumers[0].positions == {0: 1, 1: 1, 2: 1}


def test_key_filter_skips_records_without_key(broker: FakeBroker) -> None:
    # given
    broker.produce(TOPIC, "seed", key="cam-1")
    block = KafkaConsumerBlockV1()
    run(block, key_filter="cam-1")
    broker.produce(TOPIC, "anonymous")

    # when
    result = run(block, key_filter="cam-1")

    # then
    assert result["value"] == "seed"
    assert result["is_new"] is False


def test_non_json_value_gives_empty_payload(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "RUNNING")

    result = run(KafkaConsumerBlockV1())

    assert result["value"] == "RUNNING"
    assert result["payload"] == {}
    assert result["is_new"] is True


def test_json_array_value_gives_empty_payload(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "[1, 2, 3]")

    result = run(KafkaConsumerBlockV1())

    assert result["value"] == "[1, 2, 3]"
    assert result["payload"] == {}


def test_invalid_utf8_is_replaced_not_raised(broker: FakeBroker) -> None:
    broker.produce(TOPIC, b"\xff\xfe", key=b"\xff")

    result = run(KafkaConsumerBlockV1())

    assert result["error_status"] is False
    assert "�" in result["value"] and "�" in result["key"]


def test_message_level_error_is_reported_and_last_record_kept(
    broker: FakeBroker,
) -> None:
    # given
    broker.produce(TOPIC, "good")
    block = KafkaConsumerBlockV1()
    run(block)
    broker.logs[TOPIC][0].append((None, RuntimeError("Broker: Not leader")))

    # when
    result = run(block)

    # then
    assert result["error_status"] is True
    assert "Not leader" in result["error_message"]
    assert result["value"] == "good"
    assert result["is_new"] is False


def test_payload_output_is_a_copy(broker: FakeBroker) -> None:
    broker.produce(TOPIC, state("A"))
    block = KafkaConsumerBlockV1()

    first = run(block)
    first["payload"]["mutated"] = True
    second = run(block)

    assert second["payload"] == {"state": "A"}


# --------------------------------------------------------------------------------------
# Pointer mode
# --------------------------------------------------------------------------------------


def test_pointer_reads_exact_record_even_when_newer_exist(broker: FakeBroker) -> None:
    # given
    for value in ("A", "B", "C", "D", "E"):
        broker.produce(TOPIC, state(value))
    block = KafkaConsumerBlockV1()

    # when
    result = run(block, offset=2)

    # then
    assert result["value"] == state("C")
    assert result["offset"] == 2 and result["partition"] == 0
    assert result["is_new"] is True
    assert result["error_status"] is False


def test_same_pointer_is_served_from_cache_without_polling(
    broker: FakeBroker,
) -> None:
    # given
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()
    first = run(block, offset=1)
    consumer = broker.consumers[0]
    polls_before = consumer.poll_calls
    assigns_before = len(consumer.assign_calls)

    # when
    second = run(block, offset=1)

    # then
    assert first["is_new"] is True
    assert second["is_new"] is False
    assert second["value"] == "B"
    assert consumer.poll_calls == polls_before
    assert len(consumer.assign_calls) == assigns_before


def test_cached_pointer_does_not_disturb_latest_mode(broker: FakeBroker) -> None:
    # given a block in latest mode that last returned offset 2
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()
    run(block)
    broker.produce(TOPIC, "D")
    broker.produce(TOPIC, "E")

    # when the same record is asked for by pointer, then latest mode resumes
    cached = run(block, offset=2)
    latest = run(block)

    # then nothing published in between is lost
    assert cached["value"] == "C" and cached["is_new"] is False
    assert latest["value"] == "E" and latest["is_new"] is True


def test_cached_pointer_after_reconnect_does_not_leave_a_pending_wait(
    broker: FakeBroker,
) -> None:
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()
    run(block, offset=1)
    block.close()

    run(block, offset=1, connect_timeout=7.0)
    back = run(block, poll_timeout=0.01, connect_timeout=7.0)
    again = run(block, poll_timeout=0.01, connect_timeout=7.0)

    assert back["value"] == "C" and back["is_new"] is True
    assert again["is_new"] is False
    # once positioned, polls use poll_timeout, not a connect_timeout slice
    assert broker.consumers[-1].poll_timeouts[-1] == 0.01


def test_tombstone_record_gives_none_value(broker: FakeBroker) -> None:
    broker.produce(TOPIC, state("A"), key="cam-1")
    block = KafkaConsumerBlockV1()
    run(block)
    broker.produce(TOPIC, None, key="cam-1")

    result = run(block)

    assert result["value"] is None
    assert result["payload"] == {}
    assert result["key"] == "cam-1"
    assert result["is_new"] is True
    assert result["error_status"] is False


def test_new_pointer_reads_new_record(broker: FakeBroker) -> None:
    for value in ("A", "B", "C", "D", "E"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()
    run(block, offset=2)

    result = run(block, offset=4)

    assert result["value"] == "E"
    assert result["offset"] == 4
    assert result["is_new"] is True


def test_pointer_beyond_high_watermark_is_error_and_keeps_previous(
    broker: FakeBroker,
) -> None:
    # given
    for value in ("A", "B", "C", "D", "E"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()
    run(block, offset=1)

    # when
    result = run(block, offset=99)

    # then
    assert result["error_status"] is True
    assert "out of range" in result["error_message"]
    assert "offset 4" in result["error_message"]
    assert result["value"] == "B"
    assert result["offset"] == 1
    assert result["is_new"] is False


def test_pointer_below_low_watermark_is_error(broker: FakeBroker) -> None:
    for value in ("A", "B", "C", "D", "E"):
        broker.produce(TOPIC, value)
    broker.set_low_watermark(TOPIC, 0, 3)
    block = KafkaConsumerBlockV1()

    result = run(block, offset=1)

    assert result["error_status"] is True
    assert "no longer retained" in result["error_message"]
    assert result["value"] is None


def test_pointer_removed_by_compaction_is_error_not_substituted(
    broker: FakeBroker,
) -> None:
    # given
    for value in ("A", "B", "C", "D", "E"):
        broker.produce(TOPIC, value)
    broker.compact_away(TOPIC, 0, 3)
    block = KafkaConsumerBlockV1()
    run(block, offset=1)

    # when
    result = run(block, offset=3)

    # then
    assert result["error_status"] is True
    assert "compaction" in result["error_message"]
    assert "offset 4" in result["error_message"]
    assert result["value"] == "B"


def test_pointer_to_unknown_partition_is_error(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "A")

    result = run(KafkaConsumerBlockV1(), offset=0, partition=7)

    assert result["error_status"] is True
    assert "Partition 7 does not exist" in result["error_message"]


def test_pointer_ignores_key_filter(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "for-cam-1", key="cam-1")
    broker.produce(TOPIC, "for-cam-2", key="cam-2")

    result = run(KafkaConsumerBlockV1(), offset=1, key_filter="cam-1")

    assert result["value"] == "for-cam-2"
    assert result["key"] == "cam-2"


def test_pointer_on_multi_partition_topic_positions_other_partitions_at_end(
    broker: FakeBroker,
) -> None:
    # given
    broker.create_topic(TOPIC, partitions=3)
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value, partition=1)
    broker.produce(TOPIC, "Z", partition=2)
    block = KafkaConsumerBlockV1()

    # when
    result = run(block, offset=1, partition=1)

    # then
    assert result["value"] == "B"
    assert result["partition"] == 1
    assert sorted(broker.consumers[0].assign_calls[-1]) == [(0, 0), (1, 1), (2, 1)]


def test_leaving_pointer_mode_returns_the_newest_existing_record(
    broker: FakeBroker,
) -> None:
    # given
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()
    run(block, offset=0)
    broker.produce(TOPIC, "published-during-pointer-mode")

    # when
    back_to_latest = run(block)
    broker.produce(TOPIC, "published-after")
    after = run(block)

    # then: exactly like a first run, the newest record on the topic
    assert back_to_latest["value"] == "published-during-pointer-mode"
    assert back_to_latest["is_new"] is True
    assert back_to_latest["error_status"] is False
    assert broker.consumers[0].assign_calls[-1] == [(0, 3)]
    assert after["value"] == "published-after"
    assert after["is_new"] is True


def test_leaving_pointer_mode_when_the_pointer_was_the_newest_record(
    broker: FakeBroker,
) -> None:
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()
    run(block, offset=2)

    result = run(block)

    assert result["value"] == "C"
    assert result["is_new"] is False


# --------------------------------------------------------------------------------------
# Providers and connection config
# --------------------------------------------------------------------------------------


def consumer_config(broker: FakeBroker) -> Dict[str, Any]:
    return broker.consumers[0].config


def test_self_hosted_without_credentials_uses_plaintext(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "A")

    run(KafkaConsumerBlockV1())

    config = consumer_config(broker)
    assert config["bootstrap.servers"] == BOOTSTRAP
    assert config["security.protocol"] == "PLAINTEXT"
    assert not any(key.startswith("sasl.") for key in config)
    assert config["enable.auto.commit"] is False
    assert config["group.id"].startswith("roboflow-inference-")


def test_each_instance_gets_its_own_group_id(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "A")

    run(KafkaConsumerBlockV1())
    run(KafkaConsumerBlockV1())

    ids = {consumer.config["group.id"] for consumer in broker.consumers}
    assert len(ids) == 2


def test_self_hosted_with_credentials_uses_scram_over_tls(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "A")

    run(KafkaConsumerBlockV1(), username="inference", password="s3cret")

    config = consumer_config(broker)
    assert config["security.protocol"] == "SASL_SSL"
    assert config["sasl.mechanisms"] == "SCRAM-SHA-512"
    assert config["sasl.username"] == "inference"
    assert config["sasl.password"] == "s3cret"


@pytest.mark.parametrize(
    "credentials",
    [{"username": "inference"}, {"password": "s3cret"}],
)
def test_one_sided_credentials_fail_before_connecting(
    broker: FakeBroker, credentials: Dict[str, str]
) -> None:
    broker.produce(TOPIC, "A")

    result = run(KafkaConsumerBlockV1(), **credentials)

    assert result["error_status"] is True
    assert "both username and password" in result["error_message"]
    assert broker.consumers == []


def test_ssl_ca_location_is_passed_through(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "A")

    run(KafkaConsumerBlockV1(), username="u", password="p", ssl_ca_location="/ca.pem")

    assert consumer_config(broker)["ssl.ca.location"] == "/ca.pem"


class FakeTokenProvider:
    calls: List[str] = []

    @staticmethod
    def generate_auth_token(region: str):
        FakeTokenProvider.calls.append(region)
        return "signed-token", 1_700_000_000_000


class FailingTokenProvider:
    @staticmethod
    def generate_auth_token(region: str):
        class NoCredentialsError(Exception):
            pass

        raise NoCredentialsError("Unable to locate credentials")


@pytest.fixture
def msk_signer():
    FakeTokenProvider.calls = []
    with patch.object(kafka_common, "MSKAuthTokenProvider", FakeTokenProvider):
        yield FakeTokenProvider


MSK_BOOTSTRAP = (
    "b-1.cluster.abc123.c2.kafka.eu-west-1.amazonaws.com:9098,"
    "b-2.cluster.abc123.c2.kafka.eu-west-1.amazonaws.com:9098"
)


def test_aws_msk_derives_region_and_uses_oauthbearer(
    broker: FakeBroker, msk_signer
) -> None:
    broker.produce(TOPIC, "A")

    result = run(
        KafkaConsumerBlockV1(), bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK"
    )

    config = consumer_config(broker)
    assert result["error_status"] is False
    assert config["security.protocol"] == "SASL_SSL"
    assert config["sasl.mechanisms"] == "OAUTHBEARER"
    assert callable(config["oauth_cb"])
    assert "sasl.username" not in config and "sasl.password" not in config
    assert msk_signer.calls == ["eu-west-1"]


def test_aws_msk_explicit_region_wins_over_hostname(
    broker: FakeBroker, msk_signer
) -> None:
    broker.produce(TOPIC, "A")

    run(
        KafkaConsumerBlockV1(),
        bootstrap_servers="kafka.internal:9098",
        provider="AWS MSK",
        aws_region="us-east-1",
    )

    assert msk_signer.calls == ["us-east-1"]


def test_aws_msk_without_derivable_region_fails_before_connecting(
    broker: FakeBroker, msk_signer
) -> None:
    broker.produce(TOPIC, "A")

    result = run(
        KafkaConsumerBlockV1(),
        bootstrap_servers="kafka.internal:9098",
        provider="AWS MSK",
    )

    assert result["error_status"] is True
    assert "aws_region" in result["error_message"]
    assert broker.consumers == []
    assert msk_signer.calls == []


def test_aws_msk_ignores_username_and_password(broker: FakeBroker, msk_signer) -> None:
    broker.produce(TOPIC, "A")

    result = run(
        KafkaConsumerBlockV1(),
        bootstrap_servers=MSK_BOOTSTRAP,
        provider="AWS MSK",
        username="ignored",
        password="ignored",
    )

    assert result["error_status"] is False
    config = consumer_config(broker)
    assert "sasl.username" not in config and "sasl.password" not in config
    assert config["sasl.mechanisms"] == "OAUTHBEARER"


def test_msk_token_callback_converts_expiry_to_seconds(msk_signer) -> None:
    failures: List[BaseException] = []
    callback = msk_token_callback("eu-west-1", failures)

    token, expiry = callback("ignored-config")

    assert token == "signed-token"
    assert expiry == 1_700_000_000.0
    assert failures == []


def test_msk_token_callback_success_clears_earlier_failure(msk_signer) -> None:
    failures: List[BaseException] = [RuntimeError("earlier blip")]
    callback = msk_token_callback("eu-west-1", failures)

    callback("")

    assert failures == []


def test_transient_token_refresh_failure_is_reported_once_then_recovers(
    broker: FakeBroker, msk_signer
) -> None:
    # given a healthy MSK consumer
    broker.produce(TOPIC, state("A"))
    block = KafkaConsumerBlockV1()
    run(block, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    # librdkafka invoked the refresh callback in the background and it failed once
    block._auth_failures.append(RuntimeError("STS blip"))
    broker.produce(TOPIC, state("B"))

    # when
    failed = run(block, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    recovered = run(block, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")

    # then
    assert failed["error_status"] is True
    assert "AWS MSK IAM authentication failed" in failed["error_message"]
    assert recovered["error_status"] is False
    assert recovered["payload"] == {"state": "B"}
    assert len(broker.consumers) == 1


def test_pop_auth_failure_message_tolerates_a_concurrent_clear() -> None:
    # the refresh callback may clear the list on librdkafka's thread at any moment
    assert kafka_common.pop_auth_failure_message([]) is None
    failures: List[BaseException] = [RuntimeError("blip")]
    assert "blip" in kafka_common.pop_auth_failure_message(failures)
    assert failures == []


def test_msk_token_callback_records_and_reraises_failures() -> None:
    failures: List[BaseException] = []
    with patch.object(kafka_common, "MSKAuthTokenProvider", FailingTokenProvider):
        callback = msk_token_callback("eu-west-1", failures)

        with pytest.raises(Exception):
            callback("")

    assert len(failures) == 1


def test_aws_msk_signer_failure_reports_missing_credentials_and_retries(
    broker: FakeBroker,
) -> None:
    # given
    broker.produce(TOPIC, "A")
    block = KafkaConsumerBlockV1()

    # when
    with patch.object(kafka_common, "MSKAuthTokenProvider", FailingTokenProvider):
        failed = run(block, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")
    with patch.object(kafka_common, "MSKAuthTokenProvider", FakeTokenProvider):
        recovered = run(block, bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK")

    # then
    assert failed["error_status"] is True
    assert "AWS MSK IAM authentication failed" in failed["error_message"]
    assert "no AWS credentials were found" in failed["error_message"]
    assert failed["value"] is None
    assert recovered["error_status"] is False
    assert recovered["value"] == "A"
    assert len(broker.consumers) == 1


def test_aws_msk_requires_signer_package(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "A")
    with patch.object(kafka_common, "MSKAuthTokenProvider", None):
        result = run(
            KafkaConsumerBlockV1(), bootstrap_servers=MSK_BOOTSTRAP, provider="AWS MSK"
        )

    assert result["error_status"] is True
    assert "aws-msk-iam-sasl-signer-python" in result["error_message"]


@pytest.mark.parametrize(
    "servers, expected",
    [
        (MSK_BOOTSTRAP, "eu-west-1"),
        (
            "boot-abc.c1.kafka-serverless.ap-southeast-1.amazonaws.com:9098",
            "ap-southeast-1",
        ),
        ("b-1.x.kafka.cn-north-1.amazonaws.com.cn:9098", "cn-north-1"),
        ("B-1.X.KAFKA.US-EAST-2.AMAZONAWS.COM:9098", "us-east-2"),
        ("localhost:9092", None),
        ("kafka.example.com:9092,other.example.com:9092", None),
        ("", None),
    ],
)
def test_derive_msk_region(servers: str, expected: Optional[str]) -> None:
    assert derive_msk_region(servers) == expected


# --------------------------------------------------------------------------------------
# Lifecycle and failure handling
# --------------------------------------------------------------------------------------


def test_first_run_lookups_share_one_connect_timeout_budget(
    broker: FakeBroker,
) -> None:
    # given a 6-partition topic and a broker that answers each lookup slowly
    broker.create_topic(TOPIC, partitions=6)
    for partition in range(6):
        broker.produce(TOPIC, "x", partition=partition)
    broker.watermark_delay = 0.05

    # when
    started = time.monotonic()
    result = run(KafkaConsumerBlockV1(), connect_timeout=0.12)
    elapsed = time.monotonic() - started

    # then: the run failed inside roughly one connect_timeout, not 6 x connect_timeout
    assert result["error_status"] is True
    assert "timed out" in result["error_message"]
    assert elapsed < 0.12 * 3
    timeouts = broker.consumers[0].watermark_timeouts
    assert all(t <= 0.12 for t in timeouts)
    assert timeouts == sorted(timeouts, reverse=True)
    assert broker.consumers[0].closed is True


def test_pointer_on_first_run_skips_the_newest_record_lookups(
    broker: FakeBroker,
) -> None:
    broker.create_topic(TOPIC, partitions=4)
    for partition in range(4):
        broker.produce(TOPIC, f"p{partition}", partition=partition)

    result = run(KafkaConsumerBlockV1(), offset=0, partition=2)

    # one lookup for the pointed partition + one per other partition, not 2N-1
    assert result["value"] == "p2"
    assert len(broker.consumers[0].watermark_timeouts) == 4
    assert len(broker.consumers[0].assign_calls) == 1


def test_pointer_first_then_latest_positions_the_consumer(broker: FakeBroker) -> None:
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()

    run(block, offset=1)
    block.close()
    cached = run(block, offset=1)  # reconnects, cache hit, consumer not positioned
    resumed = run(block)  # positions at the newest existing record
    broker.produce(TOPIC, "D")
    latest = run(block)

    assert cached["value"] == "B"
    assert resumed["value"] == "C" and resumed["is_new"] is True
    assert latest["value"] == "D" and latest["is_new"] is True


def test_unreachable_broker_reports_error_and_retries_next_run(
    broker: FakeBroker,
) -> None:
    # given
    broker.produce(TOPIC, "A")
    broker.fail_connect = RuntimeError("Local: Broker transport failure")
    block = KafkaConsumerBlockV1()

    # when
    failed = run(block, connect_timeout=0.5)
    broker.fail_connect = None
    recovered = run(block, connect_timeout=0.5)

    # then
    assert failed["error_status"] is True
    assert "Kafka broker not reachable" in failed["error_message"]
    assert "transport failure" in failed["error_message"]
    assert failed["value"] is None and failed["payload"] == {}
    assert broker.consumers[0].closed is True
    assert recovered["error_status"] is False
    assert recovered["value"] == "A"
    assert len(broker.consumers) == 2


def test_missing_topic_is_reported(broker: FakeBroker) -> None:
    broker.create_topic("another-topic")

    result = run(KafkaConsumerBlockV1())

    assert result["error_status"] is True
    assert TOPIC in result["error_message"]
    assert broker.consumers[0].closed is True


@pytest.mark.parametrize(
    "change",
    [
        {"bootstrap_servers": "elsewhere:9092"},
        {"topic": "other"},
        {"provider": "AWS MSK", "aws_region": "us-east-1"},
        {"username": "u", "password": "p"},
    ],
)
def test_changed_connection_parameters_are_rejected(
    broker: FakeBroker, change: Dict[str, Any]
) -> None:
    # given
    broker.produce(TOPIC, "A")
    broker.create_topic("other")
    block = KafkaConsumerBlockV1()
    run(block)

    # when
    with patch.object(kafka_common, "MSKAuthTokenProvider", FakeTokenProvider):
        result = run(block, **change)

    # then
    assert result["error_status"] is True
    assert "changed between runs" in result["error_message"]
    assert result["value"] == "A"
    assert len(broker.consumers) == 1
    assert broker.consumers[0].closed is False


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"poll_timeout": "soon"}, "poll_timeout"),
        ({"poll_timeout": float("nan")}, "poll_timeout"),
        ({"connect_timeout": 0}, "connect_timeout"),
        ({"connect_timeout": -1}, "connect_timeout"),
        ({"offset": -1}, "offset"),
        ({"offset": 1.5}, "offset"),
        ({"offset": "abc"}, "offset"),
        ({"offset": 1, "partition": -1}, "partition"),
        ({"offset": 1, "partition": True}, "partition"),
        ({"bootstrap_servers": "  "}, "bootstrap_servers"),
        ({"topic": ""}, "topic"),
    ],
)
def test_selector_resolved_values_are_revalidated_before_connecting(
    broker: FakeBroker, overrides: Dict[str, Any], fragment: str
) -> None:
    broker.produce(TOPIC, "A")

    result = run(KafkaConsumerBlockV1(), **overrides)

    assert result["error_status"] is True
    assert fragment in result["error_message"]
    assert broker.consumers == []


def test_numeric_strings_from_selectors_are_coerced(broker: FakeBroker) -> None:
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)

    result = run(
        KafkaConsumerBlockV1(),
        offset="1",
        partition=0.0,
        poll_timeout="0.01",
        connect_timeout="2",
    )

    assert result["error_status"] is False
    assert result["value"] == "B"


@pytest.mark.parametrize("flag", ["GCP_SERVERLESS", "LAMBDA"])
def test_hosted_platform_is_refused(broker: FakeBroker, flag: str) -> None:
    broker.produce(TOPIC, "A")
    with patch.object(v1, flag, True):
        result = run(KafkaConsumerBlockV1())

    assert result["error_status"] is True
    assert "hosted platform" in result["error_message"]
    assert broker.consumers == []


def test_missing_confluent_kafka_is_reported() -> None:
    with patch.object(v1, "confluent_kafka", None):
        result = run(KafkaConsumerBlockV1())

    assert result["error_status"] is True
    assert "confluent-kafka" in result["error_message"]


def test_close_is_idempotent_and_releases_consumer(broker: FakeBroker) -> None:
    # given
    broker.produce(TOPIC, "A")
    block = KafkaConsumerBlockV1()
    run(block)
    consumer = broker.consumers[0]

    # when
    block.close()
    block.close()

    # then
    assert consumer.closed is True
    assert block._consumer is None


def test_close_before_any_run_is_a_noop(broker: FakeBroker) -> None:
    block = KafkaConsumerBlockV1()

    block.close()

    assert broker.consumers == []


def test_run_after_close_reconnects(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "A")
    block = KafkaConsumerBlockV1()
    run(block)
    block.close()
    broker.produce(TOPIC, "B")

    result = run(block)

    assert result["value"] == "B"
    assert len(broker.consumers) == 2


def test_two_instances_both_see_the_same_record(broker: FakeBroker) -> None:
    # given
    broker.produce(TOPIC, "seed")
    first, second = KafkaConsumerBlockV1(), KafkaConsumerBlockV1()
    run(first)
    run(second)
    broker.produce(TOPIC, state("RUNNING"))

    # when
    results = [run(first), run(second)]

    # then
    assert all(r["payload"] == {"state": "RUNNING"} for r in results)
    assert all(r["is_new"] is True for r in results)


# --------------------------------------------------------------------------------------
# Sequential read mode
# --------------------------------------------------------------------------------------


def run_seq(block: KafkaConsumerBlockV1, **overrides) -> Dict[str, Any]:
    return run(block, read_mode="sequential", **overrides)


def test_sequential_first_run_starts_at_the_newest_existing_record(
    broker: FakeBroker,
) -> None:
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()

    first = run_seq(block)
    second = run_seq(block)

    assert first["value"] == "C" and first["is_new"] is True
    assert second["value"] == "C" and second["is_new"] is False
    assert broker.consumers[0].assign_calls[0] == [(0, 2)]


def test_sequential_first_run_on_empty_topic(broker: FakeBroker) -> None:
    broker.create_topic(TOPIC)

    result = run_seq(KafkaConsumerBlockV1())

    assert result["value"] is None and result["is_new"] is False
    assert result["error_status"] is False


def test_sequential_delivers_every_record_in_order_one_per_run(
    broker: FakeBroker,
) -> None:
    # given
    broker.produce(TOPIC, "seed")
    block = KafkaConsumerBlockV1()
    run_seq(block)
    for value in ("D", "E", "F"):
        broker.produce(TOPIC, value)

    # when
    results = [run_seq(block) for _ in range(4)]

    # then: nothing skipped, nothing repeated, then the last one is repeated as not new
    assert [r["value"] for r in results] == ["D", "E", "F", "F"]
    assert [r["is_new"] for r in results] == [True, True, True, False]
    assert [r["offset"] for r in results] == [1, 2, 3, 3]


def test_sequential_does_not_skip_a_backlog_over_the_cap(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "seed")
    block = KafkaConsumerBlockV1()
    run_seq(block)
    for i in range(1500):
        broker.produce(TOPIC, f"m{i}")

    result = run_seq(block)

    assert result["value"] == "m0"
    assert len(broker.consumers[0].assign_calls) == 1


def test_sequential_key_filter_applies_to_every_run(broker: FakeBroker) -> None:
    # given the newest record belongs to another camera
    broker.produce(TOPIC, state("RUNNING"), key="cam-1")
    for _ in range(3):
        broker.produce(TOPIC, state("PAUSED"), key="cam-2")
    block = KafkaConsumerBlockV1()

    first = run_seq(block, key_filter="cam-1")
    broker.produce(TOPIC, state("X"), key="cam-2")
    broker.produce(TOPIC, state("STOPPED"), key="cam-1")
    second = run_seq(block, key_filter="cam-1")
    third = run_seq(block, key_filter="cam-1")

    assert first["payload"] == {"state": "RUNNING"} and first["is_new"] is True
    assert second["payload"] == {"state": "STOPPED"} and second["is_new"] is True
    assert third["payload"] == {"state": "STOPPED"} and third["is_new"] is False


def test_sequential_pointer_is_the_start_position(broker: FakeBroker) -> None:
    for value in ("A", "B", "C", "D", "E"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()

    # a literal pointer repeated on every run starts once and then continues
    results = [run_seq(block, offset=1) for _ in range(3)]
    # dropping the pointer keeps continuing
    continued = run_seq(block)
    # a different pointer re-seeks
    reseek = run_seq(block, offset=3)

    assert [r["value"] for r in results] == ["B", "C", "D"]
    assert all(r["is_new"] for r in results)
    assert continued["value"] == "E"
    assert reseek["value"] == "D" and reseek["is_new"] is True
    assert reseek["offset"] == 3


def test_sequential_pointer_out_of_range_keeps_previous(broker: FakeBroker) -> None:
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)
    block = KafkaConsumerBlockV1()
    run_seq(block)

    result = run_seq(block, offset=42)

    assert result["error_status"] is True
    assert "out of range" in result["error_message"]
    assert result["value"] == "C"


def test_sequential_fresh_instance_per_call_returns_the_newest_record(
    broker: FakeBroker,
) -> None:
    for value in ("A", "B", "C"):
        broker.produce(TOPIC, value)

    results = [run_seq(KafkaConsumerBlockV1()) for _ in range(2)]

    assert [r["value"] for r in results] == ["C", "C"]
    assert all(r["is_new"] for r in results)


def test_sequential_on_multiple_partitions_delivers_arrival_order(
    broker: FakeBroker,
) -> None:
    broker.create_topic(TOPIC, partitions=2)
    broker.produce(TOPIC, "seed0", partition=0)
    broker.produce(TOPIC, "seed1", partition=1)
    block = KafkaConsumerBlockV1()
    run_seq(block)
    broker.produce(TOPIC, "a0", partition=0)
    broker.produce(TOPIC, "b1", partition=1)
    broker.produce(TOPIC, "c0", partition=0)

    delivered = [run_seq(block)["value"] for _ in range(3)]

    # the fake serves partitions in order, like an interleaving broker; all three
    # records are delivered exactly once, each partition in its own order
    assert sorted(delivered) == ["a0", "b1", "c0"]
    assert delivered.index("a0") < delivered.index("c0")


def test_unknown_read_mode_fails_before_connecting(broker: FakeBroker) -> None:
    broker.produce(TOPIC, "A")

    result = run(KafkaConsumerBlockV1(), read_mode="from-beginning")

    assert result["error_status"] is True
    assert "read_mode" in result["error_message"]
    assert broker.consumers == []
