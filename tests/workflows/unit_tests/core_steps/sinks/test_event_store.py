import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from inference.core.workflows.core_steps.sinks.event_store.event_store import (
    EventStore,
)
from inference.core.workflows.core_steps.sinks.event_store.v1 import (
    BlockManifest,
    EventStoreSinkBlockV1,
    _path_is_within_specified_directory,
    _serialize_payload,
)


# ── EventStore class tests ──────────────────────────────────────────────────


class TestEventStoreAppendAndRetrieve:
    def test_append_returns_unique_event_id(self) -> None:
        # given
        store = EventStore()

        # when
        event_id = store.append(event_type="detection", payload={"class": "dog"})

        # then
        assert isinstance(event_id, str)
        assert len(event_id) == 36  # UUID4 format

    def test_append_multiple_returns_distinct_ids(self) -> None:
        # given
        store = EventStore()

        # when
        id_1 = store.append(event_type="detection", payload={"class": "dog"})
        id_2 = store.append(event_type="detection", payload={"class": "cat"})

        # then
        assert id_1 != id_2

    def test_appended_event_can_be_retrieved(self) -> None:
        # given
        store = EventStore()
        store.append(event_type="alert", payload={"confidence": 0.95})

        # when
        events = store.get_events()

        # then
        assert len(events) == 1
        assert events[0]["event_type"] == "alert"
        assert events[0]["payload"] == {"confidence": 0.95}
        assert "event_id" in events[0]
        assert "timestamp" in events[0]

    def test_events_include_metadata_when_provided(self) -> None:
        # given
        store = EventStore()

        # when
        store.append(
            event_type="detection",
            payload={"class": "car"},
            metadata={"source": "camera_1"},
        )
        events = store.get_events()

        # then
        assert events[0]["metadata"] == {"source": "camera_1"}

    def test_events_have_empty_metadata_when_not_provided(self) -> None:
        # given
        store = EventStore()

        # when
        store.append(event_type="detection", payload={"class": "car"})
        events = store.get_events()

        # then
        assert events[0]["metadata"] == {}


class TestEventStoreGetEvents:
    def test_get_events_respects_limit(self) -> None:
        # given
        store = EventStore()
        for i in range(10):
            store.append(event_type="detection", payload={"index": i})

        # when
        events = store.get_events(limit=3)

        # then
        assert len(events) == 3
        assert events[0]["payload"]["index"] == 0
        assert events[2]["payload"]["index"] == 2

    def test_get_events_returns_all_when_limit_exceeds_count(self) -> None:
        # given
        store = EventStore()
        store.append(event_type="detection", payload={"class": "dog"})

        # when
        events = store.get_events(limit=100)

        # then
        assert len(events) == 1

    def test_get_events_filters_by_event_type(self) -> None:
        # given
        store = EventStore()
        store.append(event_type="detection", payload={"class": "dog"})
        store.append(event_type="alert", payload={"level": "high"})
        store.append(event_type="detection", payload={"class": "cat"})
        store.append(event_type="metric", payload={"fps": 30})

        # when
        detections = store.get_events(event_type="detection")

        # then
        assert len(detections) == 2
        assert all(e["event_type"] == "detection" for e in detections)

    def test_get_events_filter_and_limit_combined(self) -> None:
        # given
        store = EventStore()
        for i in range(5):
            store.append(event_type="detection", payload={"index": i})
        store.append(event_type="alert", payload={"level": "low"})

        # when
        events = store.get_events(limit=2, event_type="detection")

        # then
        assert len(events) == 2
        assert events[0]["payload"]["index"] == 0
        assert events[1]["payload"]["index"] == 1

    def test_get_events_returns_empty_list_when_no_match(self) -> None:
        # given
        store = EventStore()
        store.append(event_type="detection", payload={"class": "dog"})

        # when
        events = store.get_events(event_type="nonexistent")

        # then
        assert events == []

    def test_get_events_returns_empty_list_for_empty_store(self) -> None:
        # given
        store = EventStore()

        # when
        events = store.get_events()

        # then
        assert events == []


class TestEventStoreCount:
    def test_count_starts_at_zero(self) -> None:
        # given
        store = EventStore()

        # then
        assert store.count == 0

    def test_count_increments_with_appends(self) -> None:
        # given
        store = EventStore()

        # when
        store.append(event_type="a", payload={})
        store.append(event_type="b", payload={})
        store.append(event_type="c", payload={})

        # then
        assert store.count == 3


class TestEventStoreFlushToDisk:
    def test_flush_to_disk_writes_valid_jsonl(self, tmp_path) -> None:
        # given
        store = EventStore()
        store.append(event_type="detection", payload={"class": "dog"})
        store.append(event_type="alert", payload={"level": "critical"})
        file_path = str(tmp_path / "events.jsonl")

        # when
        store.flush_to_disk(file_path)

        # then
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2
        event_0 = json.loads(lines[0])
        event_1 = json.loads(lines[1])
        assert event_0["event_type"] == "detection"
        assert event_0["payload"] == {"class": "dog"}
        assert event_1["event_type"] == "alert"
        assert event_1["payload"] == {"level": "critical"}

    def test_flush_to_disk_creates_parent_directories(self, tmp_path) -> None:
        # given
        store = EventStore()
        store.append(event_type="test", payload={"key": "value"})
        file_path = str(tmp_path / "nested" / "deep" / "events.jsonl")

        # when
        store.flush_to_disk(file_path)

        # then
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1

    def test_flush_to_disk_writes_empty_file_when_no_events(self, tmp_path) -> None:
        # given
        store = EventStore()
        file_path = str(tmp_path / "empty.jsonl")

        # when
        store.flush_to_disk(file_path)

        # then
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            content = f.read()
        assert content == ""

    def test_flush_to_disk_jsonl_fields_are_correct(self, tmp_path) -> None:
        # given
        store = EventStore()
        event_id = store.append(
            event_type="detection",
            payload={"class": "car"},
            metadata={"camera": "front"},
        )
        file_path = str(tmp_path / "events.jsonl")

        # when
        store.flush_to_disk(file_path)

        # then
        with open(file_path, "r") as f:
            event = json.loads(f.readline())
        assert event["event_id"] == event_id
        assert event["event_type"] == "detection"
        assert event["payload"] == {"class": "car"}
        assert event["metadata"] == {"camera": "front"}
        assert "timestamp" in event


class TestEventStoreClear:
    def test_clear_removes_all_events(self) -> None:
        # given
        store = EventStore()
        store.append(event_type="a", payload={})
        store.append(event_type="b", payload={})
        assert store.count == 2

        # when
        store.clear()

        # then
        assert store.count == 0
        assert store.get_events() == []


class TestEventStoreThreadSafety:
    def test_concurrent_appends_are_thread_safe(self) -> None:
        # given
        store = EventStore()
        num_threads = 10
        events_per_thread = 50

        def append_events(thread_id: int) -> None:
            for i in range(events_per_thread):
                store.append(
                    event_type=f"thread_{thread_id}",
                    payload={"thread": thread_id, "index": i},
                )

        # when
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(append_events, tid) for tid in range(num_threads)
            ]
            for future in futures:
                future.result()

        # then
        assert store.count == num_threads * events_per_thread
        all_events = store.get_events(limit=num_threads * events_per_thread)
        assert len(all_events) == num_threads * events_per_thread

    def test_concurrent_appends_and_reads_are_safe(self) -> None:
        # given
        store = EventStore()
        errors = []

        def writer() -> None:
            for i in range(100):
                store.append(event_type="write", payload={"i": i})

        def reader() -> None:
            try:
                for _ in range(100):
                    _ = store.get_events(limit=50)
                    _ = store.count
            except Exception as e:
                errors.append(e)

        # when
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # then
        assert len(errors) == 0
        assert store.count == 300  # 3 writers * 100 each


# ── BlockManifest tests ─────────────────────────────────────────────────────


def test_manifest_parsing_when_input_is_valid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/event_store_sink@v1",
        "name": "event_sink",
        "payload": "$steps.model.predictions",
        "event_type": "detection",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.type == "roboflow_core/event_store_sink@v1"
    assert result.event_type == "detection"
    assert result.persist_to_disk is False
    assert result.disable_sink is False
    assert result.cooldown_seconds == 5


def test_manifest_parsing_with_all_fields() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/event_store_sink@v1",
        "name": "event_sink",
        "payload": "$steps.model.predictions",
        "event_type": "$inputs.event_type",
        "persist_to_disk": True,
        "target_directory": "output/events",
        "file_name_prefix": "my_events",
        "cooldown_seconds": 10,
        "disable_sink": "$inputs.disable_sink",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.persist_to_disk is True
    assert result.target_directory == "output/events"
    assert result.file_name_prefix == "my_events"
    assert result.cooldown_seconds == 10


def test_manifest_outputs_are_correct() -> None:
    # when
    outputs = BlockManifest.describe_outputs()

    # then
    output_names = {o.name for o in outputs}
    assert output_names == {"event_id", "error_status", "throttling_status", "message"}


# ── EventStoreSinkBlockV1 tests ─────────────────────────────────────────────


class TestEventStoreSinkBlockV1BasicRun:
    def test_basic_run_stores_event_and_returns_event_id(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        result = block.run(
            payload={"class": "dog", "confidence": 0.95},
            event_type="detection",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is False
        assert result["throttling_status"] is False
        assert result["message"] == "Event stored successfully"
        assert len(result["event_id"]) == 36  # UUID4

    def test_run_returns_different_event_ids_on_subsequent_calls(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        r1 = block.run(
            payload={"i": 1},
            event_type="test",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )
        r2 = block.run(
            payload={"i": 2},
            event_type="test",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert r1["event_id"] != r2["event_id"]


class TestEventStoreSinkBlockV1Cooldown:
    def test_cooldown_throttles_second_call(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        results = []
        for _ in range(2):
            result = block.run(
                payload={"key": "val"},
                event_type="test",
                persist_to_disk=False,
                target_directory="",
                file_name_prefix="events",
                cooldown_seconds=100,
                disable_sink=False,
            )
            results.append(result)

        # then
        assert results[0]["throttling_status"] is False
        assert results[0]["event_id"] != ""
        assert results[1]["throttling_status"] is True
        assert results[1]["event_id"] == ""

    def test_cooldown_recovery_after_expiry(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        results = []
        for _ in range(2):
            result = block.run(
                payload={"key": "val"},
                event_type="test",
                persist_to_disk=False,
                target_directory="",
                file_name_prefix="events",
                cooldown_seconds=1,
                disable_sink=False,
            )
            results.append(result)
            time.sleep(1.5)

        # then
        assert results[0]["throttling_status"] is False
        assert results[1]["throttling_status"] is False

    def test_zero_cooldown_allows_consecutive_calls(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        results = []
        for _ in range(5):
            result = block.run(
                payload={"i": 1},
                event_type="test",
                persist_to_disk=False,
                target_directory="",
                file_name_prefix="events",
                cooldown_seconds=0,
                disable_sink=False,
            )
            results.append(result)

        # then
        assert all(r["throttling_status"] is False for r in results)
        assert all(r["event_id"] != "" for r in results)


class TestEventStoreSinkBlockV1DisableSink:
    def test_disable_sink_returns_early_with_no_error(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        result = block.run(
            payload={"class": "dog"},
            event_type="detection",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=True,
        )

        # then
        assert result == {
            "event_id": "",
            "error_status": False,
            "throttling_status": False,
            "message": "Sink was disabled by parameter `disable_sink`",
        }


class TestEventStoreSinkBlockV1PersistToDisk:
    def test_persist_to_disk_writes_file(self, tmp_path) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=True,
            allowed_write_directory=None,
        )
        target_dir = str(tmp_path / "output")

        # when
        result = block.run(
            payload={"class": "dog", "confidence": 0.9},
            event_type="detection",
            persist_to_disk=True,
            target_directory=target_dir,
            file_name_prefix="test_events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is False
        assert result["event_id"] != ""
        # Verify file was written
        files = os.listdir(target_dir)
        assert len(files) == 1
        assert files[0].startswith("test_events_")
        assert files[0].endswith(".jsonl")

    def test_persist_to_disk_forbidden_when_fs_access_disabled(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        result = block.run(
            payload={"class": "dog"},
            event_type="detection",
            persist_to_disk=True,
            target_directory="/some/directory",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is True
        assert result["event_id"] != ""  # event was still stored in memory
        assert "forbidden" in result["message"].lower()

    def test_persist_to_disk_outside_allowed_directory(self, tmp_path) -> None:
        # given
        allowed_dir = str(tmp_path / "allowed")
        os.makedirs(allowed_dir)
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=True,
            allowed_write_directory=allowed_dir,
        )

        # when
        result = block.run(
            payload={"class": "dog"},
            event_type="detection",
            persist_to_disk=True,
            target_directory="/tmp/not_allowed",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is True
        assert "not a sub-directory" in result["message"]


class TestEventStoreSinkBlockV1PayloadTypes:
    def test_dict_payload(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        result = block.run(
            payload={"class": "dog", "confidence": 0.95, "bbox": [10, 20, 100, 200]},
            event_type="detection",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is False
        assert result["event_id"] != ""

    def test_list_payload(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        result = block.run(
            payload=[1, 2, 3, "four"],
            event_type="list_event",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is False

    def test_string_payload(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        result = block.run(
            payload="simple string message",
            event_type="text_event",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is False

    def test_int_payload(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        result = block.run(
            payload=42,
            event_type="numeric_event",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is False

    def test_float_payload(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        result = block.run(
            payload=3.14159,
            event_type="numeric_event",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is False

    def test_nested_dict_payload(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when
        result = block.run(
            payload={
                "detections": [
                    {"class": "dog", "confidence": 0.9},
                    {"class": "cat", "confidence": 0.8},
                ],
                "frame": 42,
            },
            event_type="complex_event",
            persist_to_disk=False,
            target_directory="",
            file_name_prefix="events",
            cooldown_seconds=0,
            disable_sink=False,
        )

        # then
        assert result["error_status"] is False


class TestEventStoreSinkBlockV1ErrorHandling:
    def test_error_in_event_store_append_returns_error_status(self) -> None:
        # given
        block = EventStoreSinkBlockV1(
            allow_access_to_file_system=False,
            allowed_write_directory=None,
        )

        # when - patch the internal event store to raise
        with patch.object(
            block._event_store, "append", side_effect=RuntimeError("Store failure")
        ):
            result = block.run(
                payload={"class": "dog"},
                event_type="detection",
                persist_to_disk=False,
                target_directory="",
                file_name_prefix="events",
                cooldown_seconds=0,
                disable_sink=False,
            )

        # then
        assert result["error_status"] is True
        assert result["event_id"] == ""
        assert "Store failure" in result["message"]


# ── Helper function tests ───────────────────────────────────────────────────


class TestSerializePayload:
    def test_dict_payload_returned_as_is(self) -> None:
        # given
        payload = {"class": "dog", "confidence": 0.9}

        # when
        result = _serialize_payload(payload)

        # then
        assert result == payload

    def test_string_payload_wrapped_in_value_key(self) -> None:
        # given
        payload = "hello world"

        # when
        result = _serialize_payload(payload)

        # then
        assert result == {"value": "hello world"}

    def test_int_payload_wrapped_in_value_key(self) -> None:
        # given
        payload = 42

        # when
        result = _serialize_payload(payload)

        # then
        assert result == {"value": 42}

    def test_list_payload_wrapped_in_value_key(self) -> None:
        # given
        payload = [1, 2, 3]

        # when
        result = _serialize_payload(payload)

        # then
        assert result == {"value": [1, 2, 3]}

    def test_non_serializable_payload_converted_to_string(self) -> None:
        # given
        payload = object()

        # when
        result = _serialize_payload(payload)

        # then
        assert "value" in result
        assert isinstance(result["value"], str)


class TestPathWithinDirectory:
    def test_path_within_directory_returns_true(self) -> None:
        assert _path_is_within_specified_directory(
            path="/data/events/output",
            specified_directory="/data/events",
        ) is True

    def test_path_outside_directory_returns_false(self) -> None:
        assert _path_is_within_specified_directory(
            path="/tmp/other",
            specified_directory="/data/events",
        ) is False

    def test_path_equal_to_directory_returns_true(self) -> None:
        assert _path_is_within_specified_directory(
            path="/data/events",
            specified_directory="/data/events",
        ) is True
