import itertools
import json
import logging
import threading
import time
from collections import Counter, defaultdict
from queue import Queue
from unittest import mock

import pytest

from inference_server import configuration
from inference_server.usage import collector as collector_module
from inference_server.usage import delivery as delivery_module
from inference_server.usage import payload_helpers
from inference_server.usage.collector import UsageCollector
from inference_server.usage.queues import RedisQueue
from tests.unit_tests.usage.conftest import SYSTEM_INFO
from tests.unit_tests.usage.test_collector import (
    POST,
    model_entry,
    queued_payloads,
    record,
    total_frames,
)
from tests.unit_tests.usage.test_queues import FakeRedis

WAIT_S = 10


class BlockingQueue(Queue):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def put(self, item, block=True, timeout=None):
        self.entered.set()
        assert self.release.wait(WAIT_S)
        super().put(item, block=False)


class StalledQueue(Queue):
    def __init__(self):
        super().__init__()
        self.stalled = True
        self.dequeues = 0

    def empty(self):
        if self.stalled:
            return False
        return super().empty()

    def get_nowait(self):
        if not self.stalled:
            return super().get_nowait()
        self.dequeues += 1
        assert self.dequeues < 100
        return []


def custom_python_entry(step_name, *, execution_duration=0.25, block_type="block_a"):
    entry = {
        "block_type": block_type,
        "step_name": step_name,
        "execution_duration": execution_duration,
    }

    return entry


def sent_rows(post_mock):
    rows = [row for call in post_mock.call_args_list for row in call.kwargs["json"]]

    return rows


def usage_threads():
    threads = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("usage-") and thread.is_alive()
    ]

    return threads


def test_recording_never_waits_for_a_slow_queue_write(collector, monkeypatch):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 3)
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    queue = BlockingQueue()
    collector._delivery.queue = queue
    collector.start()
    try:
        for index in range(4):
            record(collector, resource_id=f"first-{index}")
        assert queue.entered.wait(WAIT_S)

        def worker(worker_index):
            for index in range(250):
                record(collector, resource_id=f"worker-{worker_index}-{index}")

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=WAIT_S)
        stuck = [thread for thread in threads if thread.is_alive()]
    finally:
        queue.release.set()

    assert not stuck
    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        complete = collector.stop()

    assert complete is True
    assert sum(row["processed_frames"] for row in sent_rows(post_mock)) == 1004
    assert collector.inline_queue_writes == 0
    assert not usage_threads()


def test_pending_windows_are_bounded_by_writing_the_oldest_in_the_caller(
    collector, monkeypatch
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    monkeypatch.setattr(delivery_module, "MAX_PENDING_ROWS", 3)
    collector._delivery.queue = Queue()

    for index in range(10):
        record(collector, resource_id=f"resource-{index}", frames=2)

    assert collector.inline_queue_writes > 0
    assert collector._delivery.queue.qsize() == collector.inline_queue_writes
    collector._write_current_usage_to_queue()
    assert total_frames(queued_payloads(collector)) == 20


def test_pending_windows_below_the_bound_are_not_written_by_the_caller(
    collector, monkeypatch
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    collector._delivery.queue = Queue()

    for index in range(10):
        record(collector, resource_id=f"resource-{index}")

    assert collector.inline_queue_writes == 0
    assert collector._delivery.queue.qsize() == 0


def test_recording_does_not_resolve_the_host_name(collector, monkeypatch):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    collector._system_info = {}

    with mock.patch.object(
        UsageCollector, "system_info", return_value=dict(SYSTEM_INFO)
    ) as system_info:
        record(collector, resource_id="before")
        system_info.assert_not_called()
        collector.start()
        try:
            assert collector._system_info_attempted.wait(WAIT_S)
            with mock.patch(POST) as post_mock:
                post_mock.return_value.status_code = 200
                collector.flush()
        finally:
            collector.stop()

    system_info.assert_called_once()
    rows = sent_rows(post_mock)
    assert len(rows) == 1
    assert rows[0]["hostname"] == "host1"
    assert rows[0]["ip_address_hash"] == "ab12c"


def test_rows_recorded_before_the_host_is_known_carry_the_real_values_when_sent(
    collector,
):
    collector._system_info = {}

    with mock.patch.object(
        UsageCollector, "system_info", return_value=dict(SYSTEM_INFO)
    ):
        record(collector, resource_id="before")
        collector.record_system_info()
        record(collector, resource_id="after")
        with mock.patch(POST) as post_mock:
            post_mock.return_value.status_code = 200
            collector.flush()

    rows = sent_rows(post_mock)
    assert sorted(row["resource_id"] for row in rows) == ["after", "before"]
    assert {row["hostname"] for row in rows} == {"host1"}
    assert {row["ip_address_hash"] for row in rows} == {"ab12c"}


def test_rows_written_to_redis_carry_the_real_host_values(monkeypatch):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    monkeypatch.setattr(configuration, "REDIS_HOST", "redis.local")
    client = FakeRedis()
    usage_collector = UsageCollector(redis_client=client)
    assert isinstance(usage_collector._delivery.queue, RedisQueue)

    with mock.patch.object(
        UsageCollector, "system_info", return_value=dict(SYSTEM_INFO)
    ):
        record(usage_collector)
        usage_collector.record_system_info()
        usage_collector.flush()

    stored = json.loads(client.executed[0][0][1]["value"])
    row = next(iter(next(iter(stored.values())).values()))
    assert row["hostname"] == "host1"
    assert row["ip_address_hash"] == "ab12c"


def test_a_dequeue_that_makes_no_progress_ends_the_drain(collector):
    queue = StalledQueue()
    collector._delivery.queue = queue
    record(collector, frames=3)

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()
        post_mock.assert_not_called()
        queue.stalled = False
        collector.flush()

    assert queue.dequeues < 5
    assert sum(row["processed_frames"] for row in sent_rows(post_mock)) == 3
    assert queue.empty()


def test_stop_reports_complete_and_leaves_no_thread_in_the_normal_case(
    collector, monkeypatch
):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    record(collector, frames=2)
    collector.start()

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        complete = collector.stop()

    assert complete is True
    assert not usage_threads()
    assert sum(row["processed_frames"] for row in sent_rows(post_mock)) == 2
    assert post_mock.call_count == 1


def owned_frames(collector):
    payloads = [item.payload for item in collector._delivery.pending]
    payloads.append(collector._usage)
    frames = sum(
        row["processed_frames"]
        for payload in payloads
        for rows in payload.values()
        for row in rows.values()
    )

    return frames


def test_stop_does_no_io_and_loses_nothing_when_the_collector_thread_is_stuck(
    collector, monkeypatch
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    entered = threading.Event()
    release = threading.Event()
    original = collector._delivery.drain_pending

    def stuck(*args, **kwargs):
        if threading.current_thread().name == delivery_module.COLLECTOR_THREAD_NAME:
            entered.set()
            assert release.wait(WAIT_S)
        return original(*args, **kwargs)

    monkeypatch.setattr(collector._delivery, "drain_pending", stuck)
    collector.start()
    try:
        record(collector, resource_id="first")
        record(collector, resource_id="second")
        assert entered.wait(WAIT_S)
        record(collector, resource_id="third")

        started = time.monotonic()
        with mock.patch(POST) as post_mock:
            post_mock.return_value.status_code = 200
            complete = collector.stop(timeout=0.3)
        elapsed = time.monotonic() - started

        assert complete is False
        assert elapsed < 3
        assert post_mock.call_count == 0
        assert owned_frames(collector) == 3
        assert collector._delivery.queue.empty()
    finally:
        release.set()
        for thread in collector._delivery.threads:
            thread.join(timeout=WAIT_S)

    with mock.patch(POST) as later_post_mock:
        later_post_mock.return_value.status_code = 200
        collector.flush()

    assert later_post_mock.call_count == 1
    assert sum(row["processed_frames"] for row in sent_rows(later_post_mock)) == 3
    assert not usage_threads()
    assert collector._delivery.queue.empty()
    assert owned_frames(collector) == 0


def test_stop_with_a_slow_queue_write_is_bounded_and_loses_nothing(
    collector, monkeypatch
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    queue = BlockingQueue()
    collector._delivery.queue = queue
    collector.start()
    try:
        record(collector, resource_id="first", frames=1)
        record(collector, resource_id="second", frames=2)
        assert queue.entered.wait(WAIT_S)
        record(collector, resource_id="third", frames=4)

        started = time.monotonic()
        with mock.patch(POST) as post_mock:
            post_mock.return_value.status_code = 200
            complete = collector.stop(timeout=0.3)
        elapsed = time.monotonic() - started

        assert complete is False
        assert elapsed < 3
        assert post_mock.call_count == 0
        assert owned_frames(collector) == 7
    finally:
        queue.release.set()
        for thread in collector._delivery.threads:
            thread.join(timeout=WAIT_S)

    assert not usage_threads()
    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()

    rows = sent_rows(post_mock)
    assert Counter(row["resource_id"] for row in rows) == {
        "first": 1,
        "second": 1,
        "third": 1,
    }
    assert sum(row["processed_frames"] for row in rows) == 7
    assert collector._delivery.queue.empty()
    assert owned_frames(collector) == 0


def produced_rows(collector):
    collector._write_current_usage_to_queue()
    rows = [
        row
        for payload in queued_payloads(collector)
        for resource_rows in payload.values()
        for row in resource_rows.values()
    ]

    return rows


def listed_entries(rows, key):
    entries = [
        entry
        for row in rows
        for entry in json.loads(row["resource_details"]).get(key, [])
    ]

    return entries


def oversized_entries(list_key, count):
    if list_key == "models":
        return [model_entry(f"model/{index}") for index in range(count)]

    return [custom_python_entry(f"step-{index}") for index in range(count)]


def entry_identity(entry):
    return entry.get("model_id") or entry["step_name"]


@pytest.mark.parametrize("entry_count", [257, 1000])
@pytest.mark.parametrize("after_existing_row", [False, True])
@pytest.mark.parametrize("list_key", ["models", "custom_python"])
def test_a_call_with_more_entries_than_the_bound_is_recorded_as_one_row(
    collector, entry_count, after_existing_row, list_key
):
    entries = oversized_entries(list_key, entry_count)
    expected_frames = [7]
    if after_existing_row:
        record(
            collector,
            frames=1,
            execution_duration=0.5,
            resource_details={"models": [model_entry("earlier/1")]},
        )
        expected_frames.insert(0, 1)
    record(
        collector,
        frames=7,
        execution_duration=2.0,
        source_duration=3.0,
        resource_details={list_key: entries},
        megapixel_buckets={"1-2": {"processed_frames": 7, "execution_duration": 2.0}},
    )

    rows = produced_rows(collector)

    assert sorted(row["processed_frames"] for row in rows) == sorted(expected_frames)
    (big,) = [row for row in rows if row["processed_frames"] == 7]
    big_details = json.loads(big["resource_details"])
    assert [entry_identity(entry) for entry in big_details[list_key]] == [
        entry_identity(entry) for entry in entries
    ]
    assert big["source_duration"] == 3.0
    assert big["execution_duration"] == 2.0
    assert big["megapixel_buckets"]["1-2"] == {
        "processed_frames": 7,
        "execution_duration": 2.0,
    }
    if after_existing_row:
        (earlier,) = [row for row in rows if row["processed_frames"] == 1]
        assert [
            entry["model_id"]
            for entry in json.loads(earlier["resource_details"])["models"]
        ] == ["earlier/1"]


def test_a_call_with_two_oversized_lists_is_recorded_as_one_row_holding_both(
    collector,
):
    models = [model_entry(f"model/{index}") for index in range(300)]
    custom_python = [custom_python_entry(f"step-{index}") for index in range(600)]
    record(
        collector,
        frames=5,
        resource_details={"models": models, "custom_python": custom_python},
    )

    (row,) = produced_rows(collector)

    details = json.loads(row["resource_details"])
    assert row["processed_frames"] == 5
    assert [entry["model_id"] for entry in details["models"]] == [
        entry["model_id"] for entry in models
    ]
    assert [entry["step_name"] for entry in details["custom_python"]] == [
        entry["step_name"] for entry in custom_python
    ]


def test_a_call_after_an_oversized_row_closes_it_and_starts_another(collector):
    bound = payload_helpers.MAX_BILLABLE_ENTRIES_PER_ROW
    big = oversized_entries("models", bound + 1)
    record(collector, frames=3, resource_details={"models": big})
    record(collector, frames=2, resource_details={"models": [model_entry("next/1")]})

    rows = produced_rows(collector)

    assert sorted(row["processed_frames"] for row in rows) == [2, 3]
    by_frames = {row["processed_frames"]: row for row in rows}
    assert len(json.loads(by_frames[3]["resource_details"])["models"]) == len(big)
    assert [
        entry["model_id"]
        for entry in json.loads(by_frames[2]["resource_details"])["models"]
    ] == ["next/1"]


def test_every_row_produced_by_the_bound_has_at_least_one_frame(collector, monkeypatch):
    monkeypatch.setattr(payload_helpers, "MAX_BILLABLE_ENTRIES_PER_ROW", 3)
    sizes = [1, 2, 5, 1, 1, 4, 3, 1, 8, 2, 1]
    for call_index, size in enumerate(sizes):
        record(
            collector,
            frames=call_index + 1,
            resource_details={
                "models": [
                    model_entry(f"model/{call_index}/{index}") for index in range(size)
                ]
            },
        )

    rows = produced_rows(collector)

    assert len(rows) > 1
    assert all(row["processed_frames"] >= 1 for row in rows)
    assert sum(row["processed_frames"] for row in rows) == sum(range(1, len(sizes) + 1))
    assert sum(
        len(json.loads(row["resource_details"])["models"]) for row in rows
    ) == sum(sizes)


def test_usage_is_conserved_across_a_failed_send_a_row_bound_flush_and_a_later_send(
    collector, monkeypatch
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 2)
    ticks = itertools.count(1000)
    monkeypatch.setattr(collector_module.time, "time_ns", lambda: next(ticks))
    expected_models = defaultdict(lambda: {"frames": 0, "exec": 0.0})
    expected_steps = defaultdict(float)
    expected_frames = []

    def call(resource_id, models, steps, *, frames, duration):
        record(
            collector,
            resource_id=resource_id,
            frames=frames,
            execution_duration=duration,
            resource_details={"models": models, "custom_python": steps},
        )
        expected_frames.append(frames)
        for entry in models:
            totals = expected_models[entry["model_id"]]
            totals["frames"] += entry["frames"]
            totals["exec"] += entry["execution_duration"]
        for step in steps:
            expected_steps[step["step_name"]] += step["execution_duration"]

    attempts = []
    statuses = [500, 200]

    def post(url, **kwargs):
        status = statuses.pop(0)
        attempts.append((status, kwargs["json"]))

        return mock.MagicMock(status_code=status)

    call(
        "r1",
        [model_entry("m1", frames=2, execution_duration=0.5, latency=10.0)],
        [custom_python_entry("s1", execution_duration=0.25)],
        frames=2,
        duration=0.5,
    )
    call(
        "r1",
        [model_entry("m2", frames=3, execution_duration=0.25, latency=5.0)],
        [custom_python_entry("s2", execution_duration=0.5)],
        frames=3,
        duration=0.25,
    )
    with mock.patch(POST, side_effect=post):
        collector.flush()
        call(
            "r1",
            [
                model_entry("m2", frames=1, execution_duration=0.125, latency=4.0),
                model_entry("m3", frames=4, execution_duration=1.0, latency=2.0),
            ],
            [custom_python_entry("s1", execution_duration=0.25)],
            frames=5,
            duration=1.125,
        )
        call(
            "r2",
            [model_entry("m3", frames=1, execution_duration=0.5, latency=1.0)],
            [custom_python_entry("s3", execution_duration=1.0)],
            frames=1,
            duration=0.5,
        )
        call(
            "r3",
            [model_entry("m1", frames=2, execution_duration=0.25, latency=3.0)],
            [custom_python_entry("s3", execution_duration=0.5)],
            frames=2,
            duration=0.25,
        )
        collector.flush()

    (failed_status, failed_rows), (accepted_status, accepted_rows) = attempts
    assert (failed_status, accepted_status) == (500, 200)
    assert sum(row["processed_frames"] for row in failed_rows) == 5
    assert sum(row["processed_frames"] for row in accepted_rows) == sum(expected_frames)
    accepted_models = defaultdict(lambda: {"frames": 0, "exec": 0.0})
    for entry in listed_entries(accepted_rows, "models"):
        totals = accepted_models[entry["model_id"]]
        totals["frames"] += entry["frames"]
        totals["exec"] += entry["execution_duration"]
    accepted_steps = defaultdict(float)
    for step in listed_entries(accepted_rows, "custom_python"):
        accepted_steps[step["step_name"]] += step["execution_duration"]
    assert dict(accepted_models) == dict(expected_models)
    assert dict(accepted_steps) == dict(expected_steps)
    assert collector._delivery.queue.empty()
    assert produced_rows(collector) == []


class CountingQueue(Queue):
    def __init__(self):
        super().__init__()
        self.reads = 0

    def get_nowait(self):
        self.reads += 1

        return super().get_nowait()

    def take(self, known_hashes):
        self.reads += 1
        taken = []
        while not super().empty():
            taken.append(super().get_nowait())

        return taken


def backoff_s(collector):
    remaining = collector._delivery.send_backoff_s

    return remaining


def test_a_failed_pass_defers_the_sender_by_the_flush_interval_then_doubles_it(
    collector, monkeypatch
):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 10)
    monkeypatch.setattr(delivery_module, "SEND_BACKOFF_CAP_S", 35)
    record(collector, frames=1)
    delays = []

    with mock.patch(POST, side_effect=ConnectionError("down")):
        for _ in range(5):
            collector.flush()
            delays.append(backoff_s(collector))

    assert delays == pytest.approx([10, 20, 35, 35, 35], abs=1)
    assert total_frames(queued_payloads(collector)) == 1


def test_an_accepted_request_resets_the_backoff(collector, monkeypatch):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 10)
    record(collector, frames=1)

    with mock.patch(POST, side_effect=ConnectionError("down")):
        collector.flush()
        collector.flush()
    deferred = backoff_s(collector)
    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()
    after_success = backoff_s(collector)
    record(collector, frames=1)
    with mock.patch(POST, side_effect=ConnectionError("down")):
        collector.flush()
    after_next_failure = backoff_s(collector)

    assert deferred == pytest.approx(20, abs=1)
    assert after_success == 0.0
    assert after_next_failure == pytest.approx(10, abs=1)


def test_a_pass_with_nothing_to_send_leaves_the_backoff_alone(collector, monkeypatch):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 10)
    record(collector, frames=1)

    with mock.patch(POST, side_effect=ConnectionError("down")):
        collector.flush()
    collector._delivery.queue = Queue()
    with mock.patch(POST) as post_mock:
        collector.flush()

    post_mock.assert_not_called()
    assert backoff_s(collector) == pytest.approx(10, abs=1)


def test_sender_lifecycle_does_not_take_rows_while_backing_off(monkeypatch):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 0.01)
    usage_collector = UsageCollector()
    usage_collector._system_info = dict(SYSTEM_INFO)
    queue = CountingQueue()
    usage_collector._delivery.queue = queue
    record(usage_collector, frames=2)
    usage_collector._write_current_usage_to_queue()
    usage_collector._delivery._next_send_at = time.monotonic() + 60

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        try:
            usage_collector.start()
            threading.Event().wait(0.2)
            reads_while_backing_off = queue.reads
            posts_while_backing_off = post_mock.call_count
        finally:
            complete = usage_collector.stop()

    assert complete is True
    assert reads_while_backing_off == 0
    assert posts_while_backing_off == 0
    assert post_mock.call_count == 1
    assert not usage_threads()


def test_a_backoff_set_after_the_sender_precheck_stops_the_pass_from_taking_rows(
    monkeypatch,
):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 10)
    usage_collector = UsageCollector()
    usage_collector._system_info = dict(SYSTEM_INFO)
    delivery = usage_collector._delivery
    queue = CountingQueue()
    delivery.queue = queue
    record(usage_collector, frames=2)
    usage_collector._write_current_usage_to_queue()
    delivery._sender_wait_s = lambda: 0.01
    paused = threading.Event()
    resume = threading.Event()
    finished = threading.Event()
    original = delivery._send_in_background

    def paused_send():
        paused.set()
        resume.wait(WAIT_S)
        original()
        finished.set()

    delivery._send_in_background = paused_send
    sender = threading.Thread(target=delivery._sender_loop, daemon=True)

    with mock.patch(POST, side_effect=ConnectionError("down")) as post_mock:
        sender.start()
        try:
            assert paused.wait(WAIT_S)
            queue.reads = 0
            usage_collector.flush()
            assert backoff_s(usage_collector) > 0
            reads_by_flush = queue.reads
            resume.set()
            assert finished.wait(WAIT_S)
        finally:
            delivery._stopping.set()
            resume.set()
            sender.join(WAIT_S)

    assert queue.reads == reads_by_flush
    assert post_mock.call_count == 1


def test_sender_lifecycle_retries_after_the_backoff_and_keeps_the_rows(
    monkeypatch, caplog
):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 0.01)
    usage_collector = UsageCollector()
    usage_collector._system_info = dict(SYSTEM_INFO)
    record(usage_collector, frames=3)
    attempts = threading.Semaphore(0)

    def post(url, **kwargs):
        attempts.release()
        raise ConnectionError("down")

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage"), mock.patch(
        POST, side_effect=post
    ):
        try:
            usage_collector.start()
            seen = [attempts.acquire(timeout=WAIT_S) for _ in range(3)]
        finally:
            complete = usage_collector.stop()

    assert seen == [True, True, True]
    assert complete is True
    assert not usage_threads()
    assert total_frames(queued_payloads(usage_collector)) == 3
    records = [
        record
        for record in caplog.records
        if record.name.startswith("inference_server.usage")
    ]
    assert records
    assert all(record.levelno == logging.DEBUG for record in records)
