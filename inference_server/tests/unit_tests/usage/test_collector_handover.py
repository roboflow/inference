import json
import logging
import socket
import sqlite3
import threading
import time
import types
from collections import Counter
from queue import Queue
from unittest import mock

import pytest

from inference_server import configuration
from inference_server.usage import collector as collector_module
from inference_server.usage import delivery as delivery_module
from inference_server.usage import queues
from inference_server.usage.collector import UsageCollector
from inference_server.usage.payload_helpers import sha256_hash
from tests.unit_tests.usage.conftest import SYSTEM_INFO
from tests.unit_tests.usage.test_collector import POST, record, usage_key
from tests.unit_tests.usage.test_collector_delivery import (
    WAIT_S,
    sent_rows,
    usage_threads,
)
from tests.unit_tests.usage.test_queues import FakeRedis, StoringRedis

DROP_LINE = delivery_module.DROP_LOG_LINE


def set_queue(usage_collector, queue):
    usage_collector._delivery.queue = queue


def pending_rows(usage_collector):
    return usage_collector._delivery._pending_rows


def pending_frames(usage_collector):
    frames = sum(
        row["processed_frames"]
        for item in usage_collector._delivery.pending
        for rows in item.payload.values()
        for row in rows.values()
    )

    return frames


def swap_pending_lock(usage_collector, lock):
    usage_collector._delivery._pending_lock = lock


def swap_queue_lock(usage_collector, lock):
    usage_collector._delivery._queue_lock = lock


def background_threads(usage_collector):
    return usage_collector._delivery.threads


def patch_clock(monkeypatch, clock):
    namespace = types.SimpleNamespace(monotonic=clock)
    monkeypatch.setattr(delivery_module, "time", namespace)


def stored_frames(payloads):
    frames = [
        row["processed_frames"]
        for payload in payloads
        for rows in payload.values()
        for row in rows.values()
    ]

    return frames


class FlakyQueue(Queue):
    def __init__(self, failures, *, raising=False):
        super().__init__()
        self.failures = failures
        self.raising = raising
        self.attempts = 0
        self.stored = []

    def put(self, item, block=True, timeout=None):
        self.attempts += 1
        if self.attempts <= self.failures:
            if self.raising:
                raise OSError("planted-secret")
            return False
        super().put(item, block=False)
        self.stored.append(item)

        return True


class RecordingQueue(Queue):
    def __init__(self):
        super().__init__()
        self.ops = []
        self.block_puts = False
        self.put_entered = threading.Event()
        self.put_release = threading.Event()

    def mark(self, name):
        self.ops.append((name, "marker"))

    def put(self, item, block=True, timeout=None):
        self.ops.append(("put", threading.current_thread().name))
        if self.block_puts:
            self.put_entered.set()
            assert self.put_release.wait(WAIT_S)
        super().put(item, block=False)

    def get_nowait(self):
        self.ops.append(("get", threading.current_thread().name))

        return super().get_nowait()

    def empty(self):
        self.ops.append(("empty", threading.current_thread().name))

        return super().empty()

    def ops_after(self, marker):
        names = [name for name, _ in self.ops]
        index = len(names) - 1 - names[::-1].index(marker)

        return self.ops[index + 1 :]


class SamplingLock:
    def __init__(self, sample):
        self._lock = threading.Lock()
        self._sample = sample
        self.peak = 0

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *exc_info):
        self.peak = max(self.peak, self._sample())
        self._lock.release()

    def acquire(self, *args, **kwargs):
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()


class ForbiddenLock:
    def __enter__(self):
        raise AssertionError("the queue lock was taken")

    def __exit__(self, *exc_info):
        return False

    def acquire(self, *args, **kwargs):
        raise AssertionError("the queue lock was taken")

    def release(self):
        raise AssertionError("the queue lock was released")


def run_in_thread(function, timeout=WAIT_S):
    outcome = {}

    def target():
        outcome["value"] = function()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    return thread, outcome


@pytest.mark.parametrize("failures", [1, 3, 7])
@pytest.mark.parametrize("raising", [False, True])
def test_a_queue_that_refuses_a_write_keeps_the_item_pending_in_order(
    collector, monkeypatch, failures, raising
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    queue = FlakyQueue(failures, raising=raising)
    set_queue(collector, queue)
    for index in range(4):
        record(collector, resource_id=f"resource-{index}", frames=2**index)

    for _ in range(failures + 4):
        collector._write_current_usage_to_queue()

    assert stored_frames(queue.stored) == [1, 2, 4, 8]
    assert queue.attempts == failures + 4
    assert pending_frames(collector) == 0
    assert pending_rows(collector) == 0
    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()
    assert sum(row["processed_frames"] for row in sent_rows(post_mock)) == 15


def test_a_locked_sqlite_file_keeps_the_rows_pending_until_it_can_be_written(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    monkeypatch.setattr(queues, "SQLITE_TIMEOUT_S", 0.05)
    db_file = tmp_path / "usage.db"
    usage_collector = UsageCollector(sqlite_db_file_path=db_file)
    usage_collector._system_info = dict(SYSTEM_INFO)
    record(usage_collector, api_key="key-1", frames=3)
    holder = sqlite3.connect(str(db_file), timeout=0.05, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        usage_collector._write_current_usage_to_queue()
        owned_while_locked = pending_frames(usage_collector)
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    usage_collector._write_current_usage_to_queue()

    assert owned_while_locked == 3
    assert pending_frames(usage_collector) == 0
    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        usage_collector.flush()
    assert post_mock.call_count == 1
    assert sum(row["processed_frames"] for row in sent_rows(post_mock)) == 3


def redis_collector(monkeypatch, client):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    monkeypatch.setattr(configuration, "REDIS_HOST", "redis.local")
    usage_collector = UsageCollector(redis_client=client)
    usage_collector._system_info = dict(SYSTEM_INFO)

    return usage_collector


def test_a_redis_pipeline_that_raises_gives_the_rows_up_as_unconfirmed(monkeypatch):
    client = FakeRedis(error=RuntimeError("planted-secret"))
    usage_collector = redis_collector(monkeypatch, client)
    record(usage_collector, frames=3)

    usage_collector._write_current_usage_to_queue()
    client.error = None
    usage_collector._write_current_usage_to_queue()

    assert pending_frames(usage_collector) == 0
    assert pending_rows(usage_collector) == 0
    assert (usage_collector.unconfirmed_rows, usage_collector.unconfirmed_frames) == (
        1,
        3,
    )
    assert usage_collector.dropped_rows == 0
    assert client.executed == []


def test_a_redis_write_that_may_have_landed_is_not_retried_and_is_counted(
    monkeypatch, caplog
):
    client = StoringRedis(raise_after_applying=True)
    usage_collector = redis_collector(monkeypatch, client)
    record(usage_collector, api_key="planted-api-key", frames=3)
    record(usage_collector, api_key="planted-api-key", resource_id="other", frames=4)

    with caplog.at_level(logging.DEBUG):
        for _ in range(4):
            usage_collector._write_current_usage_to_queue()

    assert client.attempts == 1
    assert len(client.store) == 1
    assert pending_rows(usage_collector) == 0
    assert (usage_collector.unconfirmed_rows, usage_collector.unconfirmed_frames) == (
        2,
        7,
    )
    lines = [
        r
        for r in caplog.records
        if delivery_module.UNCONFIRMED_LOG_LINE in r.getMessage()
    ]
    assert len(lines) == 1
    assert "planted" not in caplog.text


def test_unconfirmed_writes_are_reported_once_per_flush_interval(monkeypatch, caplog):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 10)
    now = [100.0]
    patch_clock(monkeypatch, lambda: now[0])
    client = StoringRedis(raise_after_applying=True)
    usage_collector = redis_collector(monkeypatch, client)

    with caplog.at_level(logging.DEBUG):
        for index in range(3):
            record(usage_collector, resource_id=f"resource-{index}")
            usage_collector._write_current_usage_to_queue()
        first_interval = [
            r
            for r in caplog.records
            if delivery_module.UNCONFIRMED_LOG_LINE in r.getMessage()
        ]
        now[0] += 11
        record(usage_collector, resource_id="resource-3")
        usage_collector._write_current_usage_to_queue()

    lines = [
        r
        for r in caplog.records
        if delivery_module.UNCONFIRMED_LOG_LINE in r.getMessage()
    ]
    assert len(first_interval) == 1
    assert len(lines) == 2
    assert usage_collector.unconfirmed_rows == 4


def test_a_redis_write_that_was_not_acknowledged_is_given_up_and_never_retried(
    monkeypatch,
):
    client = StoringRedis(zadd_results=[0])
    usage_collector = redis_collector(monkeypatch, client)
    record(usage_collector, frames=3)

    usage_collector._write_current_usage_to_queue()
    usage_collector._write_current_usage_to_queue()
    usage_collector.flush()

    assert client.attempts == 1
    assert pending_rows(usage_collector) == 0
    assert pending_frames(usage_collector) == 0
    assert (usage_collector.unconfirmed_rows, usage_collector.unconfirmed_frames) == (
        1,
        3,
    )
    assert len(client.store) == 1
    assert list(client.members) == list(client.store)


def test_pending_rows_never_pass_the_bound_while_the_queue_accepts_writes(
    collector, monkeypatch
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    monkeypatch.setattr(delivery_module, "MAX_PENDING_ROWS", 3)
    queue = FlakyQueue(0)
    set_queue(collector, queue)
    lock = SamplingLock(lambda: pending_rows(collector))
    swap_pending_lock(collector, lock)

    for index in range(12):
        record(collector, resource_id=f"resource-{index}", frames=3)

    assert lock.peak <= 3
    assert collector.dropped_rows == 0
    collector._write_current_usage_to_queue()
    assert sum(stored_frames(queue.stored)) == 36


def test_concurrent_overflow_never_passes_the_bound(collector, monkeypatch):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    monkeypatch.setattr(delivery_module, "MAX_PENDING_ROWS", 4)
    queue = FlakyQueue(0)
    set_queue(collector, queue)
    lock = SamplingLock(lambda: pending_rows(collector))
    swap_pending_lock(collector, lock)

    def worker(worker_index):
        for index in range(40):
            record(collector, resource_id=f"worker-{worker_index}-{index}", frames=1)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=WAIT_S)

    assert not [thread for thread in threads if thread.is_alive()]
    assert lock.peak <= 4
    assert collector.dropped_rows == 0
    collector._write_current_usage_to_queue()
    assert sum(stored_frames(queue.stored)) == 640


def test_a_failing_make_room_write_drops_the_oldest_and_counts_exactly_those_rows(
    collector, monkeypatch
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    monkeypatch.setattr(delivery_module, "MAX_PENDING_ROWS", 2)
    queue = FlakyQueue(10**6)
    set_queue(collector, queue)

    for index in range(5):
        record(collector, resource_id=f"resource-{index}", frames=2**index)

    assert collector.dropped_rows == 2
    assert collector.dropped_frames == 1 + 2
    assert pending_rows(collector) == 2
    assert pending_frames(collector) == 4 + 8
    queue.failures = 0
    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()
    assert sum(row["processed_frames"] for row in sent_rows(post_mock)) == 4 + 8 + 16


def test_drops_are_reported_by_one_fixed_line_per_flush_interval(
    collector, monkeypatch, caplog
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    monkeypatch.setattr(delivery_module, "MAX_PENDING_ROWS", 1)
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 10)
    now = [100.0]
    patch_clock(monkeypatch, lambda: now[0])
    set_queue(collector, FlakyQueue(10**6))

    with caplog.at_level(logging.DEBUG):
        for index in range(6):
            record(
                collector,
                api_key="planted-api-key",
                resource_id=f"planted-resource-{index}",
            )
        first_interval = [r for r in caplog.records if DROP_LINE in r.getMessage()]
        now[0] += 11
        for index in range(6, 9):
            record(
                collector,
                api_key="planted-api-key",
                resource_id=f"planted-resource-{index}",
            )

    lines = [r for r in caplog.records if DROP_LINE in r.getMessage()]
    assert len(first_interval) == 1
    assert len(lines) == 2
    assert collector.dropped_rows == 7
    assert "planted" not in caplog.text


def test_recording_below_the_bound_never_takes_the_queue_lock_or_resolves_the_host(
    collector, monkeypatch
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 2)
    collector._system_info = {}
    swap_queue_lock(collector, ForbiddenLock())

    with mock.patch.object(UsageCollector, "system_info") as system_info:
        for index in range(20):
            record(collector, resource_id=f"resource-{index}")

    system_info.assert_not_called()
    assert pending_rows(collector) == 18


def test_a_writer_that_cannot_get_the_host_in_time_uses_the_offline_values(
    monkeypatch,
):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    monkeypatch.setattr(collector_module, "SYSTEM_INFO_WAIT_S", 0.05, raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: "fallback-host")
    original = UsageCollector.system_info
    entered = threading.Event()
    release = threading.Event()

    def lookup(ip_address=None, hostname=None, dedicated_deployment_id=None):
        if ip_address is None:
            entered.set()
            assert release.wait(WAIT_S)
            return dict(SYSTEM_INFO)
        return original(
            ip_address=ip_address,
            hostname=hostname,
            dedicated_deployment_id=dedicated_deployment_id,
        )

    monkeypatch.setattr(UsageCollector, "system_info", staticmethod(lookup))
    usage_collector = UsageCollector()
    try:
        usage_collector.start()
        assert entered.wait(WAIT_S)
        record(usage_collector, frames=2)

        def flush():
            with mock.patch(POST) as post_mock:
                post_mock.return_value.status_code = 200
                usage_collector.flush()
            return sent_rows(post_mock)

        thread, outcome = run_in_thread(flush)
    finally:
        release.set()
    assert not thread.is_alive()
    usage_collector.stop()

    (row,) = outcome["value"]
    assert row["hostname"] == sha256_hash("fallback-host")
    assert row["ip_address_hash"] == sha256_hash("127.0.0.1")
    assert usage_collector._system_info["hostname"] == sha256_hash("fallback-host")
    assert not usage_threads()


def test_a_row_is_completed_once_even_when_its_write_is_retried(collector):
    collector._system_info = {}
    queue = FlakyQueue(2)
    set_queue(collector, queue)
    record(collector, frames=1)

    with mock.patch.object(
        UsageCollector, "system_info", return_value=dict(SYSTEM_INFO)
    ) as system_info:
        collector.record_system_info()
        for _ in range(3):
            collector._write_current_usage_to_queue()

    system_info.assert_called_once()
    (payload,) = queue.stored
    (row,) = next(iter(payload.values())).values()
    assert row["hostname"] == "host1"
    assert row["ip_address_hash"] == "ab12c"


def test_a_stopped_sender_hands_back_unsent_keys_without_queue_io_and_posts_each_once(
    collector, monkeypatch
):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 0.01)
    queue = RecordingQueue()
    set_queue(collector, queue)
    for key, frames in (("key-1", 1), ("key-2", 2), ("key-3", 4)):
        record(collector, api_key=key, frames=frames)
    entered = threading.Event()
    release = threading.Event()
    bearers = []
    bodies = []

    def post(url, **kwargs):
        bearers.append(kwargs["headers"]["Authorization"])
        bodies.append(kwargs["json"])
        if len(bearers) == 1:
            entered.set()
            assert release.wait(WAIT_S)
        return mock.MagicMock(status_code=200)

    with mock.patch(POST, side_effect=post):
        collector.start()
        try:
            assert entered.wait(WAIT_S)
            queue.mark("stop-called")
            started_threads = background_threads(collector)
            complete = collector.stop(timeout=0.2)
            main_ops = [
                op
                for op in queue.ops_after("stop-called")
                if op[1] == threading.current_thread().name
            ]
        finally:
            release.set()
        for thread in started_threads:
            thread.join(timeout=WAIT_S)

        assert complete is False
        assert main_ops == []
        assert len(bearers) == 1
        assert pending_frames(collector) == 2 + 4
        assert queue.ops_after("stop-called") == []
        collector.flush()

    assert Counter(bearers) == {
        "Bearer key-1": 1,
        "Bearer key-2": 1,
        "Bearer key-3": 1,
    }
    assert sum(row["processed_frames"] for body in bodies for row in body) == 7
    assert not usage_threads()


def test_a_background_thread_does_no_queue_write_after_stop_marked_it(
    collector, monkeypatch
):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    queue = RecordingQueue()
    queue.block_puts = True
    set_queue(collector, queue)
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    collector.start()
    try:
        record(collector, resource_id="first", frames=1)
        record(collector, resource_id="second", frames=2)
        assert queue.put_entered.wait(WAIT_S)
        record(collector, resource_id="third", frames=4)
        queue.mark("stop-called")
        threads = background_threads(collector)

        complete = collector.stop(timeout=0.2)
        main_ops = [
            op
            for op in queue.ops_after("stop-called")
            if op[1] == threading.current_thread().name
        ]
    finally:
        queue.put_release.set()
    for thread in threads:
        thread.join(timeout=WAIT_S)

    assert complete is False
    assert main_ops == []
    assert [op for op in queue.ops_after("stop-called") if op[0] == "put"] == []
    assert pending_frames(collector) == 7 - stored_frames_in_queue(queue)
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


def stored_frames_in_queue(queue):
    frames = sum(
        row["processed_frames"]
        for payload in list(queue.queue)
        for rows in payload.values()
        for row in rows.values()
    )

    return frames


def test_record_usage_after_stop_is_a_counted_noop(collector, monkeypatch):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    collector.start()
    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        assert collector.stop() is True

        record(collector, frames=5)
        record(collector, frames=6)
        collector.flush()

    assert collector.ignored_after_stop == 2
    post_mock.assert_not_called()
    assert pending_frames(collector) == 0
    assert not collector._usage


def test_the_final_pass_of_stop_checks_the_deadline_before_every_post(
    collector, monkeypatch
):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    for index in range(10):
        record(collector, api_key=f"key-{index}", frames=index + 1)
    now = [0.0]
    patch_clock(monkeypatch, lambda: now[0])
    bearers = []
    bodies = []

    def post(url, **kwargs):
        bearers.append(kwargs["headers"]["Authorization"])
        bodies.append(kwargs["json"])
        now[0] += 1.0
        return mock.MagicMock(status_code=200)

    collector.start()
    with mock.patch(POST, side_effect=post):
        complete = collector.stop(timeout=3.5)

        assert complete is False
        assert len(bearers) == 4
        assert pending_frames(collector) == sum(range(5, 11))
        collector.flush()

    assert Counter(bearers) == {f"Bearer key-{index}": 1 for index in range(10)}
    assert sum(row["processed_frames"] for body in bodies for row in body) == 55
    assert not usage_threads()


def legacy_failed_send_payload(api_key, *, frames):
    api_key_hash = sha256_hash(api_key, length=-1)
    row = {
        "timestamp_start": 1,
        "timestamp_stop": 2,
        "exec_session_id": "legacy-session",
        "hostname": "legacy-host",
        "ip_address_hash": "12345",
        "processed_frames": frames,
        "fps": 0,
        "source_duration": 0,
        "category": "request",
        "resource_id": "workspace/model",
        "resource_details": "{}",
        "hosted": False,
        "is_gpu_available": False,
        "python_version": "3.10.0",
        "inference_version": "0.0.1",
        "enterprise": False,
        "execution_duration": 0.5,
        "megapixel_buckets": {},
        "api_key": api_key,
    }
    payload = {api_key_hash: {usage_key("request", "workspace/model"): row}}

    return api_key_hash, payload


def sqlite_texts(db_file):
    connection = sqlite3.connect(str(db_file))
    try:
        texts = [text for (text,) in connection.execute("SELECT payload FROM usage")]
    finally:
        connection.close()

    return texts


def failed_send_over_a_legacy_file(monkeypatch, tmp_path, api_key):
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    db_file = tmp_path / "usage.db"
    api_key_hash, payload = legacy_failed_send_payload(api_key, frames=9)
    queues.SQLiteQueue(db_file_path=db_file).put(payload)
    usage_collector = UsageCollector(sqlite_db_file_path=db_file)
    usage_collector._system_info = dict(SYSTEM_INFO)
    usage_collector._delivery.register_api_key(api_key)

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 500
        usage_collector.flush()
        texts_after_failure = sqlite_texts(db_file)
        bytes_after_failure = db_file.read_bytes()
        post_mock.return_value.status_code = 200
        usage_collector.flush()

    return (
        usage_collector,
        post_mock,
        api_key_hash,
        texts_after_failure,
        bytes_after_failure,
    )


def test_legacy_rows_with_a_plaintext_key_are_normalised_when_taken_from_sqlite(
    monkeypatch, tmp_path
):
    api_key = "legacy-plaintext-key"

    (
        usage_collector,
        post_mock,
        api_key_hash,
        texts_after_failure,
        _,
    ) = failed_send_over_a_legacy_file(monkeypatch, tmp_path, api_key)

    assert len(texts_after_failure) == 1
    assert api_key not in texts_after_failure[0]
    stored = json.loads(texts_after_failure[0])
    assert set(stored) == {api_key_hash}
    (stored_row,) = stored[api_key_hash].values()
    assert stored_row["api_key_hash"] == api_key_hash
    assert "api_key" not in stored_row
    assert post_mock.call_count == 2
    last_call = post_mock.call_args
    assert last_call.kwargs["headers"]["Authorization"] == f"Bearer {api_key}"
    (sent_row,) = last_call.kwargs["json"]
    assert sent_row["api_key"] == api_key
    assert "api_key_hash" not in sent_row
    assert sent_row["processed_frames"] == 9
    assert usage_collector._delivery.queue.empty()


def test_the_sqlite_file_bytes_hold_no_plaintext_key_after_the_failed_send_without_a_rewrite(
    monkeypatch, tmp_path
):
    api_key = "legacy-plaintext-key"

    _, _, _, _, bytes_after_failure = failed_send_over_a_legacy_file(
        monkeypatch, tmp_path, api_key
    )

    assert api_key.encode() not in bytes_after_failure


def sqlite_collector(db_file):
    usage_collector = UsageCollector(sqlite_db_file_path=db_file)
    usage_collector._system_info = dict(SYSTEM_INFO)

    return usage_collector


def stored_sqlite_rows(db_file):
    connection = sqlite3.connect(str(db_file))
    try:
        rows = connection.execute(
            "SELECT id, payload FROM usage ORDER BY id"
        ).fetchall()
    finally:
        connection.close()

    return rows


def test_rows_read_back_after_a_restart_wait_for_a_process_that_knows_the_key(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    db_file = tmp_path / "usage.db"
    first_process = sqlite_collector(db_file)
    record(first_process, api_key="restarted-key", frames=4)
    first_process._write_current_usage_to_queue()
    bytes_before = db_file.read_bytes()
    second_process = sqlite_collector(db_file)
    assert second_process._delivery._hashed_api_keys == {}

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        second_process.flush()
        posts_without_the_key = post_mock.call_count
        bytes_without_the_key = db_file.read_bytes()
        second_process._delivery.register_api_key("restarted-key")
        second_process.flush()

    assert posts_without_the_key == 0
    assert bytes_without_the_key == bytes_before
    assert post_mock.call_count == 1
    assert post_mock.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer restarted-key"
    )
    assert sum(row["processed_frames"] for row in sent_rows(post_mock)) == 4
    assert second_process._delivery.queue.empty()


def test_two_processes_sharing_the_sqlite_file_each_take_the_rows_of_their_own_keys(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    db_file = tmp_path / "usage.db"
    first_process = sqlite_collector(db_file)
    second_process = sqlite_collector(db_file)
    record(first_process, api_key="key-a", frames=3)
    first_process._write_current_usage_to_queue()
    rows_of_a = stored_sqlite_rows(db_file)
    record(second_process, api_key="key-b", frames=4)
    second_process._write_current_usage_to_queue()

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        second_process.flush()
        rows_after_second = stored_sqlite_rows(db_file)
        first_process.flush()

    bearers = [
        call.kwargs["headers"]["Authorization"] for call in post_mock.call_args_list
    ]
    assert bearers == ["Bearer key-b", "Bearer key-a"]
    assert [
        sum(row["processed_frames"] for row in call.kwargs["json"])
        for call in post_mock.call_args_list
    ] == [4, 3]
    assert rows_after_second == rows_of_a
    assert stored_sqlite_rows(db_file) == []


def test_a_process_with_no_known_key_takes_nothing_from_the_shared_sqlite_file(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    db_file = tmp_path / "usage.db"
    writer = sqlite_collector(db_file)
    record(writer, api_key="key-a", frames=3)
    writer._write_current_usage_to_queue()
    bytes_before = db_file.read_bytes()
    idle_process = sqlite_collector(db_file)

    with mock.patch(POST) as post_mock:
        idle_process.flush()
        complete = idle_process.stop()

    post_mock.assert_not_called()
    assert complete is True
    assert db_file.read_bytes() == bytes_before
    assert idle_process._delivery.pending_rows == 0


def test_stop_does_not_wait_for_a_held_usage_lock(collector):
    record(collector, frames=3)
    collector._usage_lock.acquire()
    held = True
    try:
        started = time.monotonic()
        stop_thread, stop_outcome = run_in_thread(
            lambda: collector.stop(timeout=0.05), timeout=3
        )
        stop_elapsed = time.monotonic() - started
        late_thread, _ = run_in_thread(lambda: record(collector, frames=5), timeout=3)
        late_finished = not late_thread.is_alive()
        stop_finished = not stop_thread.is_alive()
    finally:
        collector._usage_lock.release()
        held = False
    stop_thread.join(timeout=WAIT_S)
    late_thread.join(timeout=WAIT_S)

    assert held is False
    assert stop_finished
    assert stop_outcome["value"] is False
    assert stop_elapsed < 2
    assert late_finished
    assert collector.ignored_after_stop == 1
    assert owned_frames_of_window(collector) == 3
    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()
        collector.flush()
    assert post_mock.call_count == 1
    assert sum(row["processed_frames"] for row in sent_rows(post_mock)) == 3


def owned_frames_of_window(collector):
    frames = sum(
        row["processed_frames"]
        for rows in collector._usage.values()
        for row in rows.values()
    )

    return frames


class GateAfter:
    def __init__(self, allowed):
        self.allowed = allowed
        self.calls = 0

    def __call__(self):
        self.calls += 1

        return self.calls <= self.allowed


class CountingSQLiteQueue(queues.SQLiteQueue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads = 0
        self.on_read = None

    def take(self, known_hashes=None, **kwargs):
        self.reads += 1
        payloads = super().take(known_hashes, **kwargs)
        if self.on_read is not None:
            self.on_read()

        return payloads


def sqlite_row_count(queue):
    return len(sqlite_texts(queue._db_file_path))


def test_the_dequeue_loop_checks_the_gate_before_every_read_and_leaves_the_rest_queued(
    collector, tmp_path
):
    queue = CountingSQLiteQueue(db_file_path=tmp_path / "usage.db")
    set_queue(collector, queue)
    collector._delivery.register_api_key("key")
    for index in range(250):
        queue.put({"key": {f"row-{index}": {"processed_frames": 1}}})
    gate = GateAfter(1)

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        complete = collector._delivery.send_queued(gate)

    assert complete is False
    assert queue.reads == 1
    assert post_mock.call_count == 0
    assert collector._delivery.pending_rows == 100
    assert sqlite_row_count(queue) == 150


def test_the_dequeue_pass_ends_after_the_rows_present_when_it_started(
    collector, tmp_path
):
    queue = CountingSQLiteQueue(db_file_path=tmp_path / "usage.db")
    set_queue(collector, queue)
    for index in range(250):
        queue.put({"key": {f"row-{index}": {"processed_frames": 1}}})
    added = []

    def writer_adds_rows():
        if added:
            return
        for index in range(100):
            added.append(index)
            queue.put({"late": {f"late-{index}": {"processed_frames": 1}}})

    queue.on_read = writer_adds_rows

    taken = collector._delivery.dump_queue()

    assert queue.reads == 3
    assert len(taken) == 300
    assert sqlite_row_count(queue) == 50


def forbid_lookups(monkeypatch):
    lookups = []

    def refuse(*args, **kwargs):
        lookups.append("network")
        raise OSError("no network")

    monkeypatch.setattr(socket, "gethostname", lambda: "fallback-host")
    monkeypatch.setattr(socket, "gethostbyname", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(
        UsageCollector,
        "record_system_info",
        lambda *args, **kwargs: lookups.append("record"),
    )

    return lookups


def test_a_writer_before_start_uses_the_offline_values_without_any_lookup(
    collector, monkeypatch
):
    collector._system_info = {}
    lookups = forbid_lookups(monkeypatch)
    record(collector, frames=2)

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()

    (row,) = sent_rows(post_mock)
    assert row["hostname"] == sha256_hash("fallback-host")
    assert row["ip_address_hash"] == sha256_hash("127.0.0.1")
    assert lookups == []
    assert collector._system_info == {}


def test_the_make_room_path_before_start_uses_the_offline_values_without_any_lookup(
    collector, monkeypatch
):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 1)
    monkeypatch.setattr(delivery_module, "MAX_PENDING_ROWS", 1)
    collector._system_info = {}
    lookups = forbid_lookups(monkeypatch)
    queue = Queue()
    set_queue(collector, queue)

    for index in range(3):
        record(collector, resource_id=f"resource-{index}", frames=2)

    assert collector.inline_queue_writes > 0
    assert lookups == []
    rows = [
        row
        for payload in list(queue.queue)
        for resource_rows in payload.values()
        for row in resource_rows.values()
    ]
    assert rows
    assert {row["hostname"] for row in rows} == {sha256_hash("fallback-host")}
    assert {row["ip_address_hash"] for row in rows} == {sha256_hash("127.0.0.1")}


def test_stop_without_start_closes_admission_and_sends_the_window_once(collector):
    record(collector, frames=2)

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        complete = collector.stop()
        record(collector, frames=5)
        collector.flush()

    assert complete is True
    assert post_mock.call_count == 1
    assert sum(row["processed_frames"] for row in sent_rows(post_mock)) == 2
    assert collector.ignored_after_stop == 1
    assert not collector._usage


def test_a_call_after_stop_does_no_hashing_validation_or_map_growth(collector):
    collector.stop()

    record(collector, api_key="late-key", category="")
    record(collector, api_key="late-key", frames=2)

    assert collector.ignored_after_stop == 2
    assert collector._delivery._hashed_api_keys == {}
    assert not collector._usage
