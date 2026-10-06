import json
import logging
import sqlite3

import pytest

from inference_server import configuration
from inference_server.usage import queues
from inference_server.usage.queues import UNCONFIRMED, RedisQueue, SQLiteQueue


class FakePipeline:
    def __init__(self, client):
        self._client = client
        self._calls = []

    def set(self, **kwargs):
        self._calls.append(("set", kwargs))

    def zadd(self, **kwargs):
        self._calls.append(("zadd", kwargs))

    def execute(self):
        if self._client.error is not None:
            raise self._client.error
        self._client.executed.append(self._calls)

        return self._client.results


class FakeRedis:
    def __init__(self, results=(True, 1), error=None):
        self.results = list(results)
        self.error = error
        self.executed = []

    def pipeline(self):
        return FakePipeline(self)


class StoringPipeline:
    def __init__(self, client):
        self._client = client
        self._calls = []

    def set(self, **kwargs):
        self._calls.append(("set", kwargs))

    def zadd(self, **kwargs):
        self._calls.append(("zadd", kwargs))

    def execute(self):
        client = self._client
        client.attempts += 1
        results = []
        for command, kwargs in self._calls:
            if command == "set":
                client.store[kwargs["name"]] = kwargs["value"]
                results.append(True)
            else:
                client.members.update(kwargs["mapping"])
                results.append(1)
        if client.zadd_results:
            results[1] = client.zadd_results.pop(0)
        if client.raise_after_applying:
            raise ConnectionResetError("planted-secret")

        return results


class StoringRedis:
    def __init__(self, *, raise_after_applying=False, zadd_results=()):
        self.raise_after_applying = raise_after_applying
        self.zadd_results = list(zadd_results)
        self.attempts = 0
        self.store = {}
        self.members = {}

    def pipeline(self):
        return StoringPipeline(self)


def test_empty():
    conn = sqlite3.connect(":memory:")
    q = SQLiteQueue(sqlite_connection=conn)

    assert q.empty(sqlite_connection=conn) is True
    conn.close()


def test_not_empty():
    conn = sqlite3.connect(":memory:")
    q = SQLiteQueue(sqlite_connection=conn)

    q.put("test", sqlite_connection=conn)

    assert q.empty(sqlite_connection=conn) is False
    conn.close()


def test_get_nowait():
    conn = sqlite3.connect(":memory:")
    q = SQLiteQueue(sqlite_connection=conn)

    q.put({"test": "test"}, sqlite_connection=conn)
    q.put({"test": "test"}, sqlite_connection=conn)
    q.put({"test": "test"}, sqlite_connection=conn)

    usage_payloads = q.get_nowait(sqlite_connection=conn)
    assert usage_payloads == [{"test": "test"}, {"test": "test"}, {"test": "test"}]
    assert q.empty(sqlite_connection=conn) is True
    conn.close()


def test_sqlite_queue_default_file_is_usage_db_in_model_cache_dir(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(cache_dir))

    q = SQLiteQueue()
    q.put({"key": {"resource": {"processed_frames": 1}}})

    db_file = cache_dir / "usage.db"
    assert db_file.exists()
    conn = sqlite3.connect(str(db_file))
    schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='usage'"
    ).fetchone()[0]
    stored = conn.execute("SELECT id, payload FROM usage").fetchall()
    conn.close()
    assert "payload TEXT NOT NULL" in schema
    assert "id INTEGER PRIMARY KEY" in schema
    assert stored == [(1, json.dumps({"key": {"resource": {"processed_frames": 1}}}))]
    assert q.full() is False


def test_sqlite_queue_returns_at_most_one_hundred_oldest_payloads(tmp_path):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    for index in range(101):
        q.put({"index": index})

    first = q.get_nowait()
    second = q.get_nowait()

    assert [payload["index"] for payload in first] == list(range(100))
    assert second == [{"index": 100}]
    assert q.empty() is True


def test_redis_queue_writes_the_payload_and_the_sorted_set_in_one_pipeline(
    monkeypatch,
):
    client = FakeRedis()
    monkeypatch.setattr(queues.time, "time", lambda: 1700000000.5)
    q = RedisQueue(redis_client=client)
    payload = {"api-key": {"request:coco/3": {"processed_frames": 1}}}

    q.put(payload)
    q.put("already-serialised")

    prefix = q._prefix
    assert prefix.startswith("{UsageCollector}:1700000000.5:")
    assert len(prefix.split(":")[-1]) == 5
    assert client.executed == [
        [
            ("set", {"name": f"{prefix}:1", "value": json.dumps(payload)}),
            (
                "zadd",
                {"name": "UsageCollector", "mapping": {f"{prefix}:1": 1700000000.5}},
            ),
        ],
        [
            ("set", {"name": f"{prefix}:2", "value": "already-serialised"}),
            (
                "zadd",
                {"name": "UsageCollector", "mapping": {f"{prefix}:2": 1700000000.5}},
            ),
        ],
    ]


def test_redis_queue_is_write_only():
    q = RedisQueue(redis_client=FakeRedis())
    q.put({"key": {}})

    assert q.full() is False
    assert q.empty() is True
    assert q.get_nowait() == []


def test_redis_queue_logs_failures_without_the_payload(caplog):
    client = FakeRedis(error=RuntimeError("redis://user:planted-secret@host"))
    q = RedisQueue(redis_client=client)

    with caplog.at_level(logging.DEBUG):
        q.put({"planted-api-key": {"request:r": {"processed_frames": 1}}})

    assert "Failed to store usage records" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "planted-api-key" not in caplog.text
    assert "planted-secret" not in caplog.text


def test_redis_queue_logs_partial_insert(caplog):
    q = RedisQueue(redis_client=FakeRedis(results=(True, 0)))

    with caplog.at_level(logging.DEBUG):
        q.put({"planted-api-key": {}})

    assert "partial insert" in caplog.text
    assert "planted-api-key" not in caplog.text


def test_redis_queue_without_a_client_requires_the_redis_package(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "redis", None)

    with pytest.raises(ImportError):
        RedisQueue()


def test_sqlite_put_reports_whether_the_payload_was_stored(tmp_path, monkeypatch):
    db_file = tmp_path / "usage.db"
    q = SQLiteQueue(db_file_path=db_file)
    monkeypatch.setattr(queues, "SQLITE_TIMEOUT_S", 0.05)

    stored = q.put({"index": 1})
    not_serialisable = q.put({"index": object()})
    holder = sqlite3.connect(str(db_file), timeout=0.05, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        while_locked = q.put({"index": 2})
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    after_unlock = q.put({"index": 3})

    assert (stored, not_serialisable, while_locked, after_unlock) == (
        True,
        False,
        False,
        True,
    )
    assert q.get_nowait() == [{"index": 1}, {"index": 3}]


def test_redis_put_reports_whether_the_payload_was_stored():
    client = FakeRedis()
    q = RedisQueue(redis_client=client)

    stored = q.put({"key": {}})
    not_serialisable = q.put({"key": object()})
    client.error = RuntimeError("unreachable")
    pipeline_raised = q.put({"key": {}})
    client.error = None
    client.results = [True, 0]
    partial_insert = q.put({"key": {}})

    assert (stored, not_serialisable, pipeline_raised, partial_insert) == (
        True,
        UNCONFIRMED,
        UNCONFIRMED,
        UNCONFIRMED,
    )


def test_redis_put_with_a_pipeline_that_raises_after_applying_reports_unconfirmed(
    caplog,
):
    client = StoringRedis(raise_after_applying=True)
    q = RedisQueue(redis_client=client)

    with caplog.at_level(logging.DEBUG):
        outcome = q.put({"planted-api-key": {}})

    assert outcome is UNCONFIRMED
    assert len(client.store) == 1
    assert "planted" not in caplog.text


def test_redis_put_with_a_non_acknowledged_result_reports_unconfirmed_after_one_attempt():
    client = StoringRedis(zadd_results=[0])
    q = RedisQueue(redis_client=client)

    outcome = q.put({"api-key": {"request:r": {"processed_frames": 1}}})

    assert outcome is UNCONFIRMED
    assert client.attempts == 1


def test_redis_consecutive_payloads_get_distinct_keys_with_the_legacy_prefix():
    client = StoringRedis()
    q = RedisQueue(redis_client=client)

    q.put({"first": {}})
    q.put({"second": {}})

    assert list(client.store) == [f"{q._prefix}:1", f"{q._prefix}:2"]
    assert list(client.members) == list(client.store)
    assert q._prefix.startswith("{UsageCollector}:")


class CloseFailingConnection:
    def __init__(self, connection):
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def close(self):
        self._connection.close()
        raise sqlite3.OperationalError("planted-secret")


def test_sqlite_put_reports_stored_when_closing_the_connection_fails_after_commit(
    tmp_path, monkeypatch, caplog
):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        queues.sqlite3,
        "connect",
        lambda *args, **kwargs: CloseFailingConnection(real_connect(*args, **kwargs)),
    )

    with caplog.at_level(logging.DEBUG):
        stored = q.put({"index": 1})
    monkeypatch.setattr(queues.sqlite3, "connect", real_connect)

    assert stored is True
    assert q.get_nowait() == [{"index": 1}]
    assert "Failed to close the usage database" in caplog.text
    assert "planted-secret" not in caplog.text


def stored_rows(db_file):
    connection = sqlite3.connect(str(db_file))
    try:
        rows = connection.execute(
            "SELECT id, payload FROM usage ORDER BY id"
        ).fetchall()
    finally:
        connection.close()

    return rows


def rows_of(api_key_hash, *, frames=1, timestamp_stop=None):
    row = {"api_key_hash": api_key_hash, "processed_frames": frames}
    if timestamp_stop is not None:
        row["timestamp_stop"] = timestamp_stop
    payload = {api_key_hash: {f"request:{api_key_hash}": row}}

    return payload


def test_sqlite_take_returns_only_payloads_of_known_hashes_and_leaves_the_rest_untouched(
    tmp_path,
):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    q.put(rows_of("a", frames=1))
    q.put(rows_of("b", frames=2))
    q.put(rows_of("a", frames=3))
    before = stored_rows(tmp_path / "usage.db")

    taken = q.take({"a"})

    assert taken == [rows_of("a", frames=1), rows_of("a", frames=3)]
    assert stored_rows(tmp_path / "usage.db") == [before[1]]
    assert q.take(set()) == []
    assert stored_rows(tmp_path / "usage.db") == [before[1]]
    assert q.take({"b"}) == [rows_of("b", frames=2)]
    assert q.empty() is True


def test_sqlite_take_splits_a_payload_of_several_hashes_in_place(tmp_path):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    q.put({**rows_of("a", frames=1), **rows_of("b", frames=2)})
    ((row_id, _),) = stored_rows(tmp_path / "usage.db")

    taken = q.take({"a"})

    assert taken == [rows_of("a", frames=1)]
    assert stored_rows(tmp_path / "usage.db") == [
        (row_id, json.dumps(rows_of("b", frames=2)))
    ]


def test_sqlite_take_reaches_known_payloads_behind_unknown_ones(tmp_path):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    for index in range(250):
        q.put(rows_of("unknown", frames=index))
    for index in range(5):
        q.put(rows_of("known", frames=index))

    taken = q.take({"known"})

    assert [
        payload["known"]["request:known"]["processed_frames"] for payload in taken
    ] == [0, 1, 2, 3, 4]
    assert q.qsize() == 250
    assert q.take({"known"}) == []


def test_sqlite_take_honours_the_batch_limit_over_the_known_payloads(tmp_path):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    for index in range(230):
        q.put(rows_of("unknown" if index % 2 else "known", frames=index))

    first = q.take({"known"})
    second = q.take({"known"})

    assert len(first) == 100
    assert len(second) == 15
    assert q.qsize() == 115


def test_get_nowait_takes_every_payload(tmp_path):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    q.put(rows_of("a"))
    q.put(rows_of("b"))

    assert q.get_nowait() == [rows_of("a"), rows_of("b")]
    assert q.empty() is True


def test_sqlite_take_deletes_unknown_payloads_older_than_the_age_cap(tmp_path, caplog):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    now_ns = queues.time.time_ns()
    stale_ns = now_ns - (queues.SQLITE_MAX_PAYLOAD_AGE_S + 60) * 1_000_000_000
    fresh_ns = now_ns - 60 * 1_000_000_000
    q.put(rows_of("gone", frames=1, timestamp_stop=stale_ns))
    q.put(rows_of("gone", frames=2, timestamp_stop=stale_ns))
    q.put(rows_of("gone", frames=3, timestamp_stop=fresh_ns))
    q.put(rows_of("ageless", frames=4))
    q.put(rows_of("known", frames=5, timestamp_stop=stale_ns))

    with caplog.at_level(logging.DEBUG):
        taken = q.take({"known"})

    assert taken == [rows_of("known", frames=5, timestamp_stop=stale_ns)]
    assert [
        json.loads(payload) for _, payload in stored_rows(tmp_path / "usage.db")
    ] == [
        rows_of("gone", frames=3, timestamp_stop=fresh_ns),
        rows_of("ageless", frames=4),
    ]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "2" in warnings[0].getMessage()
    assert "30" in warnings[0].getMessage()
    assert q.take({"known"}) == []
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_sqlite_take_keeps_a_payload_holding_an_ageless_row(tmp_path):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    stale_ns = (
        queues.time.time_ns() - (queues.SQLITE_MAX_PAYLOAD_AGE_S + 60) * 1_000_000_000
    )
    mixed = {
        **rows_of("old", frames=1, timestamp_stop=stale_ns),
        **rows_of("ageless", frames=2),
    }
    q.put(mixed)

    assert q.take(set()) == []

    assert [
        json.loads(payload) for _, payload in stored_rows(tmp_path / "usage.db")
    ] == [mixed]


def test_sqlite_take_prunes_a_payload_whose_rows_are_all_old(tmp_path):
    q = SQLiteQueue(db_file_path=tmp_path / "usage.db")
    stale_ns = (
        queues.time.time_ns() - (queues.SQLITE_MAX_PAYLOAD_AGE_S + 60) * 1_000_000_000
    )
    q.put(
        {
            **rows_of("a", timestamp_stop=stale_ns),
            **rows_of("b", timestamp_stop=stale_ns),
        }
    )

    assert q.take(set()) == []

    assert stored_rows(tmp_path / "usage.db") == []


def test_sqlite_take_drops_an_undecodable_payload(tmp_path, caplog):
    db_file = tmp_path / "usage.db"
    q = SQLiteQueue(db_file_path=db_file)
    connection = sqlite3.connect(str(db_file))
    connection.execute("INSERT INTO usage (payload) VALUES ('not json')")
    connection.commit()
    connection.close()
    q.put(rows_of("a"))

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage.queues"):
        taken = q.take({"a"})

    assert taken == [rows_of("a")]
    assert q.empty() is True
    assert "Failed to process a stored usage payload" in caplog.text
    assert "not json" not in caplog.text


def test_memory_queue_take_returns_known_parts_and_keeps_the_rest():
    q = queues.MemoryQueue(maxsize=10)
    q.put(rows_of("a", frames=1))
    q.put(rows_of("b", frames=2))
    q.put({**rows_of("a", frames=3), **rows_of("b", frames=4)})

    taken = q.take({"a"})

    assert taken == [rows_of("a", frames=1), rows_of("a", frames=3)]
    assert list(q.queue) == [rows_of("b", frames=2), rows_of("b", frames=4)]
    assert q.take(set()) == []
    assert q.qsize() == 2
    assert q.take({"b"}) == [rows_of("b", frames=2), rows_of("b", frames=4)]
    assert q.empty() is True
