from copy import deepcopy
from types import SimpleNamespace

import pytest

from inference.usage_tracking import payload_helpers as helpers

CAPABILITY = {
    "version": 1,
    "workspace_id": "workspace",
    "ownership_fingerprint": "a" * 64,
}


def raw_usage(frames=10):
    return {
        "resource": {
            "_usage_report_candidate": True,
            "resource_id": "resource",
            "category": "model",
            "processed_frames": frames,
            "timestamp_start": 1000000000000000000,
            "timestamp_stop": 1000000010000000000,
            "execution_duration": 10,
            "exec_session_id": "process",
            "stream_session_id": "stream",
            "api_key_hash": "hash",
            "source_duration": 0,
            "resource_details": '{"source":"test"}',
        }
    }


def test_prepare_and_timeout_retry_preserve_original_report(monkeypatch):
    raw = raw_usage()
    original = deepcopy(raw)
    reports = helpers.prepare_usage_reports(raw, CAPABILITY)
    report_id = next(iter(reports))
    observed = {}
    calls = []

    def send(*args, json, **kwargs):
        calls.append(deepcopy(json))
        for report in json:
            observed.setdefault(report["report_id"], report["processed_frames"])
        if len(calls) == 1:
            raise TimeoutError("accepted, response lost")
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "usage_report_results": [
                    {"report_id": report["report_id"], "status": "accepted"}
                    for report in json
                ]
            },
        )

    monkeypatch.setattr(helpers.requests, "post", send)
    assert (
        helpers.send_usage_reports(reports, "api-key", "https://example.invalid") == {}
    )
    newer = helpers.prepare_usage_reports(raw_usage(5), CAPABILITY)
    assert helpers.send_usage_reports(
        reports, "api-key", "https://example.invalid"
    ) == {report_id: "accepted"}
    helpers.send_usage_reports(newer, "api-key", "https://example.invalid")
    assert sum(observed.values()) == 15
    assert calls[0] == calls[1]
    assert raw == original
    assert "api_key" not in reports[report_id]
    assert reports[report_id]["exec_session_id"] == "stream"


def test_partial_report_outcomes_only_retire_terminal_ids():
    result = helpers.usage_report_outcomes(
        {
            "usage_report_results": [
                {"report_id": "one", "status": "accepted"},
                {
                    "report_id": "two",
                    "status": "retryable",
                    "reason": "ownership_conflict",
                },
                {"report_id": "three", "status": "rejected", "reason": "invalid_input"},
            ]
        },
        {"one", "two", "three", "four"},
    )
    assert result == {"one": "accepted", "three": "invalid_input"}


@pytest.mark.parametrize(
    "body",
    [
        None,
        {"status": "OK"},
        {"usage_report_results": None},
        {"usage_report_results": [{"report_id": "other", "status": "accepted"}]},
        {"usage_report_results": [{"report_id": "one", "status": "accepted"}] * 2},
        {"usage_report_results": [{"report_id": "one", "status": "rejected"}]},
    ],
)
def test_missing_or_malformed_acknowledgements_retain_reports(body):
    assert helpers.usage_report_outcomes(body, {"one"}) == {}


@pytest.mark.parametrize("capability", [None, CAPABILITY])
def test_successful_capability_negotiation(monkeypatch, capability):
    monkeypatch.setattr(
        helpers.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"usage_report_protocol": capability},
        ),
    )
    assert (
        helpers.get_usage_report_capability("key", "https://example.invalid")
        == capability
    )


def test_failed_negotiation_does_not_imply_legacy(monkeypatch):
    def fail(*a, **k):
        raise TimeoutError("unavailable")

    monkeypatch.setattr(helpers.requests, "get", fail)
    with pytest.raises(TimeoutError):
        helpers.get_usage_report_capability("key", "https://example.invalid")


def test_legacy_transport_does_not_emit_attempt_marker_or_mutate_retry(monkeypatch):
    raw = raw_usage()
    raw["resource"]["_legacy_delivery"] = True
    original = deepcopy(raw)
    calls = []
    monkeypatch.setattr(
        helpers.requests,
        "post",
        lambda *a, json, **k: calls.append(json) or SimpleNamespace(status_code=200),
    )
    assert helpers.send_usage_payload({"key": raw}, "https://example.invalid") == set()
    assert "_legacy_delivery" not in calls[0][0]
    assert raw == original


def configure_collector(collector, monkeypatch, persistent_queue=None):
    from queue import Queue
    from inference.usage_tracking import collector as module

    collector._queue = persistent_queue or Queue(maxsize=1)
    collector._report_delivery = {}
    collector._hashed_api_keys = {"api-key": "hash"}
    collector._settings = SimpleNamespace(
        api_usage_endpoint_url="https://example.invalid/usage/inference",
        api_plan_endpoint_url="https://example.invalid/usage/plan",
    )
    collector._plan_details = SimpleNamespace(
        _is_enterprise_col_name="enterprise",
        get_api_key_plan=lambda **kw: {"enterprise": False},
    )
    monkeypatch.setattr(module, "OFFLINE_MODE", False)
    monkeypatch.setattr(helpers, "OFFLINE_MODE", False)
    monkeypatch.setattr(
        module, "get_usage_report_capability", lambda *a, **kw: CAPABILITY
    )
    return module


def test_actual_collector_timeout_new_usage_and_queue_compaction(
    usage_collector_with_mocked_threads, monkeypatch
):
    collector = usage_collector_with_mocked_threads
    configure_collector(collector, monkeypatch)
    received, bodies = {}, []

    def send(*a, json, **kw):
        bodies.append(deepcopy(json))
        for report in json:
            received.setdefault(report["report_id"], report["processed_frames"])
        if len(bodies) == 1:
            raise TimeoutError("accepted, response lost")
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "usage_report_results": [
                    {"report_id": report["report_id"], "status": "accepted"}
                    for report in json
                ]
            },
        )

    monkeypatch.setattr(helpers.requests, "post", send)
    collector._enqueue_payload({"hash": raw_usage(10)})
    collector._flush_queue()
    retained = deepcopy(collector._report_delivery)
    collector._enqueue_payload({"hash": raw_usage(2)})
    collector._enqueue_payload({"hash": raw_usage(3)})
    assert collector._report_delivery == retained
    assert collector._queue.qsize() == 1
    collector._flush_queue()
    collector._flush_queue()
    assert bodies[0] == bodies[1]
    assert sum(received.values()) == 15
    assert collector._report_delivery == {}


def test_actual_collector_legacy_attempt_cannot_upgrade(
    usage_collector_with_mocked_threads, monkeypatch
):
    collector = usage_collector_with_mocked_threads
    module = configure_collector(collector, monkeypatch)
    monkeypatch.setattr(module, "get_usage_report_capability", lambda *a, **kw: None)
    sent = []

    def send(*a, json, **kw):
        sent.append(deepcopy(json))
        if len(sent) == 1:
            raise TimeoutError("legacy accepted, response lost")
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(helpers.requests, "post", send)
    collector._enqueue_payload({"hash": raw_usage(10)})
    collector._flush_queue()
    monkeypatch.setattr(
        module, "get_usage_report_capability", lambda *a, **kw: CAPABILITY
    )
    collector._enqueue_payload({"hash": raw_usage(5)})
    collector._flush_queue()
    assert [report[0]["processed_frames"] for report in sent] == [10, 15]
    assert all("report_version" not in report for batch in sent for report in batch)
    assert all("_legacy_delivery" not in report for batch in sent for report in batch)


def test_sqlite_prepared_report_survives_restart_and_customer_change(
    usage_collector_with_mocked_threads, monkeypatch, tmp_path
):
    from inference.usage_tracking.sqlite_queue import SQLiteQueue

    collector = usage_collector_with_mocked_threads
    queue = SQLiteQueue(db_file_path=str(tmp_path / "usage.db"))
    module = configure_collector(collector, monkeypatch, queue)
    sent = []

    def send(*a, json, **kw):
        sent.append(deepcopy(json))
        if len(sent) == 1:
            raise TimeoutError("accepted, response lost")
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "usage_report_results": [
                    {"report_id": report["report_id"], "status": "accepted"}
                    for report in json
                ]
            },
        )

    monkeypatch.setattr(helpers.requests, "post", send)
    collector._enqueue_payload({"hash": raw_usage()})
    collector._flush_queue()
    before = queue.read_report_delivery()
    assert before
    collector._queue = SQLiteQueue(db_file_path=str(tmp_path / "usage.db"))

    def should_not_renegotiate(*a, **kw):
        raise AssertionError("Attempted report renegotiated ownership")

    monkeypatch.setattr(module, "get_usage_report_capability", should_not_renegotiate)
    collector._flush_queue()
    assert sent[0] == sent[1]
    assert collector._queue.read_report_delivery() == {}
    assert collector._queue.empty()


@pytest.mark.parametrize("persistent", [False, True])
def test_collector_conflicted_owner_does_not_block_new_owner(
    usage_collector_with_mocked_threads, monkeypatch, tmp_path, persistent
):
    from inference.usage_tracking.sqlite_queue import SQLiteQueue

    collector = usage_collector_with_mocked_threads
    queue = (
        SQLiteQueue(db_file_path=str(tmp_path / "fairness.db")) if persistent else None
    )
    module = configure_collector(collector, monkeypatch, queue)
    collector._hashed_api_keys["healthy-key"] = "healthy-hash"
    accepted, conflicted = {}, []

    def send(*a, json, **kw):
        outcomes = []
        for report in json:
            if report["report_ownership_fingerprint"] == "a" * 64:
                conflicted.append(deepcopy(report))
                outcomes.append(
                    {
                        "report_id": report["report_id"],
                        "status": "retryable",
                        "reason": "ownership_conflict",
                    }
                )
            else:
                accepted.setdefault(report["report_id"], report["processed_frames"])
                outcomes.append(
                    {"report_id": report["report_id"], "status": "accepted"}
                )
        return SimpleNamespace(
            status_code=200, json=lambda: {"usage_report_results": outcomes}
        )

    monkeypatch.setattr(helpers.requests, "post", send)
    collector._enqueue_payload({"hash": raw_usage(10)})
    collector._flush_queue()
    original = deepcopy(conflicted[0])
    monkeypatch.setattr(
        module,
        "get_usage_report_capability",
        lambda key, *a, **kw: (
            CAPABILITY
            if key == "api-key"
            else dict(CAPABILITY, ownership_fingerprint="b" * 64)
        ),
    )
    for _ in range(3):
        collector._enqueue_payload({"hash": raw_usage(1), "healthy-hash": raw_usage(2)})
        collector._flush_queue()
    retained = (
        queue.read_report_delivery() if persistent else collector._report_delivery
    )
    assert list(retained) == ["hash"]
    assert len(retained["hash"]) == 1
    assert all(report == original for report in conflicted)
    assert sum(accepted.values()) == 6
    # A new ownership basis may prepare new usage while the old report stays frozen.
    monkeypatch.setattr(
        module,
        "get_usage_report_capability",
        lambda *a, **kw: dict(CAPABILITY, ownership_fingerprint="c" * 64),
    )
    collector._flush_queue()
    assert sum(accepted.values()) == 9
    retained = (
        queue.read_report_delivery() if persistent else collector._report_delivery
    )
    assert len(retained["hash"]) == 1


def test_unmarked_legacy_usage_keeps_compacted_new_usage_legacy():
    legacy = raw_usage(10)
    legacy["resource"].pop("_usage_report_candidate")
    merged = helpers.zip_usage_payloads([{"key": legacy}, {"key": raw_usage(5)}])
    assert merged[0]["key"]["resource"]["processed_frames"] == 15
    assert merged[0]["key"]["resource"]["_usage_report_candidate"] is False


def test_sqlite_failed_prepare_rolls_back_raw_deletion(tmp_path):
    import sqlite3
    from inference.usage_tracking.sqlite_queue import SQLiteQueue

    queue = SQLiteQueue(db_file_path=str(tmp_path / "rollback.db"))
    payload = {"hash": raw_usage(10)}
    queue.put(payload)
    queue.read_report_delivery()
    with sqlite3.connect(queue._db_file_path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_prepare BEFORE INSERT ON usage_report_delivery BEGIN SELECT RAISE(ABORT, 'disk failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="disk failure"):
        queue.prepare_report_delivery(
            lambda payloads, retained: (
                {
                    "hash": helpers.prepare_usage_reports(
                        payloads[0]["hash"], CAPABILITY
                    )
                },
                [],
                [],
            )
        )
    assert queue.peek_payloads() == [payload]
    assert queue.read_report_delivery() == {}


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("keep_original", [False, True])
def test_supplied_rotated_credential_replays_original_workspace_report(
    usage_collector_with_mocked_threads,
    monkeypatch,
    tmp_path,
    persistent,
    keep_original,
):
    from inference.usage_tracking.sqlite_queue import SQLiteQueue

    collector = usage_collector_with_mocked_threads
    queue = (
        SQLiteQueue(db_file_path=str(tmp_path / "rotation.db")) if persistent else None
    )
    module = configure_collector(collector, monkeypatch, queue)
    sent = []
    monkeypatch.setattr(
        helpers.requests,
        "post",
        lambda *a, json, **kw: sent.append(deepcopy(json))
        or SimpleNamespace(status_code=503),
    )
    collector._enqueue_payload({"hash": raw_usage(7)})
    collector._flush_queue()
    original = deepcopy(sent[0][0])
    if not keep_original:
        collector._hashed_api_keys.clear()
    collector._hashed_api_keys.update(
        {"replacement-key": "replacement-hash", "unrelated-key": "unrelated-hash"}
    )

    def capability_for_key(key, *a, **kw):
        if key == "api-key":
            raise ConnectionError("revoked original credential")
        return dict(
            CAPABILITY,
            workspace_id="other" if key == "unrelated-key" else "workspace",
            ownership_fingerprint="b" * 64,
        )

    monkeypatch.setattr(module, "get_usage_report_capability", capability_for_key)
    accepted = {}

    def send(*a, json, **kw):
        assert kw["headers"]["Authorization"] == "Bearer replacement-key"
        for report in json:
            accepted[report["report_id"]] = deepcopy(report)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "usage_report_results": [
                    {"report_id": report["report_id"], "status": "accepted"}
                    for report in json
                ]
            },
        )

    monkeypatch.setattr(helpers.requests, "post", send)
    collector._enqueue_payload({"replacement-hash": raw_usage(3)})
    collector._flush_queue()
    replay = accepted[original["report_id"]]
    assert replay == dict(original, api_key="replacement-key")
    assert sum(report["processed_frames"] for report in accepted.values()) == 10
    assert (
        queue.read_report_delivery() if persistent else collector._report_delivery
    ) == {}


@pytest.mark.parametrize("persistent", [False, True])
def test_unrelated_replacement_credential_cannot_send_retained_report(
    usage_collector_with_mocked_threads, monkeypatch, tmp_path, persistent
):
    from inference.usage_tracking.sqlite_queue import SQLiteQueue

    collector = usage_collector_with_mocked_threads
    queue = (
        SQLiteQueue(db_file_path=str(tmp_path / "unrelated.db")) if persistent else None
    )
    module = configure_collector(collector, monkeypatch, queue)
    monkeypatch.setattr(
        helpers.requests, "post", lambda *a, **kw: SimpleNamespace(status_code=503)
    )
    collector._enqueue_payload({"hash": raw_usage(7)})
    collector._flush_queue()
    before = deepcopy(
        queue.read_report_delivery() if persistent else collector._report_delivery
    )
    collector._hashed_api_keys = {"unrelated-key": "unrelated-hash"}
    monkeypatch.setattr(
        module,
        "get_usage_report_capability",
        lambda *a, **kw: dict(CAPABILITY, workspace_id="other"),
    )
    sent = []
    monkeypatch.setattr(helpers.requests, "post", lambda *a, **kw: sent.append(kw))
    collector._flush_queue()
    assert not sent
    assert (
        queue.read_report_delivery() if persistent else collector._report_delivery
    ) == before


@pytest.mark.parametrize("persistent", [False, True])
def test_known_earlier_credential_can_replay_after_original_is_revoked(
    usage_collector_with_mocked_threads, monkeypatch, tmp_path, persistent
):
    from inference.usage_tracking.sqlite_queue import SQLiteQueue

    collector = usage_collector_with_mocked_threads
    queue = (
        SQLiteQueue(db_file_path=str(tmp_path / "earlier-key.db"))
        if persistent
        else None
    )
    module = configure_collector(collector, monkeypatch, queue)
    collector._hashed_api_keys = {
        "replacement-key": "replacement-hash",
        "api-key": "hash",
    }

    def initial_capability(key, *a, **kw):
        if key == "replacement-key":
            raise ConnectionError("replacement temporarily unavailable")
        return CAPABILITY

    monkeypatch.setattr(module, "get_usage_report_capability", initial_capability)
    sent = []
    monkeypatch.setattr(
        helpers.requests,
        "post",
        lambda *a, json, **kw: sent.append(deepcopy(json))
        or SimpleNamespace(status_code=503),
    )
    collector._enqueue_payload({"hash": raw_usage(7)})
    collector._flush_queue()
    original = deepcopy(sent[0][0])

    def rotated_capability(key, *a, **kw):
        if key == "api-key":
            raise ConnectionError("original revoked")
        return dict(CAPABILITY, ownership_fingerprint="b" * 64)

    monkeypatch.setattr(module, "get_usage_report_capability", rotated_capability)
    collector._hashed_api_keys["replacement-key"] = "replacement-hash"
    assert list(collector._hashed_api_keys) == ["replacement-key", "api-key"]
    accepted = {}

    def send(*a, json, **kw):
        if kw["headers"]["Authorization"] != "Bearer replacement-key":
            return SimpleNamespace(status_code=401)
        for report in json:
            accepted[report["report_id"]] = deepcopy(report)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "usage_report_results": [
                    {"report_id": report["report_id"], "status": "accepted"}
                    for report in json
                ]
            },
        )

    monkeypatch.setattr(helpers.requests, "post", send)
    collector._enqueue_payload({"replacement-hash": raw_usage(3)})
    collector._flush_queue()
    assert accepted[original["report_id"]] == dict(original, api_key="replacement-key")
    assert sum(report["processed_frames"] for report in accepted.values()) == 10
    assert (
        queue.read_report_delivery() if persistent else collector._report_delivery
    ) == {}
