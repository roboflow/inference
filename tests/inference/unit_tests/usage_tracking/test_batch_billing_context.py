import asyncio
import json
from types import SimpleNamespace

import pytest

from inference.usage_tracking.decorator_helpers import (
    read_batch_billing_context,
    request_billing_context_query,
    usage_batch_billing_context,
)
from inference.usage_tracking.payload_helpers import zip_usage_payloads


def _handler(request, countinference=True, service_secret=None):
    pass


def _request(media="image", multiplier=0.1, version="batch-v1"):
    return SimpleNamespace(
        query_params={
            "billing_context": json.dumps(
                {"version": version, "media": media, "multiplier": multiplier}
            )
        }
    )


def test_only_authenticated_context_can_select_batch_rate(configured_service_secret):
    request = _request()
    assert read_batch_billing_context(_handler, (), {"request": request}) is None
    assert (
        read_batch_billing_context(
            _handler, (), {"request": request, "service_secret": "wrong"}
        )
        is None
    )
    context = read_batch_billing_context(
        _handler, (), {"request": request, "service_secret": configured_service_secret}
    )
    assert json.loads(context) == {
        "version": "batch-v1",
        "media": "image",
        "multiplier": 0.1,
    }


def _job_request(job_id):
    request = _request()
    context = json.loads(request.query_params["billing_context"])
    request.query_params["billing_context"] = json.dumps({**context, "jobId": job_id})

    return request


def test_authenticated_context_keeps_the_job_it_names(configured_service_secret):
    context = read_batch_billing_context(
        _handler,
        (),
        {
            "request": _job_request("job-nov-318dd5c75703"),
            "service_secret": configured_service_secret,
        },
    )

    assert json.loads(context) == {
        "version": "batch-v1",
        "media": "image",
        "multiplier": 0.1,
        "jobId": "job-nov-318dd5c75703",
    }


@pytest.mark.parametrize("job_id", ["", "job/1", "a" * 129, 7, ["job"]])
def test_malformed_job_ids_fail_before_execution(configured_service_secret, job_id):
    with pytest.raises(ValueError, match="Invalid authenticated"):
        read_batch_billing_context(
            _handler,
            (),
            {
                "request": _job_request(job_id),
                "service_secret": configured_service_secret,
            },
        )


@pytest.mark.parametrize(
    "multiplier", [0, -1, 1.1, float("nan"), float("inf"), 10**400, True, "0.1"]
)
def test_invalid_authenticated_rates_fail_before_execution(
    configured_service_secret, multiplier
):
    with pytest.raises(ValueError, match="Invalid authenticated"):
        read_batch_billing_context(
            _handler,
            (),
            {
                "request": _request(multiplier=multiplier),
                "service_secret": configured_service_secret,
            },
        )


def test_batch_and_regular_usage_never_merge(usage_collector_with_mocked_threads):
    collector = usage_collector_with_mocked_threads
    contexts = [
        None,
        json.dumps({"version": "batch-v1", "media": "image", "multiplier": 0.1}),
        json.dumps({"version": "batch-v1", "media": "video", "multiplier": 0.1}),
        json.dumps(
            {
                "version": "batch-v1",
                "media": "image",
                "multiplier": 0.5,
                "jobId": "job-a",
            }
        ),
        json.dumps(
            {
                "version": "batch-v1",
                "media": "image",
                "multiplier": 0.5,
                "jobId": "job-b",
            }
        ),
    ]
    keys = []
    cache_keys = []
    payloads = []
    for context in contexts:
        details = {"billable": True}
        if context:
            details["billing_context"] = context
        key = collector._usage_key("model", "model/1", details)
        keys.append(key)
        cache_keys.append(collector._resource_details_key("model", "model/1", details))
        payloads.append(
            {
                "key": {
                    key: {
                        "processed_frames": 1,
                        "resource_details": json.dumps(details),
                    }
                }
            }
        )
    assert len(set(keys)) == len(contexts)
    # Per-call contexts partition usage, but share persistent resource metadata.
    assert len(set(cache_keys)) == 1
    merged = zip_usage_payloads(payloads)
    assert sum(len(payload["key"]) for payload in merged) == len(contexts)


@pytest.mark.parametrize("resource_id", ["model/1", ""])
def test_completed_batch_jobs_do_not_accumulate_cached_metadata(
    usage_collector_with_mocked_threads, monkeypatch, resource_id
):
    collector = usage_collector_with_mocked_threads
    flushed = []
    monkeypatch.setattr(
        collector, "_enqueue_payload", lambda payload: flushed.append(payload)
    )
    collector.record_system_info()
    api_key = "batch-cache-key"
    expected_contexts = []

    for index in range(100):
        context = json.dumps(
            {
                "version": "batch-v1",
                "media": "image",
                "multiplier": 0.5,
                "jobId": f"job-{index}",
            },
            sort_keys=True,
        )
        expected_contexts.append(context)
        collector.record_usage(
            source="",
            category="model",
            api_key=api_key,
            resource_id=resource_id,
            resource_details={
                "billable": True,
                "model_architecture": "yolov8",
                "billing_context": context,
            },
        )
        collector._enqueue_usage_payload()

    assert not collector._usage
    assert list(collector._resource_details[api_key].values()) == [
        {"billable": True, "model_architecture": "yolov8"}
    ]
    api_key_hash = collector._calculate_api_key_hash(api_key=api_key)
    rows = [row for payload in flushed for row in payload[api_key_hash].values()]
    assert len(rows) == 100
    assert len({row["resource_id"] for row in rows}) == 1
    assert [
        json.loads(row["resource_details"])["billing_context"] for row in rows
    ] == expected_contexts
    assert all(row["processed_frames"] == 1 for row in rows)


@pytest.mark.parametrize("next_job_id", [None, "job-b"])
def test_cached_metadata_does_not_supply_another_calls_batch_context(
    usage_collector_with_mocked_threads, next_job_id
):
    collector = usage_collector_with_mocked_threads
    collector.record_system_info()
    api_key = "batch-cache-key"
    context = {
        "version": "batch-v1",
        "media": "image",
        "multiplier": 0.5,
        "jobId": "job-a",
    }
    original_details = {
        "billable": True,
        "model_architecture": "yolov8",
        "billing_context": json.dumps(context),
    }
    collector.record_resource_details(
        category="model",
        api_key=api_key,
        resource_id="model/1",
        resource_details=original_details,
    )
    # Read the cache directly without first replacing it through record_usage.
    details = {"billable": True}
    if next_job_id is not None:
        details["billing_context"] = json.dumps({**context, "jobId": next_job_id})
    collector._update_usage_payload(
        source="",
        category="model",
        api_key=api_key,
        resource_id="model/1",
        resource_details=details,
    )

    api_key_hash = collector._calculate_api_key_hash(api_key=api_key)
    (row,) = collector._usage[api_key_hash].values()
    assert json.loads(row["resource_details"]) == {
        "model_architecture": "yolov8",
        **details,
    }
    assert original_details["billing_context"] == json.dumps(context)


def test_authenticated_context_is_recorded_on_request_and_model_rows(
    usage_collector_with_mocked_threads, configured_service_secret
):
    collector = usage_collector_with_mocked_threads

    @collector(category="model")
    def model_call(api_key):
        pass

    @collector(category="request")
    def handler(request, api_key, countinference=True, service_secret=None):
        model_call(api_key=api_key)

    handler(request=_request(), api_key="key", service_secret=configured_service_secret)
    handler(request=_request(), api_key="key")

    recorded = {
        (row["category"], json.loads(row["resource_details"]).get("billing_context"))
        for row in collector._usage["key"].values()
    }
    batch = json.dumps(
        {"media": "image", "multiplier": 0.1, "version": "batch-v1"}, sort_keys=True
    )
    # The platform reads the rate from these rows; regular rows stay separate.
    assert recorded == {
        ("request", batch),
        ("model", batch),
        ("request", None),
        ("model", None),
    }


def test_context_reaches_nested_usage_and_resets(
    usage_collector_with_mocked_threads, configured_service_secret
):
    collector = usage_collector_with_mocked_threads
    seen = []

    @collector(category="model")
    def nested():
        seen.append(usage_batch_billing_context.get())

    @collector(category="request")
    def handler(request, countinference=True, service_secret=None):
        nested()

    handler(request=_request(), service_secret=configured_service_secret)
    assert json.loads(seen[0])["multiplier"] == 0.1
    assert usage_batch_billing_context.get() is None
    handler(request=_request())
    assert seen[1] is None


INVALID_CONTEXT_REQUESTS = [
    {"media": "audio"},
    {"version": "batch-v2"},
    {"multiplier": 0},
]


@pytest.mark.parametrize("invalid", INVALID_CONTEXT_REQUESTS)
def test_invalid_authenticated_context_never_runs_the_sync_handler(
    usage_collector_with_mocked_threads, configured_service_secret, invalid
):
    collector = usage_collector_with_mocked_threads
    calls = []

    @collector(category="request")
    def handler(request, api_key, countinference=True, service_secret=None):
        calls.append(api_key)

    with pytest.raises(ValueError):
        handler(
            request=_request(**invalid),
            api_key="key",
            service_secret=configured_service_secret,
        )

    assert calls == []
    assert "key" not in collector._usage

    handler(request=_request(), api_key="key", service_secret=configured_service_secret)

    assert calls == ["key"]
    assert collector._usage["key"]


@pytest.mark.parametrize("invalid", INVALID_CONTEXT_REQUESTS)
def test_invalid_authenticated_context_never_runs_the_async_handler(
    usage_collector_with_mocked_threads, configured_service_secret, invalid
):
    collector = usage_collector_with_mocked_threads
    calls = []

    @collector(category="request")
    async def handler(request, api_key, countinference=True, service_secret=None):
        calls.append(api_key)

    with pytest.raises(ValueError):
        asyncio.run(
            handler(
                request=_request(**invalid),
                api_key="key",
                service_secret=configured_service_secret,
            )
        )

    assert calls == []
    assert "key" not in collector._usage

    asyncio.run(
        handler(
            request=_request(),
            api_key="key",
            service_secret=configured_service_secret,
        )
    )

    assert calls == ["key"]
    assert collector._usage["key"]


def _handler_without_request(countinference=True, service_secret=None):
    pass


def test_handler_without_request_reads_middleware_captured_context(
    configured_service_secret,
):
    raw = _request().query_params["billing_context"]
    token = request_billing_context_query.set(raw)
    try:
        assert read_batch_billing_context(_handler_without_request, (), {}) is None
        context = read_batch_billing_context(
            _handler_without_request,
            (),
            {"service_secret": configured_service_secret},
        )
    finally:
        request_billing_context_query.reset(token)

    assert json.loads(context) == {
        "version": "batch-v1",
        "media": "image",
        "multiplier": 0.1,
    }


def test_handler_without_request_rejects_malformed_captured_context(
    configured_service_secret,
):
    token = request_billing_context_query.set("{not-json")
    try:
        with pytest.raises(ValueError):
            read_batch_billing_context(
                _handler_without_request,
                (),
                {"service_secret": configured_service_secret},
            )
    finally:
        request_billing_context_query.reset(token)
