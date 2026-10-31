import json
from types import SimpleNamespace

import pytest

from inference.usage_tracking.decorator_helpers import (
    read_batch_billing_context,
    usage_batch_billing_context,
)
from inference.usage_tracking.payload_helpers import zip_usage_payloads


def _handler(request, countinference=True, service_secret=None):
    pass


def _request(media="image", multiplier=0.1):
    return SimpleNamespace(
        query_params={
            "billing_context": json.dumps(
                {"version": "batch-v1", "media": media, "multiplier": multiplier}
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


@pytest.mark.parametrize("multiplier", [0, -1, 1.1, float("nan"), True, "0.1"])
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
    assert len(set(keys)) == 3
    assert len(set(cache_keys)) == 3
    merged = zip_usage_payloads(payloads)
    assert sum(len(payload["key"]) for payload in merged) == 3


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
