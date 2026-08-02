"""Usage tracking must not change what the wrapped call returns or raises.

`UsageCollector.__call__` wrapped both the decorated function and `record_usage` in a
single `try`. If recording raised, the `except` branch attributed the failure to the
inference call, recorded it a second time as an error, and re-raised, so a successful
inference was reported as failed and the caller's request failed with an exception
raised by billing code.

`_update_usage_payload` divides by `fps` guarded only by truthiness while the very next
line checks `isinstance(fps, numbers.Number)`, and it calls `json.dumps` on
caller-supplied `resource_details`, so recording has real ways to raise on input the
inference itself handled fine.
"""

import asyncio
from unittest.mock import patch

import pytest

from inference.usage_tracking.collector import UsageCollector


class _Boom(Exception):
    """Raised by the wrapped function, so it must reach the caller unchanged."""


def _collector() -> UsageCollector:
    collector = UsageCollector()
    collector._cleanup()
    return collector


class TestUsageFailureDoesNotBreakTheCall:
    def test_successful_call_still_returns_when_recording_raises(self):
        collector = _collector()

        @collector("model")
        def infer():
            return "predictions"

        with patch.object(
            UsageCollector, "record_usage", side_effect=RuntimeError("usage exploded")
        ):
            assert infer() == "predictions"

    def test_recording_failure_is_not_reported_as_an_inference_error(self):
        """The old code called record_usage a second time with error_details set."""
        collector = _collector()

        @collector("model")
        def infer():
            return "predictions"

        calls = []

        def _record(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("usage exploded")

        with patch.object(UsageCollector, "record_usage", side_effect=_record):
            infer()

        assert len(calls) == 1, "a failed recording must not be retried as an error"
        # error_details is merged into resource_details by
        # _extract_usage_params_from_func_kwargs rather than passed through.
        assert "error" not in (calls[0].get("resource_details") or {})

    def test_the_wrapped_functions_own_exception_still_propagates(self):
        collector = _collector()

        @collector("model")
        def infer():
            raise _Boom("model failed")

        with patch.object(UsageCollector, "record_usage"):
            with pytest.raises(_Boom):
                infer()

    def test_a_real_failure_is_still_recorded_with_error_details(self):
        collector = _collector()

        @collector("model")
        def infer():
            raise _Boom("model failed")

        calls = []
        with patch.object(
            UsageCollector, "record_usage", side_effect=lambda **kw: calls.append(kw)
        ):
            with pytest.raises(_Boom):
                infer()

        assert len(calls) == 1
        assert "error" in (
            calls[0].get("resource_details") or {}
        ), "a genuine failure must still be recorded as an error"

    def test_recording_failure_does_not_mask_the_functions_own_exception(self):
        """Both fail: the caller must still see the model's error, not the usage one."""
        collector = _collector()

        @collector("model")
        def infer():
            raise _Boom("model failed")

        with patch.object(
            UsageCollector, "record_usage", side_effect=RuntimeError("usage exploded")
        ):
            with pytest.raises(_Boom):
                infer()


class TestAsyncWrapperBehavesTheSame:
    def test_successful_async_call_still_returns_when_recording_raises(self):
        collector = _collector()

        @collector("model")
        async def infer():
            return "predictions"

        async def _record(**kwargs):
            raise RuntimeError("usage exploded")

        with patch.object(UsageCollector, "async_record_usage", side_effect=_record):
            assert asyncio.run(infer()) == "predictions"

    def test_async_wrapped_exception_still_propagates(self):
        collector = _collector()

        @collector("model")
        async def infer():
            raise _Boom("model failed")

        async def _record(**kwargs):
            raise RuntimeError("usage exploded")

        with patch.object(UsageCollector, "async_record_usage", side_effect=_record):
            with pytest.raises(_Boom):
                asyncio.run(infer())
