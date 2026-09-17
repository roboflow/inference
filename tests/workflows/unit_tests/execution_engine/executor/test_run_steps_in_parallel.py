from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

import pytest

from inference.core.workflows.execution_engine.v1.executor.utils import (
    run_steps_in_parallel,
)

# A test-local ContextVar unrelated to billing: the point of these tests is
# that run_steps_in_parallel() propagates any ContextVar generically, not that
# it special-cases a hand-picked list of them.
_test_ctx_var: ContextVar[bool] = ContextVar("test_ctx_var", default=False)


def test_run_steps_in_parallel_propagates_context_var_with_owned_executor() -> None:
    # given - no executor passed in, run_steps_in_parallel creates its own
    token = _test_ctx_var.set(True)

    def read_ctx_var() -> bool:
        return _test_ctx_var.get()

    try:
        # when
        results = run_steps_in_parallel(
            steps=[read_ctx_var, read_ctx_var], max_workers=2
        )
    finally:
        _test_ctx_var.reset(token)

    # then
    assert results == [True, True]


def test_run_steps_in_parallel_propagates_context_var_with_external_executor() -> None:
    # given - caller supplies its own ThreadPoolExecutor
    token = _test_ctx_var.set(True)

    def read_ctx_var() -> bool:
        return _test_ctx_var.get()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            # when
            results = run_steps_in_parallel(
                steps=[read_ctx_var, read_ctx_var],
                max_workers=2,
                executor=executor,
            )
    finally:
        _test_ctx_var.reset(token)

    # then
    assert results == [True, True]


def test_run_steps_in_parallel_does_not_leak_across_executor_reuse() -> None:
    # given - a single-worker executor so the same worker thread is reused
    # across both calls below
    with ThreadPoolExecutor(max_workers=1) as executor:

        def suppressing_task_that_raises() -> None:
            assert _test_ctx_var.get() is True
            raise RuntimeError("boom")

        token = _test_ctx_var.set(True)
        try:
            # when - first "request" sets the var and its step blows up
            with pytest.raises(RuntimeError, match="boom"):
                run_steps_in_parallel(
                    steps=[suppressing_task_that_raises],
                    max_workers=1,
                    executor=executor,
                )
        finally:
            _test_ctx_var.reset(token)

        def ordinary_task() -> bool:
            return _test_ctx_var.get()

        # when - a later ordinary "request" reuses the same executor/thread,
        # with no override active in the calling thread
        results = run_steps_in_parallel(
            steps=[ordinary_task],
            max_workers=1,
            executor=executor,
        )

    # then - the earlier request's suppression must not have leaked
    assert results == [False]
