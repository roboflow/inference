from inference_sdk.config import (
    RemoteProcessingTimeCollector,
    execution_id,
    remote_processing_times,
)

from inference.core.workflows.core_steps.common.utils import run_in_parallel


def test_run_in_parallel_propagates_processing_time_collector() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    collector_token = remote_processing_times.set(collector)
    exec_token = execution_id.set("test-exec-id")

    def task_that_reads_context() -> tuple:
        ctx_collector = remote_processing_times.get()
        ctx_exec_id = execution_id.get()
        if ctx_collector is not None:
            ctx_collector.add(0.1, model_id="test_model")
        return (ctx_collector is not None, ctx_exec_id)

    try:
        # when
        results = run_in_parallel(
            tasks=[task_that_reads_context, task_that_reads_context],
            max_workers=2,
        )
    finally:
        remote_processing_times.reset(collector_token)
        execution_id.reset(exec_token)

    # then
    assert all(has_collector for has_collector, _ in results)
    assert all(eid == "test-exec-id" for _, eid in results)
    entries = collector.drain()
    assert len(entries) == 2
    assert entries[0][0] == "test_model"


def test_run_in_parallel_works_without_collector_set() -> None:
    # given - no collector set (default None)
    token = remote_processing_times.set(None)

    def simple_task() -> int:
        return 42

    try:
        # when
        results = run_in_parallel(tasks=[simple_task, simple_task], max_workers=2)
    finally:
        remote_processing_times.reset(token)

    # then
    assert results == [42, 42]


def test_run_in_parallel_shared_collector_across_threads() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    token = remote_processing_times.set(collector)

    def add_time(model_id: str):
        def task():
            ctx_collector = remote_processing_times.get()
            ctx_collector.add(0.5, model_id=model_id)
            return model_id

        return task

    try:
        # when
        results = run_in_parallel(
            tasks=[add_time("m1"), add_time("m2"), add_time("m3")],
            max_workers=3,
        )
    finally:
        remote_processing_times.reset(token)

    # then
    assert sorted(results) == ["m1", "m2", "m3"]
    entries = collector.drain()
    assert len(entries) == 3
    total = sum(t for _, t in entries)
    assert abs(total - 1.5) < 1e-9
