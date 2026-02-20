import json
import threading

from inference_sdk.config import RemoteProcessingTimeCollector, remote_processing_times


def test_collector_add_and_drain() -> None:
    # given
    collector = RemoteProcessingTimeCollector()

    # when
    collector.add(0.5, model_id="model_a")
    collector.add(0.3, model_id="model_b")

    # then
    entries = collector.drain()
    assert entries == [("model_a", 0.5), ("model_b", 0.3)]


def test_drain_clears_entries() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    collector.add(0.5, model_id="m1")

    # when
    first = collector.drain()
    second = collector.drain()

    # then
    assert len(first) == 1
    assert len(second) == 0
    assert collector.has_data() is False


def test_collector_has_data_when_empty() -> None:
    # given
    collector = RemoteProcessingTimeCollector()

    # then
    assert collector.has_data() is False


def test_collector_has_data_when_populated() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    collector.add(0.1)

    # then
    assert collector.has_data() is True


def test_collector_default_model_id() -> None:
    # given
    collector = RemoteProcessingTimeCollector()

    # when
    collector.add(0.5)

    # then
    entries = collector.drain()
    assert entries == [("unknown", 0.5)]


def test_collector_thread_safety() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    num_threads = 10
    adds_per_thread = 100

    def add_times(thread_id: int) -> None:
        for i in range(adds_per_thread):
            collector.add(0.001, model_id=f"thread_{thread_id}")

    # when
    threads = [
        threading.Thread(target=add_times, args=(t,)) for t in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # then
    entries = collector.drain()
    assert len(entries) == num_threads * adds_per_thread
    total = sum(t for _, t in entries)
    assert abs(total - num_threads * adds_per_thread * 0.001) < 1e-6


def test_summarize_returns_total_and_json() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    collector.add(0.5, model_id="yolov8")
    collector.add(0.3, model_id="clip")

    # when
    total, detail = collector.summarize()

    # then
    assert abs(total - 0.8) < 1e-9
    parsed = json.loads(detail)
    assert parsed == [
        {"m": "yolov8", "t": 0.5},
        {"m": "clip", "t": 0.3},
    ]
    # drain was called internally, so collector should be empty
    assert collector.has_data() is False


def test_summarize_omits_detail_when_too_large() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    for i in range(200):
        collector.add(0.1, model_id=f"model_{i:04d}")

    # when
    total, detail = collector.summarize(max_detail_bytes=100)

    # then
    assert total > 0
    assert detail is None


def test_contextvar_default_is_none() -> None:
    # when
    value = remote_processing_times.get()

    # then
    assert value is None


def test_contextvar_set_and_get() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    token = remote_processing_times.set(collector)

    try:
        # when
        retrieved = remote_processing_times.get()

        # then
        assert retrieved is collector
    finally:
        remote_processing_times.reset(token)
