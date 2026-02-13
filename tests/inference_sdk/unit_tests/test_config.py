import threading

from inference_sdk.config import RemoteProcessingTimeCollector, remote_processing_times


def test_collector_add_and_get_entries() -> None:
    # given
    collector = RemoteProcessingTimeCollector()

    # when
    collector.add(0.5, model_id="model_a")
    collector.add(0.3, model_id="model_b")

    # then
    entries = collector.get_entries()
    assert entries == [("model_a", 0.5), ("model_b", 0.3)]


def test_collector_get_total() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    collector.add(0.5, model_id="m1")
    collector.add(0.3, model_id="m2")
    collector.add(0.2, model_id="m1")

    # when
    total = collector.get_total()

    # then
    assert abs(total - 1.0) < 1e-9


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
    entries = collector.get_entries()
    assert entries == [("unknown", 0.5)]


def test_collector_get_entries_returns_copy() -> None:
    # given
    collector = RemoteProcessingTimeCollector()
    collector.add(0.5, model_id="m1")

    # when
    entries = collector.get_entries()
    entries.append(("m2", 0.9))

    # then
    assert len(collector.get_entries()) == 1


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
    entries = collector.get_entries()
    assert len(entries) == num_threads * adds_per_thread
    assert abs(collector.get_total() - num_threads * adds_per_thread * 0.001) < 1e-6


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
