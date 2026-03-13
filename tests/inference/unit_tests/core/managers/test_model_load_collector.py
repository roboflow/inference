import json

from inference.core.managers.model_load_collector import (
    ModelLoadCollector,
    RequestModelIds,
    model_load_info,
    request_model_ids,
    request_workflow_id,
)


def test_empty_collector_has_no_data() -> None:
    # given
    collector = ModelLoadCollector()

    # then
    assert collector.has_data() is False


def test_collector_has_data_after_record() -> None:
    # given
    collector = ModelLoadCollector()

    # when
    collector.record("some/1", 1.5)

    # then
    assert collector.has_data() is True


def test_summarize_single_entry() -> None:
    # given
    collector = ModelLoadCollector()
    collector.record("some/1", 2.5)

    # when
    total, detail = collector.summarize()

    # then
    assert abs(total - 2.5) < 1e-9
    parsed = json.loads(detail)
    assert len(parsed) == 1
    assert parsed[0]["m"] == "some/1"
    assert abs(parsed[0]["t"] - 2.5) < 1e-9


def test_summarize_multiple_entries() -> None:
    # given
    collector = ModelLoadCollector()
    collector.record("model-a/1", 2.5)
    collector.record("model-b/2", 1.0)
    collector.record("model-c/3", 0.3)

    # when
    total, detail = collector.summarize()

    # then
    assert abs(total - 3.8) < 1e-9
    parsed = json.loads(detail)
    assert len(parsed) == 3
    assert [e["m"] for e in parsed] == ["model-a/1", "model-b/2", "model-c/3"]


def test_summarize_omits_detail_when_exceeding_max_bytes() -> None:
    # given
    collector = ModelLoadCollector()
    collector.record("some/1", 1.0)

    # when - use a very small max_detail_bytes to force omission
    total, detail = collector.summarize(max_detail_bytes=5)

    # then
    assert abs(total - 1.0) < 1e-9
    assert detail is None


def test_summarize_empty_collector() -> None:
    # given
    collector = ModelLoadCollector()

    # when
    total, detail = collector.summarize()

    # then
    assert total == 0.0
    assert detail == "[]"


def test_context_var_defaults_to_none() -> None:
    # then
    assert model_load_info.get(None) is None


# --- RequestModelIds tests ---


def test_request_model_ids_empty() -> None:
    # given
    ids = RequestModelIds()

    # then
    assert ids.get_ids() == set()


def test_request_model_ids_add_and_get() -> None:
    # given
    ids = RequestModelIds()

    # when
    ids.add("model-a/1")
    ids.add("model-b/2")

    # then
    assert ids.get_ids() == {"model-a/1", "model-b/2"}


def test_request_model_ids_deduplicates() -> None:
    # given
    ids = RequestModelIds()

    # when
    ids.add("model-a/1")
    ids.add("model-a/1")
    ids.add("model-b/2")

    # then
    assert ids.get_ids() == {"model-a/1", "model-b/2"}


def test_request_model_ids_context_var_defaults_to_none() -> None:
    # then
    assert request_model_ids.get(None) is None


def test_request_workflow_id_context_var_defaults_to_none() -> None:
    # then
    assert request_workflow_id.get(None) is None
