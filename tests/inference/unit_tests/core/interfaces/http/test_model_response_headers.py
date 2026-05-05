import json

from inference.core.constants import (
    MODEL_COLD_START_COUNT_HEADER,
    MODEL_COLD_START_HEADER,
    MODEL_ID_HEADER,
    MODEL_LOAD_DETAILS_HEADER,
    MODEL_LOAD_TIME_HEADER,
)
from inference.core.interfaces.http.request_metrics import build_model_response_headers


def test_build_model_response_headers_for_remote_only_cold_start() -> None:
    # when
    result = build_model_response_headers(
        local_model_ids=set(),
        local_cold_start_entries=[],
        remote_model_ids={"remote-model/1"},
        remote_cold_start_entries=[("remote-model/1", 0.8)],
        remote_cold_start_count=1,
        remote_cold_start_total_load_time=0.8,
    )

    # then
    assert result[MODEL_COLD_START_HEADER] == "true"
    assert result[MODEL_COLD_START_COUNT_HEADER] == "1"
    assert result[MODEL_ID_HEADER] == "remote-model/1"
    assert abs(float(result[MODEL_LOAD_TIME_HEADER]) - 0.8) < 1e-9
    assert json.loads(result[MODEL_LOAD_DETAILS_HEADER]) == [
        {"m": "remote-model/1", "t": 0.8}
    ]


def test_build_model_response_headers_merges_local_and_remote_models() -> None:
    # when
    result = build_model_response_headers(
        local_model_ids={"local-model/1"},
        local_cold_start_entries=[("local-model/1", 0.3)],
        remote_model_ids={"remote-model/2"},
        remote_cold_start_entries=[("remote-model/2", 0.7)],
        remote_cold_start_count=1,
        remote_cold_start_total_load_time=0.7,
    )

    # then
    assert result[MODEL_ID_HEADER] == "local-model/1,remote-model/2"
    assert result[MODEL_COLD_START_HEADER] == "true"
    assert result[MODEL_COLD_START_COUNT_HEADER] == "2"
    assert abs(float(result[MODEL_LOAD_TIME_HEADER]) - 1.0) < 1e-9
    assert json.loads(result[MODEL_LOAD_DETAILS_HEADER]) == [
        {"m": "local-model/1", "t": 0.3},
        {"m": "remote-model/2", "t": 0.7},
    ]


def test_build_model_response_headers_omits_partial_remote_details() -> None:
    # when
    result = build_model_response_headers(
        local_model_ids=set(),
        local_cold_start_entries=[],
        remote_model_ids={"model-a/1", "model-b/2"},
        remote_cold_start_entries=[],
        remote_cold_start_count=1,
        remote_cold_start_total_load_time=1.4,
    )

    # then
    assert result[MODEL_COLD_START_HEADER] == "true"
    assert result[MODEL_COLD_START_COUNT_HEADER] == "1"
    assert result[MODEL_ID_HEADER] == "model-a/1,model-b/2"
    assert abs(float(result[MODEL_LOAD_TIME_HEADER]) - 1.4) < 1e-9
    assert MODEL_LOAD_DETAILS_HEADER not in result


def test_build_model_response_headers_sets_zero_count_when_no_cold_start() -> None:
    # when
    result = build_model_response_headers(
        local_model_ids={"model-a/1"},
        local_cold_start_entries=[],
        remote_model_ids=set(),
        remote_cold_start_entries=[],
        remote_cold_start_count=0,
        remote_cold_start_total_load_time=0.0,
    )

    # then
    assert result[MODEL_COLD_START_HEADER] == "false"
    assert result[MODEL_COLD_START_COUNT_HEADER] == "0"
    assert result[MODEL_ID_HEADER] == "model-a/1"
    assert MODEL_LOAD_TIME_HEADER not in result
    assert MODEL_LOAD_DETAILS_HEADER not in result
