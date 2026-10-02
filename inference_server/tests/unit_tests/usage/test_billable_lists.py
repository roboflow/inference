import json

import pytest

from inference_server.usage import payload_helpers
from inference_server.usage.payload_helpers import (
    merge_resource_details,
    merge_usage_dicts,
    zip_usage_payloads,
)


def model_entry(model_id, *, frames=1, execution_duration=0.5, latency=10.0, **extra):
    entry = {
        "model_id": model_id,
        "model_architecture": "yolov8",
        "model_variant": "n",
        "task_type": "object-detection",
        "model_input_height": 640,
        "model_input_width": 640,
        "model_latency_ms": latency,
        "execution_duration": execution_duration,
        "frames": frames,
        **extra,
    }

    return entry


def custom_python_entry(block_type, step_name, *, execution_duration=0.25):
    entry = {
        "block_type": block_type,
        "step_name": step_name,
        "execution_duration": execution_duration,
    }

    return entry


def row(details, *, frames=1):
    usage_row = {
        "resource_id": "workflow-1",
        "api_key_hash": "hash",
        "timestamp_start": 1,
        "timestamp_stop": 2,
        "processed_frames": frames,
        "source_duration": 0,
        "execution_duration": 1,
        "resource_details": json.dumps(details),
    }

    return usage_row


def test_merge_usage_dicts_keeps_models_of_both_rows():
    first = row({"billable": True, "models": [model_entry("coco/3")]})
    second = row({"billable": True, "models": [model_entry("other/1")]})

    merged = merge_usage_dicts(first, second)

    details = json.loads(merged["resource_details"])
    assert [entry["model_id"] for entry in details["models"]] == ["coco/3", "other/1"]
    assert isinstance(merged["resource_details"], str)


def test_merge_usage_dicts_sums_amounts_and_keeps_latest_latency_of_the_same_model():
    first = row(
        {
            "models": [
                model_entry(
                    "coco/3",
                    frames=2,
                    execution_duration=0.5,
                    latency=10.0,
                    model_variant="first",
                )
            ]
        }
    )
    second = row(
        {
            "models": [
                model_entry(
                    "coco/3",
                    frames=3,
                    execution_duration=0.25,
                    latency=5.0,
                    model_variant="second",
                )
            ]
        }
    )

    merged = merge_usage_dicts(first, second)

    details = json.loads(merged["resource_details"])
    assert details["models"] == [
        model_entry(
            "coco/3",
            frames=5,
            execution_duration=0.75,
            latency=5.0,
            model_variant="second",
        )
    ]


def test_merge_usage_dicts_does_not_sum_model_latency():
    first = row({"models": [model_entry("coco/3", latency=10.0)]})
    second = row({"models": [model_entry("coco/3", latency=5.0)]})

    merged = merge_usage_dicts(first, second)

    (entry,) = json.loads(merged["resource_details"])["models"]
    assert entry["frames"] == 2
    assert entry["execution_duration"] == 1.0
    assert entry["model_latency_ms"] == 5.0


def test_merge_usage_dicts_merges_custom_python_by_block_type_and_step_name():
    first = row(
        {
            "custom_python": [
                custom_python_entry("block_a", "step_1", execution_duration=0.5),
                custom_python_entry("block_a", "step_2", execution_duration=1.0),
            ]
        }
    )
    second = row(
        {
            "custom_python": [
                custom_python_entry("block_a", "step_1", execution_duration=0.25),
                custom_python_entry("block_b", "step_1", execution_duration=2.0),
            ]
        }
    )

    merged = merge_usage_dicts(first, second)

    details = json.loads(merged["resource_details"])
    assert details["custom_python"] == [
        custom_python_entry("block_a", "step_1", execution_duration=0.75),
        custom_python_entry("block_a", "step_2", execution_duration=1.0),
        custom_python_entry("block_b", "step_1", execution_duration=2.0),
    ]


def test_merge_usage_dicts_keeps_last_wins_for_other_resource_details_keys():
    first = row(
        {"source_info": "first", "only_first": 1, "models": [model_entry("coco/3")]}
    )
    second = row({"source_info": "second", "models": [model_entry("other/1")]})

    merged = merge_usage_dicts(first, second)

    details = json.loads(merged["resource_details"])
    assert details["source_info"] == "second"
    assert "only_first" not in details


def test_merge_usage_dicts_without_lists_is_last_wins_as_legacy():
    first = row({"billable": True, "source_info": "first"})
    second = row({"billable": True, "source_info": "second"})

    merged = merge_usage_dicts(first, second)

    assert merged["resource_details"] is second["resource_details"]
    assert merged["processed_frames"] == 2


def test_merge_usage_dicts_keeps_lists_carried_by_one_row_only():
    first = row({"models": [model_entry("coco/3")]})
    second = row({"billable": True})

    merged = merge_usage_dicts(first, second)

    details = json.loads(merged["resource_details"])
    assert details == {"billable": True, "models": [model_entry("coco/3")]}


def test_merge_resource_details_returns_the_form_of_the_later_value():
    stored = json.dumps({"models": [model_entry("coco/3")]})
    current = {"models": [model_entry("coco/3")]}

    merged = merge_resource_details(stored, current)

    assert isinstance(merged, dict)
    assert merged["models"][0]["frames"] == 2


@pytest.mark.parametrize("stored", ["not json", "[1, 2]", None, 7])
def test_merge_resource_details_leaves_undecodable_values_to_last_wins(stored):
    current = {"models": [model_entry("coco/3")]}

    assert merge_resource_details(stored, current) is current


def test_zip_usage_payloads_merges_lists_across_payloads():
    payloads = [
        {"hash": {"workflows:workflow-1": row({"models": [model_entry("coco/3")]})}},
        {"hash": {"workflows:workflow-1": row({"models": [model_entry("coco/3")]})}},
        {"hash": {"workflows:workflow-1": row({"models": [model_entry("other/1")]})}},
    ]

    zipped = zip_usage_payloads(usage_payloads=payloads)

    assert len(zipped) == 1
    merged = zipped[0]["hash"]["workflows:workflow-1"]
    details = json.loads(merged["resource_details"])
    assert merged["processed_frames"] == 3
    assert [(entry["model_id"], entry["frames"]) for entry in details["models"]] == [
        ("coco/3", 2),
        ("other/1", 1),
    ]


def test_zip_usage_payloads_does_not_sum_model_latency():
    payloads = [
        {
            "hash": {
                "workflows:workflow-1": row(
                    {"models": [model_entry("coco/3", latency=10.0)]}
                )
            }
        },
        {
            "hash": {
                "workflows:workflow-1": row(
                    {"models": [model_entry("coco/3", latency=5.0)]}
                )
            }
        },
    ]

    zipped = zip_usage_payloads(usage_payloads=payloads)

    (merged,) = zipped
    (entry,) = json.loads(merged["hash"]["workflows:workflow-1"]["resource_details"])[
        "models"
    ]
    assert entry["frames"] == 2
    assert entry["execution_duration"] == 1.0
    assert entry["model_latency_ms"] == 5.0


def test_zip_usage_payloads_starts_a_new_row_when_a_list_would_exceed_the_bound(
    monkeypatch,
):
    monkeypatch.setattr(payload_helpers, "MAX_BILLABLE_ENTRIES_PER_ROW", 2)
    payloads = [
        {
            "hash": {
                "workflows:workflow-1": row({"models": [model_entry(f"model/{index}")]})
            }
        }
        for index in range(5)
    ]

    zipped = zip_usage_payloads(usage_payloads=payloads)

    rows = [payload["hash"]["workflows:workflow-1"] for payload in zipped]
    model_ids = [
        [entry["model_id"] for entry in json.loads(r["resource_details"])["models"]]
        for r in rows
    ]
    assert sorted(model_ids) == [
        ["model/0", "model/1"],
        ["model/2", "model/3"],
        ["model/4"],
    ]
    assert sum(r["processed_frames"] for r in rows) == 5
