import json

from inference.usage_tracking.payload_helpers import (
    merge_usage_dicts,
    zip_usage_payloads,
)


def _row(resource_details=None, **overrides):
    row = {
        "resource_id": "workflow-1",
        "category": "workflows",
        "timestamp_start": 10,
        "timestamp_stop": 20,
        "processed_frames": 1,
        "execution_duration": 0.5,
        "exec_session_id": "session",
    }
    if resource_details is not None:
        row["resource_details"] = resource_details
    row.update(overrides)
    return row


def test_merge_usage_dicts_sums_models_with_same_id_and_keeps_later_fields() -> None:
    # given
    d1 = _row(
        {
            "models": [
                {
                    "model_id": "a/1",
                    "frames": 2,
                    "execution_duration": 1.0,
                    "model_variant": "old",
                }
            ]
        }
    )
    d2 = _row(
        {
            "models": [
                {
                    "model_id": "a/1",
                    "frames": 3,
                    "execution_duration": 0.5,
                    "model_variant": "new",
                }
            ]
        }
    )

    # when
    result = merge_usage_dicts(d1, d2)

    # then
    assert result["resource_details"]["models"] == [
        {
            "model_id": "a/1",
            "frames": 5,
            "execution_duration": 1.5,
            "model_variant": "new",
        }
    ]


def test_merge_usage_dicts_keeps_models_with_different_ids_in_order() -> None:
    # given
    d1 = _row({"models": [{"model_id": "a/1", "frames": 1, "execution_duration": 1}]})
    d2 = _row({"models": [{"model_id": "b/2", "frames": 4, "execution_duration": 2}]})

    # when
    result = merge_usage_dicts(d1, d2)

    # then
    assert result["resource_details"]["models"] == [
        {"model_id": "a/1", "frames": 1, "execution_duration": 1},
        {"model_id": "b/2", "frames": 4, "execution_duration": 2},
    ]


def test_merge_usage_dicts_merges_custom_python_by_block_type_and_step_name() -> None:
    # given
    d1 = _row(
        {
            "custom_python": [
                {"block_type": "cp", "step_name": "s1", "execution_duration": 1.0},
                {"block_type": "cp", "step_name": "s2", "execution_duration": 2.0},
            ]
        }
    )
    d2 = _row(
        {
            "custom_python": [
                {"block_type": "cp", "step_name": "s1", "execution_duration": 0.25},
            ]
        }
    )

    # when
    result = merge_usage_dicts(d1, d2)

    # then
    assert result["resource_details"]["custom_python"] == [
        {"block_type": "cp", "step_name": "s1", "execution_duration": 1.25},
        {"block_type": "cp", "step_name": "s2", "execution_duration": 2.0},
    ]


def test_merge_usage_dicts_without_billable_lists_is_last_wins_as_before() -> None:
    # given
    d1 = _row({"workflow_id": "old", "extra": 1}, fps=None)
    d2 = _row({"workflow_id": "new"}, processed_frames=2, execution_duration=0.25)

    # when
    result = merge_usage_dicts(d1, d2)

    # then
    assert result == {
        "resource_id": "workflow-1",
        "category": "workflows",
        "timestamp_start": 10,
        "timestamp_stop": 20,
        "processed_frames": 3,
        "execution_duration": 0.75,
        "exec_session_id": "session",
        "fps": None,
        "resource_details": {"workflow_id": "new"},
    }


def test_merge_usage_dicts_with_resource_details_on_one_side_takes_that_side() -> None:
    # given
    d1 = _row()
    d2 = _row({"models": [{"model_id": "a/1", "frames": 1}]})

    # when
    result = merge_usage_dicts(d1, d2)

    # then
    assert result["resource_details"] == {"models": [{"model_id": "a/1", "frames": 1}]}


def test_merge_usage_dicts_handles_string_resource_details() -> None:
    # given
    d1 = _row(
        json.dumps({"models": [{"model_id": "a/1", "frames": 1}], "keep": "left"})
    )
    d2 = _row(json.dumps({"models": [{"model_id": "a/1", "frames": 2}]}))

    # when
    result = merge_usage_dicts(d1, d2)

    # then
    assert isinstance(result["resource_details"], str)
    assert json.loads(result["resource_details"]) == {
        "models": [{"model_id": "a/1", "frames": 3}]
    }


def test_merge_usage_dicts_leaves_undecodable_resource_details_last_wins() -> None:
    # given
    d1 = _row("not json")
    d2 = _row({"models": [{"model_id": "a/1", "frames": 1}]})

    # when
    result = merge_usage_dicts(d1, d2)

    # then
    assert result["resource_details"] == {"models": [{"model_id": "a/1", "frames": 1}]}


def test_merge_usage_dicts_keeps_later_models_on_unhashable_identity() -> None:
    # given
    d1 = _row(
        {
            "models": [{"model_id": [], "frames": 1}],
            "custom_python": [
                {"block_type": "b", "step_name": "s", "execution_duration": 1}
            ],
        }
    )
    d2 = _row(
        {
            "models": [{"model_id": "a/1", "frames": 2}],
            "custom_python": [
                {"block_type": "b", "step_name": "s", "execution_duration": 2}
            ],
        }
    )

    # when
    result = merge_usage_dicts(d1, d2)

    # then
    assert result["resource_details"]["models"] == [{"model_id": "a/1", "frames": 2}]
    assert result["resource_details"]["custom_python"] == [
        {"block_type": "b", "step_name": "s", "execution_duration": 3}
    ]


def test_merge_usage_dicts_keeps_later_models_on_non_numeric_amount() -> None:
    # given
    d1 = _row({"models": [{"model_id": "a/1", "frames": 1}]})
    d2 = _row({"models": [{"model_id": "a/1", "frames": "2"}]})

    # when
    result = merge_usage_dicts(d1, d2)

    # then
    assert result["resource_details"]["models"] == [{"model_id": "a/1", "frames": "2"}]


def test_zip_usage_payloads_preserves_entries_per_resource() -> None:
    # given
    def payload(resource_id, model_id, frames):
        return {
            "hash": {
                f"workflows:{resource_id}": _row(
                    {
                        "models": [
                            {
                                "model_id": model_id,
                                "frames": frames,
                                "execution_duration": 1,
                            }
                        ]
                    },
                    resource_id=resource_id,
                )
            }
        }

    payloads = [
        payload("wf-a", "m/1", 1),
        payload("wf-b", "m/9", 7),
        payload("wf-a", "m/1", 2),
    ]

    # when
    result = zip_usage_payloads(payloads)

    # then
    assert len(result) == 1
    rows = result[0]["hash"]
    assert rows["workflows:wf-a"]["resource_details"]["models"] == [
        {"model_id": "m/1", "frames": 3, "execution_duration": 2}
    ]
    assert rows["workflows:wf-b"]["resource_details"]["models"] == [
        {"model_id": "m/9", "frames": 7, "execution_duration": 1}
    ]
