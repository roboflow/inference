from inference.core.interfaces.http.handlers.workflows import (
    filter_out_unwanted_workflow_outputs,
)


def test_filter_out_unwanted_workflow_outputs_when_nothing_to_filter() -> None:
    # given
    workflow_results = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    # when
    result = filter_out_unwanted_workflow_outputs(
        workflow_results=workflow_results,
        excluded_fields=None,
    )

    # then
    assert result == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_filter_out_unwanted_workflow_outputs_when_empty_filter() -> None:
    # given
    workflow_results = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    # when
    result = filter_out_unwanted_workflow_outputs(
        workflow_results=workflow_results,
        excluded_fields=[],
    )

    # then
    assert result == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_filter_out_unwanted_workflow_outputs_when_fields_to_be_filtered() -> None:
    # given
    workflow_results = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    # when
    result = filter_out_unwanted_workflow_outputs(
        workflow_results=workflow_results,
        excluded_fields=["a"],
    )

    # then
    assert result == [
        {"b": 2},
        {"b": 4},
    ]


def test_filter_out_unwanted_workflow_outputs_when_filter_defines_non_existing_fields() -> (
    None
):
    # given
    workflow_results = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    # when
    result = filter_out_unwanted_workflow_outputs(
        workflow_results=workflow_results,
        excluded_fields=["non-existing"],
    )

    # then
    assert result == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]
