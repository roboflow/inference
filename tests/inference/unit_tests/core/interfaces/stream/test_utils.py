import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.stream import utils
from inference.core.interfaces.stream.utils import (
    broadcast_elements,
    on_pipeline_end,
    save_workflows_profiler_trace,
    wrap_in_list,
)


def test_wrap_in_list_when_list_provided() -> None:
    # given
    element = [1, 2, 3]

    # when
    result = wrap_in_list(element=element)

    # then
    assert result == [1, 2, 3], "Order of elements must be preserved"
    assert result is element, "The same object should be returned"


def test_wrap_in_list_when_single_element_provided() -> None:
    # given
    element = 1

    # when
    result = wrap_in_list(element=element)

    # then
    assert result == [1], "Expected to wrap element with list"


def test_broadcast_elements_when_desired_length_matches_elements() -> None:
    # given
    element = [1, 2, 3]

    # when
    result = broadcast_elements(
        elements=element, desired_length=3, error_description="some"
    )

    # then
    assert result == [1, 2, 3], "Order of elements must be preserved"
    assert result is element, "The same object should be returned"


def test_broadcast_elements_when_desired_length_do_not_match_elements() -> None:
    # given
    element = [1, 2, 3]

    # when
    with pytest.raises(ValueError):
        _ = broadcast_elements(
            elements=element, desired_length=4, error_description="some"
        )


def test_broadcast_elements_when_desired_length_do_not_match_elements_but_can_be_broadcast() -> (
    None
):
    # given
    element = [1]

    # when
    result = broadcast_elements(
        elements=element, desired_length=3, error_description="some"
    )

    # then
    assert result == [1, 1, 1]


def test_broadcast_elements_when_input_is_empty() -> None:
    # given
    element = []

    # when
    with pytest.raises(ValueError):
        _ = broadcast_elements(
            elements=element, desired_length=3, error_description="some"
        )


def test_save_workflows_profiler_trace(empty_directory: str) -> None:
    # when
    save_workflows_profiler_trace(
        directory=empty_directory,
        profiler_trace=[{"my": "trace"}],
    )

    # then
    json_files_in_directory = glob(os.path.join(empty_directory, "*.json"))
    assert len(json_files_in_directory) == 1, "Expected single JSON file to be created"
    with open(json_files_in_directory[0], "r") as f:
        result = json.load(f)
    assert result == [{"my": "trace"}], "Expected dump to preserve content"


@mock.patch.object(utils, "ENABLE_WORKFLOWS_PROFILING", False)
def test_on_pipeline_end_when_profiling_disabled(empty_directory: str) -> None:
    # given
    profiler = MagicMock()
    profiler.export_trace.return_value = [{"my": "trace"}]
    thread_pool_executor = ThreadPoolExecutor(max_workers=3)

    # when
    on_pipeline_end(
        thread_pool_executor=thread_pool_executor,
        cancel_thread_pool_tasks_on_exit=True,
        profiler=profiler,
        profiling_directory=empty_directory,
    )

    # then
    assert thread_pool_executor._shutdown is True, "Expected pool executor to be closed"
    json_files_in_directory = glob(os.path.join(empty_directory, "*.json"))
    assert len(json_files_in_directory) == 0, "Expected no profiler trace saved"


@mock.patch.object(utils, "ENABLE_WORKFLOWS_PROFILING", True)
def test_on_pipeline_end_when_profiling_disabled(empty_directory: str) -> None:
    # given
    profiler = MagicMock()
    profiler.export_trace.return_value = [{"my": "trace"}]
    thread_pool_executor = ThreadPoolExecutor(max_workers=3)

    # when
    on_pipeline_end(
        thread_pool_executor=thread_pool_executor,
        cancel_thread_pool_tasks_on_exit=True,
        profiler=profiler,
        profiling_directory=empty_directory,
    )

    # then
    assert thread_pool_executor._shutdown is True, "Expected pool executor to be closed"
    json_files_in_directory = glob(os.path.join(empty_directory, "*.json"))
    assert len(json_files_in_directory) == 1, "Expected profiler trace saved"
    with open(json_files_in_directory[0], "r") as f:
        result = json.load(f)
    assert result == [{"my": "trace"}], "Expected dump to preserve content"
