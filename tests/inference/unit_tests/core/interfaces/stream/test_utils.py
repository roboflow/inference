import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from inspect import signature
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.stream import utils
from inference.core.interfaces.stream.utils import (
    broadcast_elements,
    initialise_video_sources,
    on_pipeline_end,
    prepare_video_sources,
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


@pytest.mark.parametrize(
    "callable_under_test",
    [prepare_video_sources, initialise_video_sources],
)
def test_video_source_helpers_preserve_legacy_positional_parameter_order(
    callable_under_test,
) -> None:
    legacy_parameters = (
        "video_reference",
        "video_source_properties",
        "status_update_handlers",
        "source_buffer_filling_strategy",
        "source_buffer_consumption_strategy",
        "desired_source_fps",
        "decoding_buffer_size",
        "allow_tensor_frames",
    )

    assert tuple(signature(callable_under_test).parameters) == (
        *legacy_parameters,
        "video_source_options",
    )


@mock.patch.object(utils.VideoSource, "init")
def test_prepare_video_sources_accepts_legacy_positional_arguments(
    video_source_init: MagicMock,
) -> None:
    prepare_video_sources(["a"], None, None, None, None, 12, 3, True)

    video_source_init.assert_called_once_with(
        video_reference="a",
        status_update_handlers=None,
        buffer_filling_strategy=None,
        buffer_consumption_strategy=None,
        video_source_properties=None,
        video_source_options=None,
        source_id=0,
        desired_fps=12,
        buffer_size=3,
        allow_tensor_frames=True,
    )


@mock.patch.object(utils.VideoSource, "init")
def test_prepare_video_sources_broadcasts_per_source_options(
    video_source_init: MagicMock,
) -> None:
    prepare_video_sources(
        video_reference=["a", "b"],
        video_source_properties=None,
        video_source_options={"rtsp_tls_validation_flags": 0},
        status_update_handlers=None,
        source_buffer_filling_strategy=None,
        source_buffer_consumption_strategy=None,
    )

    assert video_source_init.call_count == 2
    for call in video_source_init.call_args_list:
        assert call.kwargs["video_source_options"] == {"rtsp_tls_validation_flags": 0}


@mock.patch.object(utils.VideoSource, "init")
def test_prepare_video_sources_applies_aligned_per_source_options(
    video_source_init: MagicMock,
) -> None:
    prepare_video_sources(
        video_reference=["a", "b"],
        video_source_properties=None,
        video_source_options=[
            {"rtsp_tls_validation_flags": 0},
            None,
        ],
        status_update_handlers=None,
        source_buffer_filling_strategy=None,
        source_buffer_consumption_strategy=None,
    )

    assert video_source_init.call_count == 2
    assert video_source_init.call_args_list[0].kwargs["video_source_options"] == {
        "rtsp_tls_validation_flags": 0
    }
    assert video_source_init.call_args_list[1].kwargs["video_source_options"] is None


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
def test_on_pipeline_end_when_profiling_enabled(empty_directory: str) -> None:
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


@mock.patch.object(utils, "ENABLE_WORKFLOWS_PROFILING", True)
def test_on_pipeline_end_when_profiling_directory_readonly(
    empty_directory: str,
) -> None:
    """Pipeline should stop cleanly even when profiling directory is read-only."""
    # given
    profiler = MagicMock()
    profiler.export_trace.return_value = [{"my": "trace"}]
    thread_pool_executor = ThreadPoolExecutor(max_workers=3)
    read_only_dir = os.path.join(empty_directory, "readonly_profiling")

    # Mock os.makedirs to raise OSError (read-only filesystem)
    original_makedirs = os.makedirs

    def mock_makedirs(name, *args, **kwargs):
        if name == os.path.abspath(read_only_dir):
            raise OSError(30, "Read-only file system", name)
        return original_makedirs(name, *args, **kwargs)

    # when
    with mock.patch("os.makedirs", side_effect=mock_makedirs):
        on_pipeline_end(
            thread_pool_executor=thread_pool_executor,
            cancel_thread_pool_tasks_on_exit=True,
            profiler=profiler,
            profiling_directory=read_only_dir,
        )

    # then - pipeline should still stop cleanly
    assert thread_pool_executor._shutdown is True, "Expected pool executor to be closed"
    # No profiling files should have been created
    json_files_in_directory = glob(os.path.join(read_only_dir, "*.json"))
    assert (
        len(json_files_in_directory) == 0
    ), "Expected no profiler trace saved on read-only FS"
