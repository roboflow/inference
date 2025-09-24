import json
import os.path
from glob import glob

import pandas as pd
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.sinks.local_file.v1 import (
    BlockManifest,
    LocalFileSinkBlockV1,
    path_is_within_specified_directory,
)


def test_manifest_parsing_when_input_is_valid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/local_file_sink@v1",
        "name": "predictions_sink",
        "content": "$steps.json_formatter.output",
        "file_type": "json",
        "output_mode": "separate_files",
        "target_directory": "$inputs.target_directory",
        "file_name_prefix": "prediction",
        "max_entries_per_file": "$inputs.max_entries_per_file",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/local_file_sink@v1",
        name="predictions_sink",
        content="$steps.json_formatter.output",
        file_type="json",
        output_mode="separate_files",
        target_directory="$inputs.target_directory",
        file_name_prefix="prediction",
        max_entries_per_file="$inputs.max_entries_per_file",
    )


def test_manifest_parsing_when_max_entries_per_file_param_is_invalid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/local_file_sink@v1",
        "name": "predictions_sink",
        "content": "$steps.json_formatter.output",
        "file_type": "json",
        "output_mode": "separate_files",
        "target_directory": "$inputs.target_directory",
        "file_name_prefix": "prediction",
        "max_entries_per_file": 0,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_saving_into_file_when_execution_is_prevented(empty_directory: str) -> None:
    # given
    block = LocalFileSinkBlockV1(
        allow_access_to_file_system=False, allowed_write_directory=None
    )

    # when
    with pytest.raises(RuntimeError):
        _ = block.run(
            content="content-1",
            file_type="txt",
            output_mode="separate_files",
            target_directory=empty_directory,
            file_name_prefix="my_file",
            max_entries_per_file=100,
        )

    # then
    assert len(os.listdir(empty_directory)) == 0


def test_saving_txt_into_separate_files(empty_directory: str) -> None:
    # given
    block = LocalFileSinkBlockV1(
        allow_access_to_file_system=True, allowed_write_directory=None
    )

    # when
    result_1 = block.run(
        content="content-1",
        file_type="txt",
        output_mode="separate_files",
        target_directory=empty_directory,
        file_name_prefix="my_file",
        max_entries_per_file=100,
    )
    result_2 = block.run(
        content="content-2",
        file_type="txt",
        output_mode="separate_files",
        target_directory=empty_directory,
        file_name_prefix="my_file",
        max_entries_per_file=100,
    )

    # then
    assert result_1["error_status"] is False
    assert result_2["error_status"] is False
    saved_files = sorted(glob(os.path.join(empty_directory, "my_file_*.txt")))
    assert len(saved_files) == 2, "Expected 2 separate files"
    with open(saved_files[0]) as f:
        file_one_content = f.read()
        assert file_one_content == "content-1\n"
    with open(saved_files[1]) as f:
        file_one_content = f.read()
        assert file_one_content == "content-2\n"


def test_saving_txt_into_separate_files_when_allowed_directory_pointed(
    empty_directory: str,
) -> None:
    # given
    block = LocalFileSinkBlockV1(
        allow_access_to_file_system=True, allowed_write_directory=empty_directory
    )

    # when
    result_1 = block.run(
        content="content-1",
        file_type="txt",
        output_mode="separate_files",
        target_directory=empty_directory,
        file_name_prefix="my_file",
        max_entries_per_file=100,
    )
    result_2 = block.run(
        content="content-2",
        file_type="txt",
        output_mode="separate_files",
        target_directory=empty_directory,
        file_name_prefix="my_file",
        max_entries_per_file=100,
    )

    # then
    assert result_1["error_status"] is False
    assert result_2["error_status"] is False
    saved_files = sorted(glob(os.path.join(empty_directory, "my_file_*.txt")))
    assert len(saved_files) == 2, "Expected 2 separate files"
    with open(saved_files[0]) as f:
        file_one_content = f.read()
        assert file_one_content == "content-1\n"
    with open(saved_files[1]) as f:
        file_one_content = f.read()
        assert file_one_content == "content-2\n"


def test_saving_files_when_allowed_write_directory_does_not_match(
    empty_directory: str,
) -> None:
    # given
    block = LocalFileSinkBlockV1(
        allow_access_to_file_system=True, allowed_write_directory=empty_directory
    )

    # when
    with pytest.raises(ValueError):
        _ = block.run(
            content="content-1",
            file_type="txt",
            output_mode="separate_files",
            target_directory=f"{empty_directory}but-something-else",
            file_name_prefix="my_file",
            max_entries_per_file=100,
        )


def test_saving_json_into_separate_files(empty_directory: str) -> None:
    # given
    block = LocalFileSinkBlockV1(
        allow_access_to_file_system=True, allowed_write_directory=None
    )

    # when
    result_1 = block.run(
        content=json.dumps({"some": "data"}),
        file_type="json",
        output_mode="separate_files",
        target_directory=empty_directory,
        file_name_prefix="my_file",
        max_entries_per_file=100,
    )
    result_2 = block.run(
        content=json.dumps({"other": "data"}),
        file_type="json",
        output_mode="separate_files",
        target_directory=empty_directory,
        file_name_prefix="my_file",
        max_entries_per_file=100,
    )

    # then
    assert result_1["error_status"] is False
    assert result_2["error_status"] is False
    saved_files = sorted(glob(os.path.join(empty_directory, "my_file_*.json")))
    assert len(saved_files) == 2, "Expected 2 separate files"
    with open(saved_files[0]) as f:
        file_one_content = json.load(f)
        assert file_one_content == {"some": "data"}
    with open(saved_files[1]) as f:
        file_one_content = json.load(f)
        assert file_one_content == {"other": "data"}


def test_saving_csv_into_separate_files(empty_directory: str) -> None:
    # given
    block = LocalFileSinkBlockV1(
        allow_access_to_file_system=True, allowed_write_directory=None
    )
    df_1 = pd.DataFrame([{"some": "data"}])
    df_2 = pd.DataFrame([{"other": "data"}])

    # when
    result_1 = block.run(
        content=df_1.to_csv(index=False),
        file_type="csv",
        output_mode="separate_files",
        target_directory=empty_directory,
        file_name_prefix="my_file",
        max_entries_per_file=100,
    )
    result_2 = block.run(
        content=df_2.to_csv(index=False),
        file_type="csv",
        output_mode="separate_files",
        target_directory=empty_directory,
        file_name_prefix="my_file",
        max_entries_per_file=100,
    )

    # then
    assert result_1["error_status"] is False
    assert result_2["error_status"] is False
    saved_files = sorted(glob(os.path.join(empty_directory, "my_file_*.csv")))
    assert len(saved_files) == 2, "Expected 2 separate files"
    assert df_1["some"].tolist() == pd.read_csv(saved_files[0])["some"].tolist()
    assert df_2["other"].tolist() == pd.read_csv(saved_files[1])["other"].tolist()


def test_saving_txt_into_append_log(empty_directory: str) -> None:
    # given
    block = LocalFileSinkBlockV1(
        allow_access_to_file_system=True, allowed_write_directory=None
    )

    # when
    for i in range(5):
        _ = block.run(
            content=f"content-{i}",
            file_type="txt",
            output_mode="append_log",
            target_directory=empty_directory,
            file_name_prefix="my_file",
            max_entries_per_file=3,
        )
    # flushing buffer
    del block

    # then
    saved_files = sorted(glob(os.path.join(empty_directory, "my_file_*.txt")))
    assert len(saved_files) == 2, "Expected 2 separate files"
    with open(saved_files[0]) as f:
        file_one_content = f.read()
        assert (
            file_one_content == "content-0\ncontent-1\ncontent-2\n"
        ), "3 entries expected in the first file"
    with open(saved_files[1]) as f:
        file_one_content = f.read()
        assert (
            file_one_content == "content-3\ncontent-4\n"
        ), "2 entries expected in the second file"


def test_saving_json_into_append_log(empty_directory: str) -> None:
    # given
    block = LocalFileSinkBlockV1(
        allow_access_to_file_system=True, allowed_write_directory=None
    )

    # when
    for i in range(5):
        _ = block.run(
            content=json.dumps({"data": i}),
            file_type="json",
            output_mode="append_log",
            target_directory=empty_directory,
            file_name_prefix="my_file",
            max_entries_per_file=3,
        )
    # flushing buffer
    del block

    # then
    saved_files = sorted(glob(os.path.join(empty_directory, "my_file_*.jsonl")))
    assert len(saved_files) == 2, "Expected 2 separate files"
    with open(saved_files[0]) as f:
        lines = [json.loads(line) for line in f.readlines() if len(line.split()) > 0]
        assert len(lines) == 3, "Expected three entries"
        assert lines == [{"data": 0}, {"data": 1}, {"data": 2}]
    with open(saved_files[1]) as f:
        lines = [json.loads(line) for line in f.readlines() if len(line.split()) > 0]
        assert len(lines) == 2, "Expected two entries"
        assert lines == [{"data": 3}, {"data": 4}]


def test_saving_csv_into_append_log(empty_directory: str) -> None:
    # given
    block = LocalFileSinkBlockV1(
        allow_access_to_file_system=True, allowed_write_directory=None
    )

    # when
    for i in range(5):
        _ = block.run(
            content=pd.DataFrame([{"data": i}]).to_csv(index=False),
            file_type="csv",
            output_mode="append_log",
            target_directory=empty_directory,
            file_name_prefix="my_file",
            max_entries_per_file=3,
        )
    # flushing buffer
    del block

    # then
    saved_files = sorted(glob(os.path.join(empty_directory, "my_file_*.csv")))
    assert len(saved_files) == 2, "Expected 2 separate files"
    assert pd.read_csv(saved_files[0])["data"].tolist() == [0, 1, 2]
    assert pd.read_csv(saved_files[1])["data"].tolist() == [3, 4]


def test_path_is_within_specified_directory_when_relative_paths_given_and_values_match() -> (
    None
):
    # when
    result = path_is_within_specified_directory(
        path="some/other", specified_directory="some"
    )

    # then
    assert result is True


def test_path_is_within_specified_directory_when_relative_paths_given_and_values_do_not_match() -> (
    None
):
    # when
    result = path_is_within_specified_directory(
        path="somes/other", specified_directory="some"
    )

    # then
    assert result is False


def test_path_is_within_specified_directory_when_absolute_paths_given_and_values_match() -> (
    None
):
    # when
    result = path_is_within_specified_directory(
        path="/some/other", specified_directory="/some"
    )

    # then
    assert result is True


def test_path_is_within_specified_directory_when_absolute_paths_given_and_values_do_not_match() -> (
    None
):
    # when
    result = path_is_within_specified_directory(
        path="/somes/other", specified_directory="/some"
    )

    # then
    assert result is False


def test_path_is_within_specified_directory_when_mixed_paths_given_and_values_match() -> (
    None
):
    # when
    result = path_is_within_specified_directory(
        path="some/other",
        specified_directory=os.path.abspath(os.path.curdir),
    )

    # then
    assert result is True


def test_path_is_within_specified_directory_when_mixed_paths_given_and_values_do_not_match() -> (
    None
):
    # when
    result = path_is_within_specified_directory(
        path="../other",
        specified_directory=os.path.abspath(os.path.curdir),
    )

    # then
    assert result is False
