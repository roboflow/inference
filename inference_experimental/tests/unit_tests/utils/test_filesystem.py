import json
import os.path
from json import JSONDecodeError

import pytest
from inference_exp.utils.file_system import (
    dump_json,
    ensure_parent_dir_exists,
    pre_allocate_file,
    read_json,
    remove_file_if_exists,
    stream_file_lines,
)


def test_stream_file_lines_when_non_existing_file_selected() -> None:
    # given
    generator = stream_file_lines("/some/non/existing/file.txt")

    # when
    with pytest.raises(FileNotFoundError):
        _ = list(generator)


def test_stream_file_lines_when_existing_non_text_file_selected(
    binary_file: str,
) -> None:
    # given
    generator = stream_file_lines(binary_file)

    # when
    with pytest.raises(UnicodeDecodeError):
        _ = list(generator)


def test_stream_file_lines_when_existing_empty_text_file_selected(
    empty_text_file: str,
) -> None:
    # given
    generator = stream_file_lines(empty_text_file)

    # when
    result = list(generator)

    # then
    assert result == [], "Expected empty list as no lines in the file"


def test_stream_file_lines_when_existing_non_empty_text_file_selected(
    non_empty_text_file: str,
) -> None:
    # given
    generator = stream_file_lines(non_empty_text_file)

    # when
    result = list(generator)

    # then
    assert result == [
        "This is the first line",
        "This is the second line",
    ], "Expected the exact content"


def test_read_json_from_non_existing_file() -> None:
    # when
    with pytest.raises(FileNotFoundError):
        _ = read_json(path="/some/non/existing/file.json")


def test_read_json_from_invalid_json_file(non_empty_text_file: str) -> None:
    # when
    with pytest.raises(JSONDecodeError):
        _ = read_json(path=non_empty_text_file)


def test_read_json_from_valid_json_file(valid_json_file: str) -> None:
    # when
    result = read_json(path=valid_json_file)

    # then
    assert result == {"some": "value"}, "Expected the exact content"


def test_pre_allocate_file_in_existing_dir(empty_local_dir: str) -> None:
    # given
    path = os.path.join(empty_local_dir, "some.txt")

    # when
    pre_allocate_file(path=path, file_size=1024)

    # then
    assert os.path.exists(path), "Expected file to exists"
    assert (
        os.path.getsize(path) == 1024
    ), "Expected file to have the exact size of 1024 bytes"


def test_pre_allocate_file_in_non_existing_dir(empty_local_dir: str) -> None:
    # given
    path = os.path.join(empty_local_dir, "my_sub_dir", "some.txt")

    # when
    pre_allocate_file(path=path, file_size=1024)

    # then
    assert os.path.exists(path), "Expected file to exists"
    assert (
        os.path.getsize(path) == 1024
    ), "Expected file to have the exact size of 1024 bytes"


def test_ensure_parent_dir_exists_when_it_exists(empty_local_dir: str) -> None:
    # given
    parent_dir = os.path.join(empty_local_dir, "sub-dir")
    file_path = os.path.join(parent_dir, "my-file.txt")
    os.makedirs(parent_dir, exist_ok=True)

    # when
    ensure_parent_dir_exists(path=file_path)

    # then
    assert os.path.isdir(parent_dir), "Expected parent dir to exists"
    assert list(os.listdir(empty_local_dir)) == [
        "sub-dir"
    ], "Only single sub-dir should be created"


def test_ensure_parent_dir_exists_when_it_does_not_exist(empty_local_dir: str) -> None:
    # given
    parent_dir = os.path.join(empty_local_dir, "sub-dir")
    file_path = os.path.join(parent_dir, "my-file.txt")

    # when
    ensure_parent_dir_exists(path=file_path)

    # then
    assert os.path.isdir(parent_dir), "Expected parent dir to exists"
    assert list(os.listdir(empty_local_dir)) == [
        "sub-dir"
    ], "Only single sub-dir should be created"


def test_remove_file_if_exists_when_file_does_not_exist(empty_local_dir: str) -> None:
    # given
    some_file = os.path.join(empty_local_dir, "non-existing.txt")

    # when
    remove_file_if_exists(path=some_file)

    # then - no error


def test_remove_file_if_exists_when_existing_dir_name_provided(
    empty_local_dir: str,
) -> None:
    # given
    some_dir = os.path.join(empty_local_dir, "some-dir")
    os.makedirs(some_dir, exist_ok=True)

    # when
    remove_file_if_exists(path=some_dir)

    # then
    assert os.path.isdir(
        some_dir
    ), "Dir should not be removed by function to remove file"


def test_remove_file_if_exists_when_existing_file(non_empty_local_dir: str) -> None:
    # given
    path = os.path.join(non_empty_local_dir, "some.txt")
    assert os.path.isfile(path)

    # when
    remove_file_if_exists(path=path)

    # then
    assert not os.path.exists(path), "File should be removed"
    assert os.path.exists(
        os.path.join(non_empty_local_dir, "sub_dir")
    ), "Additional directory content should be left as is"


def test_dump_json(empty_local_dir: str) -> None:
    # given
    target_path = os.path.join(empty_local_dir, "example.json")

    # when
    dump_json(
        path=target_path,
        content={"some": "value"},
    )

    # then
    with open(target_path) as f:
        result = json.load(f)
    assert result == {"some": "value"}
