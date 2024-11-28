import json
import os.path

import pytest

from inference_cli.lib.utils import (
    dump_json,
    ensure_target_directory_is_empty,
    read_env_file,
    read_file_lines,
)


def test_read_file_lines(text_file_path: str) -> None:
    # when
    result = read_file_lines(path=text_file_path)

    # then
    assert result == [
        "KEY_1=VALUE_1",
        "KEY_2",
        "KEY_3=VALUE_3",
    ], "Results must contain only non-empty lines from origin files without whitespaces at start and at the end of line"


def test_read_env_file(text_file_path: str) -> None:
    # when
    result = read_env_file(path=text_file_path)

    # then
    assert result == {
        "KEY_1": "VALUE_1",
        "KEY_3": "VALUE_3",
    }, "Decoded value must only contain lines with valid KEY=VALUE format"


def test_dump_json(empty_directory: str) -> None:
    # given
    target_path = os.path.join(empty_directory, "some", "file.json")
    content = {"some": "content"}

    # when
    dump_json(path=target_path, content=content)

    # then
    with open(target_path, "r") as f:
        result = json.load(f)
    assert result == {"some": "content"}


@pytest.mark.parametrize("allow_override", [True, False])
@pytest.mark.parametrize("only_files", [True, False])
def test_ensure_target_directory_is_empty_when_empty_directory_given(
    empty_directory: str,
    allow_override: bool,
    only_files: bool,
) -> None:
    # when
    ensure_target_directory_is_empty(
        output_directory=empty_directory,
        allow_override=allow_override,
        only_files=only_files,
    )

    # then - no errors


def test_ensure_target_directory_is_empty_when_directory_with_sub_dir_provided_but_only_files_matter(
    empty_directory: str,
) -> None:
    # given
    sub_dir_path = os.path.join(empty_directory, "sub_dir")
    os.makedirs(sub_dir_path, exist_ok=True)

    # when
    ensure_target_directory_is_empty(
        output_directory=empty_directory,
        allow_override=False,
        only_files=True,
    )

    # then - no errors


def test_ensure_target_directory_is_empty_when_directory_with_sub_dir_provided_but_not_only_files_matter(
    empty_directory: str,
) -> None:
    # given
    sub_dir_path = os.path.join(empty_directory, "sub_dir")
    os.makedirs(sub_dir_path, exist_ok=True)

    # when
    with pytest.raises(RuntimeError):
        ensure_target_directory_is_empty(
            output_directory=empty_directory,
            allow_override=False,
            only_files=False,
        )


def test_ensure_target_directory_is_empty_when_directory_with_sub_dir_provided_but_not_only_files_matter_and_override_allowed(
    empty_directory: str,
) -> None:
    # given
    sub_dir_path = os.path.join(empty_directory, "sub_dir")
    os.makedirs(sub_dir_path, exist_ok=True)

    # when
    ensure_target_directory_is_empty(
        output_directory=empty_directory,
        allow_override=True,
        only_files=False,
    )

    # then - no errors
