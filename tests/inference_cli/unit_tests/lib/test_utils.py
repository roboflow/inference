import json
import os.path
from pathlib import Path
from typing import Callable, List
from unittest import mock

import pytest

from inference_cli.lib.utils import (
    IMAGES_EXTENSIONS,
    dump_json,
    ensure_target_directory_is_empty,
    get_all_images_in_directory,
    get_all_videos_in_directory,
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


def test_get_all_images_in_directory(empty_directory: str) -> None:
    # given
    for extension in IMAGES_EXTENSIONS:
        _create_empty_file(directory=empty_directory, file_name=f"image.{extension}")
    _create_empty_file(directory=empty_directory, file_name=f".tmp")
    _create_empty_file(directory=empty_directory, file_name=f".bin")
    expected_files = len(os.listdir(empty_directory)) - 2

    # when
    result = get_all_images_in_directory(input_directory=empty_directory)

    # then
    assert len(result) == expected_files


@pytest.mark.parametrize(
    "list_directory,file_names",
    [
        (get_all_images_in_directory, ["IMG_0001.JPG", "Frame_B.png", "lower.jpg"]),
        (get_all_videos_in_directory, ["CLIP_0001.MP4", "Take_B.mov", "lower.mp4"]),
    ],
)
def test_directory_listing_preserves_path_case(
    tmp_path: Path,
    list_directory: Callable[[str], List[str]],
    file_names: List[str],
) -> None:
    # given
    directory = tmp_path / "Shots"
    directory.mkdir()
    for file_name in file_names:
        (directory / file_name).touch()

    # when
    result = list_directory(input_directory=str(directory))

    # then
    assert sorted(result) == sorted(
        (directory / file_name).as_posix() for file_name in file_names
    )
    assert all(os.path.exists(path) for path in result)


@pytest.mark.parametrize(
    "list_directory",
    [get_all_images_in_directory, get_all_videos_in_directory],
)
def test_directory_listing_deduplicates_discovered_paths_keeping_order(
    list_directory: Callable[[str], List[str]],
) -> None:
    # given
    first_path, second_path = Path("Shots/Second.JPG"), Path("Shots/First.JPG")

    # when
    with mock.patch(
        "inference_cli.lib.utils.sv.list_files_with_extensions",
        return_value=[first_path, second_path, first_path],
    ):
        result = list_directory(input_directory="Shots")

    # then
    assert result == [first_path.as_posix(), second_path.as_posix()]


@pytest.mark.parametrize(
    "list_directory,upper_name,lower_name",
    [
        (get_all_images_in_directory, "FRAME.PNG", "frame.png"),
        (get_all_videos_in_directory, "CLIP.MP4", "clip.mp4"),
    ],
)
def test_directory_listing_keeps_case_distinct_files(
    tmp_path: Path,
    list_directory: Callable[[str], List[str]],
    upper_name: str,
    lower_name: str,
) -> None:
    # given
    upper_path, lower_path = tmp_path / upper_name, tmp_path / lower_name
    upper_path.touch()
    if lower_path.exists():
        pytest.skip("The filesystem does not support case-distinct file names")
    lower_path.touch()

    # when
    result = list_directory(input_directory=str(tmp_path))

    # then
    assert sorted(result) == sorted([upper_path.as_posix(), lower_path.as_posix()])


def _create_empty_file(directory: str, file_name: str) -> None:
    file_path = os.path.join(directory, file_name)
    path = Path(file_path)
    path.touch(exist_ok=True)
