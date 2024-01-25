import json
import os.path

from inference_cli.lib.utils import dump_json, read_env_file, read_file_lines


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
