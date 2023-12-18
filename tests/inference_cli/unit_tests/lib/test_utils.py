from inference_cli.lib.utils import read_file_lines, read_env_file


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
