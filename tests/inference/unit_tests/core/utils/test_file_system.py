import json
import os.path

import pytest
from humanfriendly.testing import touch

from inference.core.utils.file_system import (
    dump_bytes,
    dump_json,
    dump_text_lines,
    ensure_parent_dir_exists,
    ensure_write_is_allowed,
    read_json,
    read_text_file,
)


def test_read_json_when_file_does_not_exist(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "non_existing.json")

    # when
    with pytest.raises(FileNotFoundError):
        _ = read_json(path=file_path)


def test_read_json_when_file_is_not_json(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some.json")
    with open(file_path, "w") as f:
        f.write("NOT JSON")

    # when
    with pytest.raises(ValueError):
        _ = read_json(path=file_path)


def test_read_json_when_file_is_json(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some.json")
    with open(file_path, "w") as f:
        json.dump({"some": "value"}, f)

    # when
    result = read_json(path=file_path)

    # then
    assert result == {"some": "value"}


def test_dump_json_when_file_exists(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some.json")
    touch(filename=file_path)

    # when
    with pytest.raises(RuntimeError):
        dump_json(path=file_path, content={"some": "key"})


def test_dump_json_when_file_exists_but_override_allowed(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some.json")
    touch(filename=file_path)

    # when
    dump_json(path=file_path, content={"some": "key"}, allow_override=True, indent=4)
    with open(file_path) as f:
        result = json.load(f)

    # then
    assert result == {"some": "key"}


def test_dump_json_when_parent_dir_does_not_exist(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "dir", "some.json")

    # when
    dump_json(path=file_path, content={"some": "key"}, indent=4)
    with open(file_path) as f:
        result = json.load(f)

    # then
    assert result == {"some": "key"}


def test_dump_json_when_parent_dir_exists(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some.json")

    # when
    dump_json(path=file_path, content={"some": "key"}, indent=4)
    with open(file_path) as f:
        result = json.load(f)

    # then
    assert result == {"some": "key"}


def test_read_text_file_when_file_does_not_exists(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "non_existing.txt")

    # when
    with pytest.raises(FileNotFoundError):
        _ = read_text_file(file_path)


def test_read_text_file_when_file_exists_and_plain_content_to_be_read(
    example_text_file: str,
) -> None:
    # when
    result = read_text_file(example_text_file)

    # then
    assert result == "\nThis is first line\nThis is second line\n"


def test_read_text_file_when_file_exists_and_plain_content_to_be_read_with_stripping(
    example_text_file: str,
) -> None:
    # when
    result = read_text_file(example_text_file, strip_white_chars=True)

    # then
    assert result == "This is first line\nThis is second line"


def test_read_text_file_when_file_exists_and_text_lines_to_be_read(
    example_text_file: str,
) -> None:
    # when
    result = read_text_file(example_text_file, split_lines=True)

    # then
    assert result == ["\n", "This is first line\n", "This is second line\n"]


def test_read_text_file_when_file_exists_and_text_lines_to_be_read_with_stripping(
    example_text_file: str,
) -> None:
    # when
    result = read_text_file(example_text_file, split_lines=True, strip_white_chars=True)

    # then
    assert result == ["This is first line", "This is second line"]


def test_dump_text_lines_when_directory_exists_and_file_does_not_exist(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "file.txt")

    # when
    dump_text_lines(path=file_path, content=["A", "B"])

    # then
    assert_text_file_content_correct(file_path=file_path, content="A\nB")


def test_dump_text_lines_when_directory_exists_and_file_exists(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "file.txt")
    touch(filename=file_path)

    # when
    with pytest.raises(RuntimeError):
        dump_text_lines(path=file_path, content=["A", "B"])

    # then
    assert_text_file_content_correct(file_path=file_path, content="")


def test_dump_text_lines_when_directory_exists_and_file_exists_but_override_allowed(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "file.txt")
    touch(filename=file_path)

    # when
    dump_text_lines(path=file_path, content=["A", "B"], allow_override=True)

    # then
    assert_text_file_content_correct(file_path=file_path, content="A\nB")


def test_dump_text_lines_when_directory_does_not_exist_and_file_does_not_exist(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some", "file.txt")

    # when
    dump_text_lines(path=file_path, content=["A", "B"])

    # then
    assert_text_file_content_correct(file_path=file_path, content="A\nB")


def test_dump_text_lines_when_directory_does_not_exist_and_file_exists(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some", "file.txt")
    touch(filename=file_path)

    # when
    with pytest.raises(RuntimeError):
        dump_text_lines(path=file_path, content=["A", "B"])

    # then
    assert_text_file_content_correct(file_path=file_path, content="")


def test_dump_text_lines_when_directory_does_not_exist_and_file_exists_but_override_allowed(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some", "file.txt")
    touch(filename=file_path)

    # when
    dump_text_lines(path=file_path, content=["A", "B"], allow_override=True)

    # then
    assert_text_file_content_correct(file_path=file_path, content="A\nB")


def test_dump_bytes_when_directory_exists_and_file_does_not_exist(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "file.dat")

    # when
    dump_bytes(path=file_path, content=b"SOME BYTES")

    # then
    assert_bytes_file_content_correct(file_path=file_path, content=b"SOME BYTES")


def test_dump_bytes_when_directory_exists_and_file_exists(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "file.dat")
    touch(filename=file_path)

    # when
    with pytest.raises(RuntimeError):
        dump_bytes(path=file_path, content=b"SOME BYTES")

    # then
    assert_bytes_file_content_correct(file_path=file_path, content=b"")


def test_dump_bytes_when_directory_exists_and_file_exists_but_override_allowed(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "file.dat")
    touch(filename=file_path)

    # when
    dump_bytes(path=file_path, content=b"SOME BYTES", allow_override=True)

    # then
    assert_bytes_file_content_correct(file_path=file_path, content=b"SOME BYTES")


def test_dump_bytes_when_directory_does_not_exist_and_file_does_not_exist(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some", "file.dat")

    # when
    dump_bytes(path=file_path, content=b"SOME BYTES")

    # then
    assert_bytes_file_content_correct(file_path=file_path, content=b"SOME BYTES")


def test_dump_bytes_when_directory_does_not_exist_and_file_exists(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some", "file.dat")
    touch(filename=file_path)

    # when
    with pytest.raises(RuntimeError):
        dump_bytes(path=file_path, content=b"SOME BYTES")

    # then
    assert_bytes_file_content_correct(file_path=file_path, content=b"")


def test_dump_bytes_when_directory_does_not_exist_and_file_exists_but_override_allowed(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "some", "file.dat")
    touch(filename=file_path)

    # when
    dump_bytes(path=file_path, content=b"SOME BYTES", allow_override=True)

    # then
    assert_bytes_file_content_correct(file_path=file_path, content=b"SOME BYTES")


def test_ensure_parent_dir_exists_when_dir_exists(empty_local_dir: str) -> None:
    # given
    path = os.path.join(empty_local_dir, "file.txt")

    # when
    ensure_parent_dir_exists(path=path)

    # then
    assert len(os.listdir(empty_local_dir)) == 0


def test_ensure_parent_dir_exists_when_dir_does_not_exist(empty_local_dir: str) -> None:
    # given
    path = os.path.join(empty_local_dir, "some", "file.txt")

    # when
    ensure_parent_dir_exists(path=path)

    # then
    assert os.listdir(empty_local_dir) == ["some"]


@pytest.mark.parametrize("allow_override", [True, False])
def test_ensure_write_is_allowed_when_file_does_not_exist(
    allow_override: bool,
    empty_local_dir: str,
) -> None:
    # given
    path = os.path.join(empty_local_dir, "file.txt")

    # when
    ensure_write_is_allowed(path=path, allow_override=allow_override)

    # then no exception


def test_ensure_write_is_allowed_when_file_exists_and_override_allowed(
    empty_local_dir: str,
) -> None:
    # given
    path = os.path.join(empty_local_dir, "file.txt")
    touch(path)

    # when
    ensure_write_is_allowed(path=path, allow_override=True)

    # then no exception


def test_ensure_write_is_allowed_when_file_exists_and_override_not_allowed(
    empty_local_dir: str,
) -> None:
    # given
    path = os.path.join(empty_local_dir, "file.txt")
    touch(path)

    # when
    with pytest.raises(RuntimeError):
        ensure_write_is_allowed(path=path, allow_override=False)


def assert_text_file_content_correct(file_path: str, content: str) -> None:
    with open(file_path) as f:
        assert f.read() == content


def assert_bytes_file_content_correct(file_path: str, content: bytes) -> None:
    with open(file_path, "rb") as f:
        assert f.read() == content
