import json
import os
import os.path
import tempfile

import pytest
from humanfriendly.testing import touch

from inference.core.utils.file_system import (
    AtomicPath,
    dump_bytes,
    dump_bytes_atomic,
    dump_json,
    dump_json_atomic,
    dump_text_lines,
    dump_text_lines_atomic,
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


# Tests for AtomicPath context manager


def test_atomic_path_successful_write(empty_local_dir: str) -> None:
    # given
    target_path = os.path.join(empty_local_dir, "test.txt")
    content = "test content"

    # when
    with AtomicPath(target_path) as temp_path:
        assert temp_path != target_path
        assert os.path.dirname(temp_path) == os.path.dirname(target_path)
        with open(temp_path, "w") as f:
            f.write(content)

    # then
    assert os.path.exists(target_path)
    assert not os.path.exists(temp_path)
    with open(target_path) as f:
        assert f.read() == content


def test_atomic_path_cleans_up_on_exception(empty_local_dir: str) -> None:
    # given
    target_path = os.path.join(empty_local_dir, "test.txt")
    temp_path_ref = None

    # when
    try:
        with AtomicPath(target_path) as temp_path:
            temp_path_ref = temp_path
            assert os.path.exists(temp_path)
            with open(temp_path, "w") as f:
                f.write("partial content")
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # then
    assert not os.path.exists(target_path)
    assert not os.path.exists(temp_path_ref)


def test_atomic_path_with_existing_file_no_override(empty_local_dir: str) -> None:
    # given
    target_path = os.path.join(empty_local_dir, "test.txt")
    touch(target_path)

    # when/then
    with pytest.raises(RuntimeError):
        with AtomicPath(target_path, allow_override=False) as temp_path:
            pass


def test_atomic_path_with_existing_file_override(empty_local_dir: str) -> None:
    # given
    target_path = os.path.join(empty_local_dir, "test.txt")
    original_content = "original"
    new_content = "new content"

    with open(target_path, "w") as f:
        f.write(original_content)

    # when
    with AtomicPath(target_path, allow_override=True) as temp_path:
        with open(temp_path, "w") as f:
            f.write(new_content)

    # then
    with open(target_path) as f:
        assert f.read() == new_content


def test_atomic_path_creates_parent_dirs(empty_local_dir: str) -> None:
    # given
    target_path = os.path.join(empty_local_dir, "subdir", "test.txt")
    content = "test content"

    # when
    with AtomicPath(target_path) as temp_path:
        with open(temp_path, "w") as f:
            f.write(content)

    # then
    assert os.path.exists(target_path)
    with open(target_path) as f:
        assert f.read() == content


# Tests for dump_json_atomic


def test_dump_json_atomic_when_file_does_not_exist(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.json")
    content = {"key": "value", "number": 42}

    # when
    dump_json_atomic(path=file_path, content=content)

    # then
    assert os.path.exists(file_path)
    with open(file_path) as f:
        assert json.load(f) == content


def test_dump_json_atomic_when_file_exists_no_override(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.json")
    touch(file_path)

    # when/then
    with pytest.raises(RuntimeError):
        dump_json_atomic(path=file_path, content={"key": "value"}, allow_override=False)


def test_dump_json_atomic_when_file_exists_with_override(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.json")
    original_content = {"old": "data"}
    new_content = {"new": "data"}

    with open(file_path, "w") as f:
        json.dump(original_content, f)

    # when
    dump_json_atomic(path=file_path, content=new_content, allow_override=True)

    # then
    with open(file_path) as f:
        assert json.load(f) == new_content


def test_dump_json_atomic_with_indent(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.json")
    content = {"key": "value", "nested": {"inner": "data"}}

    # when
    dump_json_atomic(path=file_path, content=content, indent=2)

    # then
    with open(file_path) as f:
        file_content = f.read()
        assert "  " in file_content  # Check indentation exists
        assert json.loads(file_content) == content


def test_dump_json_atomic_creates_parent_dir(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "subdir", "test.json")
    content = {"key": "value"}

    # when
    dump_json_atomic(path=file_path, content=content)

    # then
    assert os.path.exists(file_path)
    with open(file_path) as f:
        assert json.load(f) == content


# Tests for dump_text_lines_atomic


def test_dump_text_lines_atomic_when_file_does_not_exist(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.txt")
    content = ["line1", "line2", "line3"]

    # when
    dump_text_lines_atomic(path=file_path, content=content)

    # then
    assert os.path.exists(file_path)
    with open(file_path) as f:
        assert f.read() == "line1\nline2\nline3"


def test_dump_text_lines_atomic_when_file_exists_no_override(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.txt")
    touch(file_path)

    # when/then
    with pytest.raises(RuntimeError):
        dump_text_lines_atomic(path=file_path, content=["line1"], allow_override=False)


def test_dump_text_lines_atomic_when_file_exists_with_override(
    empty_local_dir: str,
) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.txt")
    with open(file_path, "w") as f:
        f.write("original content")

    new_content = ["new", "lines"]

    # when
    dump_text_lines_atomic(path=file_path, content=new_content, allow_override=True)

    # then
    with open(file_path) as f:
        assert f.read() == "new\nlines"


def test_dump_text_lines_atomic_with_custom_connector(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.txt")
    content = ["line1", "line2", "line3"]

    # when
    dump_text_lines_atomic(path=file_path, content=content, lines_connector="|")

    # then
    with open(file_path) as f:
        assert f.read() == "line1|line2|line3"


def test_dump_text_lines_atomic_creates_parent_dir(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "subdir", "test.txt")
    content = ["line1", "line2"]

    # when
    dump_text_lines_atomic(path=file_path, content=content)

    # then
    assert os.path.exists(file_path)
    with open(file_path) as f:
        assert f.read() == "line1\nline2"


# Tests for dump_bytes_atomic


def test_dump_bytes_atomic_when_file_does_not_exist(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.bin")
    content = b"binary content \x00\x01\x02"

    # when
    dump_bytes_atomic(path=file_path, content=content)

    # then
    assert os.path.exists(file_path)
    with open(file_path, "rb") as f:
        assert f.read() == content


def test_dump_bytes_atomic_when_file_exists_no_override(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.bin")
    touch(file_path)

    # when/then
    with pytest.raises(RuntimeError):
        dump_bytes_atomic(path=file_path, content=b"data", allow_override=False)


def test_dump_bytes_atomic_when_file_exists_with_override(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "test.bin")
    with open(file_path, "wb") as f:
        f.write(b"original data")

    new_content = b"new binary data"

    # when
    dump_bytes_atomic(path=file_path, content=new_content, allow_override=True)

    # then
    with open(file_path, "rb") as f:
        assert f.read() == new_content


def test_dump_bytes_atomic_creates_parent_dir(empty_local_dir: str) -> None:
    # given
    file_path = os.path.join(empty_local_dir, "subdir", "test.bin")
    content = b"binary content"

    # when
    dump_bytes_atomic(path=file_path, content=content)

    # then
    assert os.path.exists(file_path)
    with open(file_path, "rb") as f:
        assert f.read() == content


# Test atomicity of operations


def test_atomic_write_maintains_original_on_error(empty_local_dir: str) -> None:
    """Test that original file is preserved if write fails partway through"""
    # given
    file_path = os.path.join(empty_local_dir, "test.txt")
    original_content = "original content that should be preserved"

    with open(file_path, "w") as f:
        f.write(original_content)

    # when - simulate a write error by mocking
    class WriteError(Exception):
        pass

    try:
        with AtomicPath(file_path, allow_override=True) as temp_path:
            with open(temp_path, "w") as f:
                f.write("partial new content")
                # Simulate error partway through write
                raise WriteError("Simulated write failure")
    except WriteError:
        pass

    # then - original file should be unchanged
    assert os.path.exists(file_path)
    with open(file_path) as f:
        assert f.read() == original_content


def test_atomic_operations_concurrent_safety(empty_local_dir: str) -> None:
    """Test that temp files don't collide when multiple atomic writes happen"""
    # given
    target_path = os.path.join(empty_local_dir, "test.txt")

    # when - create multiple atomic writes to same target
    temp_paths = []
    contexts = []

    for i in range(3):
        ctx = AtomicPath(target_path, allow_override=True)
        temp_path = ctx.__enter__()
        temp_paths.append(temp_path)
        contexts.append(ctx)

    # then - all temp paths should be unique
    assert len(set(temp_paths)) == 3

    # cleanup
    for ctx, temp_path in zip(contexts, temp_paths):
        try:
            ctx.__exit__(None, None, None)
        except:
            pass
        if os.path.exists(temp_path):
            os.unlink(temp_path)
