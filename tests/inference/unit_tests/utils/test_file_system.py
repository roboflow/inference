import json
import os.path

import pytest
from humanfriendly.testing import touch

from inference.core.utils.file_system import read_json, dump_json


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
