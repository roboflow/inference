import json
import os.path
import tempfile
from typing import Generator

import pytest


@pytest.fixture
def existing_module_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "lazy_class_test_package", "valid.py")
    )


@pytest.fixture
def non_existing_module_path() -> str:
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "lazy_class_test_package", "non_existing.py"
        )
    )


@pytest.fixture(scope="function")
def empty_local_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="function")
def non_empty_local_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        _create_file(target_dir=tmp_dir, file_name="some.txt")
        sub_dir = os.path.join(tmp_dir, "sub_dir")
        os.makedirs(sub_dir, exist_ok=True)
        _create_file(target_dir=sub_dir, file_name="other.txt")
        yield tmp_dir


@pytest.fixture(scope="function")
def binary_file() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "file.bin")
        with open(file_path, "wb") as f:
            f.write(b"\xf8\xde\x0a\x97\x46\x0c\x0f\x3f\x7b\x59")
        yield file_path


@pytest.fixture(scope="function")
def empty_text_file() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "file.bin")
        with open(file_path, "w") as f:
            f.write("")
        yield file_path


@pytest.fixture(scope="function")
def non_empty_text_file() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "file.bin")
        with open(file_path, "w") as f:
            f.write("This is the first line\n")
            f.write("\n")
            f.write("This is the second line\n")
            f.write("\n")
            f.write("\n")
        yield file_path


@pytest.fixture(scope="function")
def valid_json_file() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "file.json")
        with open(file_path, "w") as f:
            json.dump({"some": "value"}, f)
        yield file_path


def _create_file(target_dir: str, file_name: str) -> None:
    with open(os.path.join(target_dir, file_name), "w") as f:
        f.write("You can't handle the truth!")
