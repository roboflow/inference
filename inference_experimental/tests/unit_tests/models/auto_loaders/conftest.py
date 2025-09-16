import os.path
import tempfile
from typing import Generator

import pytest

TEST_MODULES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_modules")
)


@pytest.fixture
def example_module_path() -> str:
    return os.path.join(TEST_MODULES_DIR, "example_module.py")


@pytest.fixture
def example_broken_module_path() -> str:
    return os.path.join(TEST_MODULES_DIR, "broken_module.py")


@pytest.fixture
def example_non_python_file_path() -> str:
    return os.path.join(TEST_MODULES_DIR, "example_non_python_file.txt")


@pytest.fixture
def example_model_package_dir() -> str:
    return os.path.join(TEST_MODULES_DIR, "example_model")


@pytest.fixture
def not_a_json_file_config_path() -> str:
    return os.path.join(
        TEST_MODULES_DIR, "example_model_configs", "not_a_json_config.txt"
    )


@pytest.fixture
def not_a_dict_inside_config_path() -> str:
    return os.path.join(
        TEST_MODULES_DIR, "example_model_configs", "not_a_dict_inside.json"
    )


@pytest.fixture
def unknown_backend_config_path() -> str:
    return os.path.join(
        TEST_MODULES_DIR, "example_model_configs", "model_config_unknown_backend.json"
    )


@pytest.fixture
def full_config_path() -> str:
    return os.path.join(TEST_MODULES_DIR, "example_model_configs", "full_config.json")


@pytest.fixture(scope="function")
def empty_local_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir
