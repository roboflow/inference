import os.path

import pytest

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))


@pytest.fixture
def not_a_json_file_path() -> str:
    return os.path.join(ASSETS_DIR, "not_a_json_file.txt")


@pytest.fixture
def json_without_dict_path() -> str:
    return os.path.join(ASSETS_DIR, "json_without_dict.json")


@pytest.fixture
def empty_trt_config() -> str:
    return os.path.join(ASSETS_DIR, "empty_trt_config.json")


@pytest.fixture
def env_without_class_map() -> str:
    return os.path.join(ASSETS_DIR, "env_without_class_map.json")


@pytest.fixture
def env_with_class_map() -> str:
    return os.path.join(ASSETS_DIR, "env_with_class_map.json")


@pytest.fixture
def env_with_malformed_class_map() -> str:
    return os.path.join(ASSETS_DIR, "env_with_malformed_class_map.json")


@pytest.fixture
def env_with_malformed_class_map_index() -> str:
    return os.path.join(ASSETS_DIR, "env_with_malformed_class_map_index.json")


@pytest.fixture
def trt_config_invalid_static_bs() -> str:
    return os.path.join(ASSETS_DIR, "trt_config_invalid_static_bs.json")


@pytest.fixture
def trt_config_invalid_dynamic_bs_min_below_zero() -> str:
    return os.path.join(ASSETS_DIR, "trt_config_invalid_dynamic_bs_min_below_zero.json")


@pytest.fixture
def trt_config_invalid_dynamic_bs_min_above_max() -> str:
    return os.path.join(ASSETS_DIR, "trt_config_invalid_dynamic_bs_min_above_max.json")


@pytest.fixture
def trt_config_invalid_dynamic_bs_opt_below_min() -> str:
    return os.path.join(ASSETS_DIR, "trt_config_invalid_dynamic_bs_opt_below_min.json")


@pytest.fixture
def trt_config_invalid_dynamic_bs_opt_above_max() -> str:
    return os.path.join(ASSETS_DIR, "trt_config_invalid_dynamic_bs_opt_above_max.json")


@pytest.fixture
def valid_trt_config_with_dynamic_bs() -> str:
    return os.path.join(ASSETS_DIR, "valid_trt_config_with_dynamic_bs.json")
