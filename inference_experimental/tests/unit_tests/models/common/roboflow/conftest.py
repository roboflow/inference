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


@pytest.fixture
def empty_key_points_metadata() -> str:
    return os.path.join(ASSETS_DIR, "empty_key_points_metadata.json")


@pytest.fixture
def key_points_metadata_which_is_not_a_list() -> str:
    return os.path.join(ASSETS_DIR, "key_points_metadata_which_is_not_a_list.json")


@pytest.fixture
def valid_key_points_metadata() -> str:
    return os.path.join(ASSETS_DIR, "valid_key_points_metadata.json")


@pytest.fixture
def key_points_metadata_without_object_class_id() -> str:
    return os.path.join(ASSETS_DIR, "key_points_metadata_without_object_class_id.json")


@pytest.fixture
def key_points_metadata_with_invalid_object_class_id() -> str:
    return os.path.join(
        ASSETS_DIR, "key_points_metadata_with_invalid_object_class_id.json"
    )


@pytest.fixture
def key_points_metadata_without_key_points() -> str:
    return os.path.join(ASSETS_DIR, "key_points_metadata_without_key_points.json")


@pytest.fixture
def key_points_metadata_with_missing_classes() -> str:
    return os.path.join(ASSETS_DIR, "key_points_metadata_with_missing_classes.json")


@pytest.fixture
def key_points_metadata_with_missing_edges() -> str:
    return os.path.join(ASSETS_DIR, "key_points_metadata_with_missing_edges.json")


@pytest.fixture
def key_points_metadata_with_malformed_edges() -> str:
    return os.path.join(ASSETS_DIR, "key_points_metadata_with_malformed_edges.json")


@pytest.fixture
def key_points_metadata_with_edges_pointing_non_existing_class() -> str:
    return os.path.join(
        ASSETS_DIR, "key_points_metadata_with_edges_pointing_non_existing_class.json"
    )


@pytest.fixture
def valid_model_characteristics() -> str:
    return os.path.join(ASSETS_DIR, "valid_model_characteristics.json")


@pytest.fixture
def model_characteristics_with_model_type_missing() -> str:
    return os.path.join(
        ASSETS_DIR, "model_characteristics_with_model_type_missing.json"
    )


@pytest.fixture
def model_characteristics_with_task_type_missing() -> str:
    return os.path.join(ASSETS_DIR, "model_characteristics_with_task_type_missing.json")


@pytest.fixture
def empty_environment() -> str:
    return os.path.join(ASSETS_DIR, "empty_environment.json")


@pytest.fixture
def environment_without_preprocessing() -> str:
    return os.path.join(ASSETS_DIR, "environment_without_preprocessing.json")


@pytest.fixture
def environment_with_not_a_json_preprocessing() -> str:
    return os.path.join(ASSETS_DIR, "environment_with_not_a_json_preprocessing.json")


@pytest.fixture
def valid_environment_stretch() -> str:
    return os.path.join(ASSETS_DIR, "valid_environment_stretch.json")


@pytest.fixture
def valid_environment_letterbox() -> str:
    return os.path.join(ASSETS_DIR, "valid_environment_letterbox.json")


@pytest.fixture
def valid_environment_no_resize() -> str:
    return os.path.join(ASSETS_DIR, "valid_environment_no_resize.json")


@pytest.fixture
def environment_resize_without_dimensions() -> str:
    return os.path.join(ASSETS_DIR, "environment_resize_without_dimensions.json")


@pytest.fixture
def environment_resize_invalid_dimensions() -> str:
    return os.path.join(ASSETS_DIR, "environment_resize_invalid_dimensions.json")


@pytest.fixture
def environment_resize_without_format() -> str:
    return os.path.join(ASSETS_DIR, "environment_resize_without_format.json")


@pytest.fixture
def environment_resize_invalid_format() -> str:
    return os.path.join(ASSETS_DIR, "environment_resize_invalid_format.json")


@pytest.fixture
def class_names_empty() -> str:
    return os.path.join(ASSETS_DIR, "class_names_empty.txt")


@pytest.fixture
def class_names_valid() -> str:
    return os.path.join(ASSETS_DIR, "class_names_valid.txt")


@pytest.fixture
def inference_config_invalid_image_pre_processing() -> str:
    return os.path.join(
        ASSETS_DIR, "inference_config_invalid_image_pre_processing.json"
    )


@pytest.fixture
def inference_config_invalid_network_input() -> str:
    return os.path.join(ASSETS_DIR, "inference_config_invalid_network_input.json")


@pytest.fixture
def inference_config_missing_network_input() -> str:
    return os.path.join(ASSETS_DIR, "inference_config_missing_network_input.json")


@pytest.fixture
def inference_config_invalid_forward_pass_config() -> str:
    return os.path.join(ASSETS_DIR, "inference_config_invalid_forward_pass_config.json")


@pytest.fixture
def inference_config_invalid_post_processing_config() -> str:
    return os.path.join(
        ASSETS_DIR, "inference_config_invalid_post_processing_config.json"
    )


@pytest.fixture
def inference_config_invalid_class_names_operations() -> str:
    return os.path.join(
        ASSETS_DIR, "inference_config_invalid_class_names_operations.json"
    )


@pytest.fixture
def inference_config_valid_config() -> str:
    return os.path.join(ASSETS_DIR, "inference_config_valid_config.json")
