import pytest
from inference_exp.entities import ImageDimensions
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.common.roboflow.model_packages import (
    ModelCharacteristics,
    PreProcessingConfig,
    PreProcessingMode,
    TRTConfig,
    parse_class_map_from_environment_file,
    parse_class_names_file,
    parse_key_points_metadata,
    parse_model_characteristics,
    parse_pre_processing_config,
    parse_trt_config,
)


def test_parse_class_map_from_environment_file_when_file_not_exists() -> None:
    with pytest.raises(CorruptedModelPackageError):
        parse_class_map_from_environment_file(
            environment_file_path="/some/invalid/file.json"
        )


def test_parse_class_map_from_environment_file_when_file_is_not_a_valid_json(
    not_a_json_file_path: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        parse_class_map_from_environment_file(
            environment_file_path=not_a_json_file_path
        )


def test_parse_class_map_from_environment_file_when_file_contains_json_but_not_dict_object(
    json_without_dict_path: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        parse_class_map_from_environment_file(
            environment_file_path=json_without_dict_path
        )


def test_parse_class_map_from_environment_file_when_file_does_not_have_class_map(
    env_without_class_map: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        parse_class_map_from_environment_file(
            environment_file_path=env_without_class_map
        )


def test_parse_class_map_from_environment_file_when_file_is_valid(
    env_with_class_map: str,
) -> None:
    # when
    result = parse_class_map_from_environment_file(
        environment_file_path=env_with_class_map
    )

    # then
    assert result == [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
    ]


def test_parse_class_map_from_environment_file_when_file_contains_malformed_class_map(
    env_with_malformed_class_map: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_class_map_from_environment_file(
            environment_file_path=env_with_malformed_class_map
        )


def test_parse_class_map_from_environment_file_when_file_contains_malformed_class_map_index(
    env_with_malformed_class_map_index: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_class_map_from_environment_file(
            environment_file_path=env_with_malformed_class_map_index
        )


def test_parse_trt_config_when_invalid_path_provided() -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_trt_config(config_path="/invalid/path.json")


def test_parse_trt_config_when_not_a_json_file_provided(
    not_a_json_file_path: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_trt_config(config_path=not_a_json_file_path)


def test_parse_trt_config_when_json_file_without_dict_object_provided(
    json_without_dict_path: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_trt_config(config_path=json_without_dict_path)


def test_parse_trt_config_when_empty_config_provided(empty_trt_config: str) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_trt_config(config_path=empty_trt_config)


def test_parse_trt_config_when_invalid_static_bs_provided(
    trt_config_invalid_static_bs: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_trt_config(config_path=trt_config_invalid_static_bs)


def test_parse_trt_config_when_dynamic_bs_min_is_below_zero(
    trt_config_invalid_dynamic_bs_min_below_zero: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_trt_config(config_path=trt_config_invalid_dynamic_bs_min_below_zero)


def test_parse_trt_config_when_dynamic_bs_min_above_max(
    trt_config_invalid_dynamic_bs_min_above_max: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_trt_config(config_path=trt_config_invalid_dynamic_bs_min_above_max)


def test_parse_trt_config_when_dynamic_bs_opt_below_min(
    trt_config_invalid_dynamic_bs_opt_below_min: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_trt_config(config_path=trt_config_invalid_dynamic_bs_opt_below_min)


def test_parse_trt_config_when_dynamic_bs_opt_above_max(
    trt_config_invalid_dynamic_bs_opt_above_max: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_trt_config(config_path=trt_config_invalid_dynamic_bs_opt_above_max)


def test_parse_trt_config_when_valid_dynamic_bs_config_provided(
    valid_trt_config_with_dynamic_bs: str,
) -> None:
    # when
    result = parse_trt_config(config_path=valid_trt_config_with_dynamic_bs)

    # then
    assert result == TRTConfig(
        static_batch_size=None,
        dynamic_batch_size_min=1,
        dynamic_batch_size_opt=2,
        dynamic_batch_size_max=8,
    )


def test_parse_key_points_metadata_when_file_not_available() -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path="/non/existing/file.json"
        )


def test_parse_key_points_metadata_when_not_a_json_file_provided(
    not_a_json_file_path: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(key_points_metadata_path=not_a_json_file_path)


def test_parse_key_points_metadata_when_empty_file_provided(
    empty_key_points_metadata: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path=empty_key_points_metadata
        )


def test_parse_key_points_metadata_when_content_is_not_list(
    key_points_metadata_which_is_not_a_list: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path=key_points_metadata_which_is_not_a_list
        )


def test_parse_key_points_metadata_when_valid_input_provided(
    valid_key_points_metadata: str,
) -> None:
    # when
    result = parse_key_points_metadata(
        key_points_metadata_path=valid_key_points_metadata
    )

    # then
    assert result == [
        ["shoulder-right", "shoulder-left", "neck"],
        [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
    ]


def test_parse_key_points_metadata_when_object_class_id_not_available(
    key_points_metadata_without_object_class_id: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path=key_points_metadata_without_object_class_id
        )


def test_parse_key_points_metadata_when_invalid_object_class_id_provided(
    key_points_metadata_with_invalid_object_class_id: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path=key_points_metadata_with_invalid_object_class_id
        )


def test_parse_key_points_metadata_without_key_points(
    key_points_metadata_without_key_points: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path=key_points_metadata_without_key_points
        )


def test_parse_key_points_metadata_with_missing_classes(
    key_points_metadata_with_missing_classes: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path=key_points_metadata_with_missing_classes
        )


def test_parse_model_characteristics_when_file_not_found() -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_model_characteristics(config_path="/not/existing.json")


def test_parse_model_characteristics_when_config_is_not_a_json_file(
    not_a_json_file_path: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_model_characteristics(config_path=not_a_json_file_path)


def test_parse_model_characteristics_when_project_task_type_missing(
    model_characteristics_with_task_type_missing: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_model_characteristics(
            config_path=model_characteristics_with_task_type_missing
        )


def test_parse_model_characteristics_when_model_type_missing(
    model_characteristics_with_model_type_missing: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_model_characteristics(
            config_path=model_characteristics_with_model_type_missing
        )


def test_parse_model_characteristics_when_valid_config_provided(
    valid_model_characteristics: str,
) -> None:
    # when
    result = parse_model_characteristics(config_path=valid_model_characteristics)

    # then
    assert result == ModelCharacteristics(
        task_type="object-detection",
        model_type="yolov8",
    )


def test_parse_pre_processing_config_when_invalid_path_provided() -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_pre_processing_config(config_path="/some/invalid.json")


def test_parse_pre_processing_config_when_not_a_json_file_provided(
    not_a_json_file_path: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_pre_processing_config(config_path=not_a_json_file_path)


def test_parse_pre_processing_config_not_a_dict_payload_provided(
    empty_environment: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_pre_processing_config(config_path=empty_environment)


def test_parse_pre_processing_config_when_pre_processing_config_is_missing(
    environment_without_preprocessing: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_pre_processing_config(config_path=environment_without_preprocessing)


def test_parse_pre_processing_config_when_not_a_json_preprocess_config_provided(
    environment_with_not_a_json_preprocessing: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_pre_processing_config(
            config_path=environment_with_not_a_json_preprocessing
        )


def test_parse_pre_processing_config_when_valid_environment_config_provided_with_stretch(
    valid_environment_stretch: str,
) -> None:
    # when
    result = parse_pre_processing_config(config_path=valid_environment_stretch)

    # then
    assert result == PreProcessingConfig(
        mode=PreProcessingMode.STRETCH,
        target_size=ImageDimensions(width=560, height=560),
        padding_value=None,
    )


def test_parse_pre_processing_config_when_valid_environment_config_provided_with_letterbox(
    valid_environment_letterbox: str,
) -> None:
    # when
    result = parse_pre_processing_config(config_path=valid_environment_letterbox)

    # then
    assert result == PreProcessingConfig(
        mode=PreProcessingMode.LETTERBOX,
        target_size=ImageDimensions(width=560, height=560),
        padding_value=127,
    )


def test_parse_pre_processing_config_when_valid_environment_config_provided_without_resize(
    valid_environment_no_resize: str,
) -> None:
    # when
    result = parse_pre_processing_config(config_path=valid_environment_no_resize)

    # then
    assert result == PreProcessingConfig(
        mode=PreProcessingMode.NONE,
        target_size=None,
        padding_value=None,
    )


def test_parse_pre_processing_config_when_resize_without_dimensions_provided(
    environment_resize_without_dimensions: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_pre_processing_config(
            config_path=environment_resize_without_dimensions
        )


def test_parse_pre_processing_config_when_resize_with_invalid_dimensions_provided(
    environment_resize_invalid_dimensions: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_pre_processing_config(
            config_path=environment_resize_invalid_dimensions
        )


def test_parse_pre_processing_config_when_resize_without_format_provided(
    environment_resize_without_format: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_pre_processing_config(config_path=environment_resize_without_format)


def test_parse_pre_processing_config_when_resize_with_invalid_format_provided(
    environment_resize_invalid_format: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_pre_processing_config(config_path=environment_resize_invalid_format)


def test_parse_class_names_file_when_invalid_path_provided() -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_class_names_file(class_names_path="/invalid/file.txt")


def test_parse_class_names_file_when_empty_config_provided(
    class_names_empty: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_class_names_file(class_names_path=class_names_empty)


def test_parse_class_names_file_when_valid_config_provided(
    class_names_valid: str,
) -> None:
    # when
    result = parse_class_names_file(class_names_path=class_names_valid)

    # then
    assert result == ["some", "other", "yet-another"]
