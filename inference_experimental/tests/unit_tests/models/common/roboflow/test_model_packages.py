import pytest
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.common.roboflow.model_packages import (
    AnySizePadding,
    AutoOrient,
    ColorMode,
    ForwardPassConfiguration,
    ImagePreProcessing,
    InferenceConfig,
    NetworkInputDefinition,
    ResizeMode,
    StaticCrop,
    TrainingInputSize,
    TRTConfig,
    parse_class_names_file,
    parse_inference_config,
    parse_key_points_metadata,
    parse_trt_config,
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
    keypoints, edges = parse_key_points_metadata(
        key_points_metadata_path=valid_key_points_metadata
    )

    # then
    assert keypoints == [
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
    assert edges == [[(0, 2)], [(0, 2)]]


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


def test_parse_key_points_metadata_with_missing_edges(
    key_points_metadata_with_missing_edges: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path=key_points_metadata_with_missing_edges
        )


def test_parse_key_points_metadata_with_malformed_edges(
    key_points_metadata_with_malformed_edges: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path=key_points_metadata_with_malformed_edges
        )


def test_parse_key_points_metadata_with_edges_pointing_non_existing_class(
    key_points_metadata_with_edges_pointing_non_existing_class: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_key_points_metadata(
            key_points_metadata_path=key_points_metadata_with_edges_pointing_non_existing_class
        )


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


def test_parse_inference_config_when_path_does_not_exists() -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path="/some/non/existing.json",
            allowed_resize_modes={ResizeMode.CENTER_CROP},
        )


def test_parse_inference_config_when_file_is_not_json(class_names_valid: str) -> None:
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path=class_names_valid, allowed_resize_modes={ResizeMode.CENTER_CROP}
        )


def test_parse_inference_config_when_json_file_does_not_define_dict_config(
    json_without_dict_path: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path=json_without_dict_path,
            allowed_resize_modes={ResizeMode.STRETCH_TO},
        )


def test_parse_inference_config_when_invalid_image_pre_processing(
    inference_config_invalid_image_pre_processing: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path=inference_config_invalid_image_pre_processing,
            allowed_resize_modes={ResizeMode.STRETCH_TO},
        )


def test_parse_inference_config_when_invalid_network_input(
    inference_config_invalid_network_input: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path=inference_config_invalid_network_input,
            allowed_resize_modes={ResizeMode.STRETCH_TO},
        )


def test_parse_inference_config_when_missing_network_input(
    inference_config_missing_network_input: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path=inference_config_missing_network_input,
            allowed_resize_modes={ResizeMode.STRETCH_TO},
        )


def test_parse_inference_config_when_invalid_forward_pass_config(
    inference_config_invalid_forward_pass_config: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path=inference_config_invalid_forward_pass_config,
            allowed_resize_modes={ResizeMode.STRETCH_TO},
        )


def test_parse_inference_config_when_invalid_post_processing_config(
    inference_config_invalid_post_processing_config: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path=inference_config_invalid_post_processing_config,
            allowed_resize_modes={ResizeMode.STRETCH_TO},
        )


def test_parse_inference_config_when_invalid_class_names_operations(
    inference_config_invalid_class_names_operations: str,
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path=inference_config_invalid_class_names_operations,
            allowed_resize_modes={ResizeMode.STRETCH_TO},
        )


def test_parse_inference_config_when_parsing_should_succeed(
    inference_config_valid_config: str,
) -> None:
    # when
    result = parse_inference_config(
        config_path=inference_config_valid_config,
        allowed_resize_modes={ResizeMode.STRETCH_TO},
    )

    # then
    expected_result = InferenceConfig(
        image_pre_processing=ImagePreProcessing(
            **{
                "auto-orient": AutoOrient(enabled=True),
                "static-crop": StaticCrop(
                    enabled=True, x_min=9, x_max=94, y_min=9, y_max=75
                ),
            }
        ),
        network_input=NetworkInputDefinition(
            training_input_size=TrainingInputSize(height=412, width=412),
            dynamic_spatial_size_supported=True,
            dynamic_spatial_size_mode=AnySizePadding(type="any-size"),
            color_mode=ColorMode.RGB,
            resize_mode=ResizeMode.STRETCH_TO,
            padding_value=None,
            input_channels=3,
            scaling_factor=None,
            normalization=None,
        ),
        forward_pass=ForwardPassConfiguration(
            static_batch_size=None, max_dynamic_batch_size=None
        ),
        post_processing=None,
        model_initialization=None,
        class_names_operations=None,
    )
    assert result == expected_result


def test_parse_inference_config_when_resize_mode_is_invalid(
    inference_config_valid_config: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = parse_inference_config(
            config_path=inference_config_valid_config,
            allowed_resize_modes={ResizeMode.LETTERBOX},
        )
