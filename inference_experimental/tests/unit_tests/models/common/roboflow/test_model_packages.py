import pytest
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.common.roboflow.model_packages import (
    TRTConfig,
    parse_class_names_file,
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
