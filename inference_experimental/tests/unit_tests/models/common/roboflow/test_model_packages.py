import pytest
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.common.roboflow.model_packages import (
    TRTConfig,
    parse_class_map_from_environment_file,
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
