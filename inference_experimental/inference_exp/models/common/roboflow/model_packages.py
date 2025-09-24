import json
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from inference_exp.entities import ImageDimensions
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.utils.file_system import read_json, stream_file_lines


def parse_class_names_file(class_names_path: str) -> List[str]:
    try:
        result = list(stream_file_lines(path=class_names_path))
        if not result:
            raise ValueError("Empty class list")
        return result
    except (OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Could not decode file which is supposed to provide list of model class names. Error: {error}."
            f"If you created model package manually, please verify its consistency in docs. In case that the "
            f"weights are hosted on the Roboflow platform - contact support.",
            help_url="https://todo",
        ) from error


class PreProcessingMode(Enum):
    NONE = "NONE"
    STRETCH = "STRETCH"
    LETTERBOX = "LETTERBOX"


@dataclass(frozen=True)
class PreProcessingConfig:
    mode: PreProcessingMode
    target_size: Optional[ImageDimensions] = None
    padding_value: Optional[int] = None


PADDING_VALUES_MAPPING = {
    "black edges": 0,
    "grey edges": 127,
    "white edges": 255,
}
PreProcessingMetadata = namedtuple(
    "PreProcessingMetadata",
    [
        "pad_left",
        "pad_top",
        "pad_right",
        "pad_bottom",
        "original_size",
        "inference_size",
        "scale_width",
        "scale_height",
    ],
)


def parse_pre_processing_config(config_path: str) -> PreProcessingConfig:
    try:
        content = read_json(path=config_path)
        if not content:
            raise ValueError("file is empty.")
        if not isinstance(content, dict):
            raise ValueError("file is malformed (not a JSON dictionary)")
        if "PREPROCESSING" not in content:
            raise ValueError("file is malformed (lack of `PREPROCESSING` key)")
        preprocessing_dict = json.loads(content["PREPROCESSING"])
        resize_config = preprocessing_dict["resize"]
        if not resize_config["enabled"]:
            return PreProcessingConfig(mode=PreProcessingMode.NONE)
        if "width" not in resize_config or "height" not in resize_config:
            raise ValueError(
                "file is malformed (lack of `width` or `height` key in dictionary specifying preprocessing)"
            )
        target_size = ImageDimensions(
            width=int(resize_config["width"]), height=int(resize_config["height"])
        )
        if "format" not in resize_config:
            raise ValueError(
                "file is malformed (lack of `format` key in dictionary specifying preprocessing)"
            )
        if resize_config["format"] == "Stretch to":
            return PreProcessingConfig(
                mode=PreProcessingMode.STRETCH, target_size=target_size
            )
        for padding_color_infix, padding_value in PADDING_VALUES_MAPPING.items():
            if padding_color_infix in resize_config["format"]:
                return PreProcessingConfig(
                    mode=PreProcessingMode.LETTERBOX,
                    target_size=target_size,
                    padding_value=padding_value,
                )
        raise ValueError("could not determine resize method or padding color")
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Environment file is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://todo",
        ) from error


@dataclass
class ModelCharacteristics:
    task_type: str
    model_type: str


def parse_model_characteristics(config_path: str) -> ModelCharacteristics:
    try:
        parsed_config = read_json(path=config_path)
        if not isinstance(parsed_config, dict):
            raise ValueError(
                f"decoded value is {type(parsed_config)}, but dictionary expected"
            )
        if (
            "project_task_type" not in parsed_config
            or "model_type" not in parsed_config
        ):
            raise ValueError(
                "could not find required entries in config - either "
                "'project_task_type' or 'model_type' field is missing"
            )
        return ModelCharacteristics(
            task_type=parsed_config["project_task_type"],
            model_type=parsed_config["model_type"],
        )
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Model type config file is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://todo",
        ) from error


def parse_key_points_metadata(key_points_metadata_path: str) -> List[List[str]]:
    try:
        parsed_config = read_json(path=key_points_metadata_path)
        if not isinstance(parsed_config, list):
            raise ValueError(
                "config should contain list of key points descriptions for each instance"
            )
        result: List[Optional[List[str]]] = [None] * len(parsed_config)
        for instance_key_point_description in parsed_config:
            if "object_class_id" not in instance_key_point_description:
                raise ValueError(
                    "instance key point description lack 'object_class_id' key"
                )
            object_class_id: int = instance_key_point_description["object_class_id"]
            if not 0 <= object_class_id < len(result):
                raise ValueError("`object_class_id` field point invalid class")
            if "keypoints" not in instance_key_point_description:
                raise ValueError(
                    f"`keypoints` field not available in config for class with id {object_class_id}"
                )
            result[object_class_id] = _retrieve_key_points_names(
                key_points=instance_key_point_description["keypoints"],
            )
        if any(e is None for e in result):
            raise ValueError(
                "config does not provide metadata describing each instance key points"
            )
        return result
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Key points config file is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://todo",
        ) from error


def _retrieve_key_points_names(key_points: dict) -> List[str]:
    key_points_dump = sorted(
        [(int(k), v) for k, v in key_points.items()],
        key=lambda e: e[0],
    )
    return [e[1] for e in key_points_dump]


@dataclass
class TRTConfig:
    static_batch_size: Optional[int]
    dynamic_batch_size_min: Optional[int]
    dynamic_batch_size_opt: Optional[int]
    dynamic_batch_size_max: Optional[int]


def parse_trt_config(config_path: str) -> TRTConfig:
    try:
        parsed_config = read_json(path=config_path)
        if not isinstance(parsed_config, dict):
            raise ValueError(
                f"Expected config format is dict, found {type(parsed_config)} instead"
            )
        config = TRTConfig(
            static_batch_size=parsed_config.get("static_batch_size"),
            dynamic_batch_size_min=parsed_config.get("dynamic_batch_size_min"),
            dynamic_batch_size_opt=parsed_config.get("dynamic_batch_size_opt"),
            dynamic_batch_size_max=parsed_config.get("dynamic_batch_size_max"),
        )
        if config.static_batch_size is not None:
            if config.static_batch_size <= 0:
                raise ValueError(
                    f"invalid static batch size - {config.static_batch_size}"
                )
            return config
        if (
            config.dynamic_batch_size_min is None
            or config.dynamic_batch_size_opt is None
            or config.dynamic_batch_size_max is None
        ):
            raise ValueError(
                "configuration does not provide information about boundaries for dynamic batch size"
            )
        if (
            config.dynamic_batch_size_min <= 0
            or config.dynamic_batch_size_max < config.dynamic_batch_size_min
            or config.dynamic_batch_size_opt < config.dynamic_batch_size_min
            or config.dynamic_batch_size_opt > config.dynamic_batch_size_max
        ):
            raise ValueError(f"invalid dynamic batch size")
        return config
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"TRT config file of the model package is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://todo",
        ) from error


def parse_class_map_from_environment_file(environment_file_path: str) -> List[str]:
    try:
        parsed_config = read_json(path=environment_file_path)
        if "CLASS_MAP" not in parsed_config:
            raise ValueError("config does not provide `CLASS_MAP` config")
        class_map_dict = parsed_config["CLASS_MAP"]
        class_map: List[Optional[str]] = [None] * len(class_map_dict)
        for class_id, class_name in class_map_dict.items():
            class_map[int(class_id)] = class_name
        if any(c is None for c in class_map):
            raise ValueError(
                "class mapping does not provide class name for every class id"
            )
        return class_map
    except (IOError, OSError, ValueError, IndexError) as error:
        raise CorruptedModelPackageError(
            message=f"Environment file is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://todo",
        ) from error
