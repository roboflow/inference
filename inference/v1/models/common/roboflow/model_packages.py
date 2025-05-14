import json
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from inference.v1.entities import ImageDimensions
from inference.v1.errors import CorruptedModelPackageError
from inference.v1.utils.file_system import stream_file_lines, read_json


def parse_class_names_file(class_names_path: str) -> List[str]:
    try:
        return list(stream_file_lines(path=class_names_path))
    except OSError as error:
        raise CorruptedModelPackageError(
            f"Could not decode file {class_names_path} which is supposed to provide list of model class names. "
            f"If you created model package manually, please verify its consistency in docs. In case that the "
            f"weights are hosted on the Roboflow platform - contact support."
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


def parse_pre_processing_config(environment_file_path: str) -> PreProcessingConfig:
    try:
        content = read_json(path=environment_file_path)
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
        target_size = ImageDimensions(
            width=int(resize_config["width"]), height=int(resize_config["height"])
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
            f"Environment file located under path {environment_file_path} is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs."
        )


@dataclass
class ModelCharacteristics:
    task_type: str
    model_type: str


def parse_model_characteristics(config_path: str) -> ModelCharacteristics:
    try:
        with open(config_path) as f:
            parsed_config = json.load(f)
            if "project_task_type" not in parsed_config or "model_type" not in parsed_config:
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
            f"Model type config file located under path {config_path} is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs."
        )


def parse_key_points_metadata(key_points_metadata_path: str) -> List[List[str]]:
    try:
        with open(key_points_metadata_path) as f:
            parsed_config = json.load(f)
            if not isinstance(parsed_config, list):
                raise ValueError("config should contain list of key points descriptions for each instance")
            result: List[Optional[List[str]]] = [None] * len(parsed_config)
            for instance_key_point_description in parsed_config:
                if "object_class_id" not in instance_key_point_description:
                    raise ValueError("instance key point description lack 'object_class_id' key")
                object_class_id: int = instance_key_point_description["object_class_id"]
                if not 0 <= object_class_id <= len(result):
                    raise ValueError("`object_class_id` field point invalid class")
                result[object_class_id] = _retrieve_key_points_names(
                    instance_key_point_description=instance_key_point_description,
                )
            if any(e is None for e in result):
                raise ValueError("config does not provide metadata describing each instance key points")
            return result
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            f"Key points config file located under path {key_points_metadata_path} is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs."
        )


def _retrieve_key_points_names(instance_key_point_description: dict) -> List[str]:
    key_points_dump = sorted(
        [(int(k), v) for k, v in instance_key_point_description["keypoints"]],
        key=lambda e: e[0]
    )
    return [e[1] for e in key_points_dump]
