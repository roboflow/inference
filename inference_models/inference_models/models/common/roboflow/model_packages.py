from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from inference_models.entities import ImageDimensions
from inference_models.errors import CorruptedModelPackageError
from inference_models.logger import LOGGER
from inference_models.utils.file_system import read_json, stream_file_lines


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
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error


PADDING_VALUES_MAPPING = {
    "black edges": 0,
    "grey edges": 127,
    "white edges": 255,
}
StaticCropOffset = namedtuple(
    "StaticCropOffset",
    [
        "offset_x",
        "offset_y",
        "crop_width",
        "crop_height",
    ],
)
PreProcessingMetadata = namedtuple(
    "PreProcessingMetadata",
    [
        "pad_left",
        "pad_top",
        "pad_right",
        "pad_bottom",
        "original_size",
        "size_after_pre_processing",
        "inference_size",
        "scale_width",
        "scale_height",
        "static_crop_offset",
    ],
)


def parse_key_points_metadata(
    key_points_metadata_path: str,
) -> Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
    try:
        parsed_config = read_json(path=key_points_metadata_path)
        if not isinstance(parsed_config, list):
            raise ValueError(
                "config should contain list of key points descriptions for each instance"
            )
        class_names: List[Optional[List[str]]] = [None] * len(parsed_config)
        skeletons: List[Optional[List[Tuple[int, int]]]] = [None] * len(parsed_config)
        for instance_key_point_description in parsed_config:
            if "object_class_id" not in instance_key_point_description:
                raise ValueError(
                    "instance key point description lack 'object_class_id' key"
                )
            object_class_id: int = instance_key_point_description["object_class_id"]
            if not 0 <= object_class_id < len(class_names):
                raise ValueError("`object_class_id` field point invalid class")
            if "keypoints" not in instance_key_point_description:
                raise ValueError(
                    f"`keypoints` field not available in config for class with id {object_class_id}"
                )
            class_names[object_class_id] = _retrieve_key_points_names(
                key_points=instance_key_point_description["keypoints"],
            )
            key_points_count = len(class_names[object_class_id])
            if "edges" not in instance_key_point_description:
                raise ValueError(
                    f"`edges` field not available in config for class with id {object_class_id}"
                )
            skeletons[object_class_id] = _retrieve_skeleton(
                edges=instance_key_point_description["edges"],
                key_points_count=key_points_count,
            )
        if any(e is None for e in class_names):
            raise ValueError(
                "config does not provide metadata describing each instance key points"
            )
        if any(e is None for e in skeletons):
            raise ValueError(
                "config does not provide metadata describing each instance skeleton"
            )
        return class_names, skeletons
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Key points config file is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error


def _retrieve_key_points_names(key_points: dict) -> List[str]:
    key_points_dump = sorted(
        [(int(k), v) for k, v in key_points.items()],
        key=lambda e: e[0],
    )
    return [e[1] for e in key_points_dump]


def _retrieve_skeleton(
    edges: List[dict], key_points_count: int
) -> List[Tuple[int, int]]:
    result = []
    for edge in edges:
        if not isinstance(edge, dict) or "from" not in edge or "to" not in edge:
            raise ValueError(
                "skeleton edge malformed - invalid format or lack of required keys"
            )
        start = edge["from"]
        end = edge["to"]
        if not 0 <= start < key_points_count or not 0 <= end < key_points_count:
            raise ValueError(
                "skeleton edge malformed - identifier of skeleton edge end is out of allowed range determined by "
                "the number of key points in the skeleton"
            )
        result.append((edge["from"], edge["to"]))
    return result


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
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error


class AutoOrient(BaseModel):
    enabled: bool


class StaticCrop(BaseModel):
    enabled: bool
    x_min: int
    x_max: int
    y_min: int
    y_max: int


class ContrastType(str, Enum):
    ADAPTIVE_EQUALIZATION = "Adaptive Equalization"
    CONTRAST_STRETCHING = "Contrast Stretching"
    HISTOGRAM_EQUALIZATION = "Histogram Equalization"


class Contrast(BaseModel):
    enabled: bool
    type: ContrastType


class Grayscale(BaseModel):
    enabled: bool


class ImagePreProcessing(BaseModel):
    auto_orient: Optional[AutoOrient] = Field(alias="auto-orient", default=None)
    static_crop: Optional[StaticCrop] = Field(alias="static-crop", default=None)
    contrast: Optional[Contrast] = Field(default=None)
    grayscale: Optional[Grayscale] = Field(default=None)


class TrainingInputSize(BaseModel):
    height: int
    width: int


class DivisiblePadding(BaseModel):
    type: Literal["pad-to-be-divisible"]
    value: int


class AnySizePadding(BaseModel):
    type: Literal["any-size"]


class ColorMode(str, Enum):
    BGR = "bgr"
    RGB = "rgb"


class ResizeMode(str, Enum):
    STRETCH_TO = "stretch"
    LETTERBOX = "letterbox"
    CENTER_CROP = "center-crop"
    FIT_LONGER_EDGE = "fit-longer-edge"
    LETTERBOX_REFLECT_EDGES = "letterbox-reflect-edges"


Number = Union[int, float]


class NetworkInputDefinition(BaseModel):
    training_input_size: TrainingInputSize
    dynamic_spatial_size_supported: bool
    dynamic_spatial_size_mode: Optional[Union[DivisiblePadding, AnySizePadding]] = (
        Field(discriminator="type", default=None)
    )
    color_mode: ColorMode
    resize_mode: ResizeMode
    padding_value: Optional[int] = Field(default=None)
    input_channels: int
    scaling_factor: Optional[Number] = Field(default=None)
    normalization: Optional[Tuple[List[Number], List[Number]]] = Field(default=None)


class ForwardPassConfiguration(BaseModel):
    static_batch_size: Optional[int] = Field(default=None)
    max_dynamic_batch_size: Optional[int] = Field(default=None)


class FusedNMSParameters(BaseModel):
    max_detections: int
    confidence_threshold: float
    iou_threshold: float
    class_agnostic: int


class NMSPostProcessing(BaseModel):
    type: Literal["nms"]
    fused: bool
    nms_parameters: Optional[FusedNMSParameters] = Field(default=None)


class SigmoidPostProcessing(BaseModel):
    type: Literal["sigmoid"]
    fused: bool


class SoftMaxPostProcessing(BaseModel):
    type: Literal["softmax"]
    fused: bool


ImagePreProcessingValidator = BeforeValidator(
    lambda value: value if value is not None else ImagePreProcessing()
)


class ClassNameRemoval(BaseModel):
    type: Literal["class_name_removal"]
    class_name: str


class InferenceConfig(BaseModel):
    image_pre_processing: Annotated[ImagePreProcessing, ImagePreProcessingValidator] = (
        Field(default_factory=lambda: ImagePreProcessing())
    )
    network_input: NetworkInputDefinition
    forward_pass: ForwardPassConfiguration = Field(
        default_factory=lambda: ForwardPassConfiguration()
    )
    post_processing: Optional[
        Union[NMSPostProcessing, SoftMaxPostProcessing, SigmoidPostProcessing]
    ] = Field(default=None, discriminator="type")
    model_initialization: Optional[dict] = Field(default=None)
    class_names_operations: Optional[
        List[Annotated[Union[ClassNameRemoval], Field(discriminator="type")]]
    ] = Field(default=None)


def parse_inference_config(
    config_path: str,
    allowed_resize_modes: Set[ResizeMode],
    implicit_resize_mode_substitutions: Optional[
        Dict[ResizeMode, Tuple[ResizeMode, Optional[int], Optional[str]]]
    ] = None,
) -> InferenceConfig:
    try:
        decoded_config = read_json(path=config_path)
        if not isinstance(decoded_config, dict):
            raise ValueError(
                f"Expected config format is dict, found {type(decoded_config)} instead"
            )
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Inference config file of the model package is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error
    try:
        parsed_config = InferenceConfig.model_validate(decoded_config)
    except ValidationError as error:
        raise CorruptedModelPackageError(
            message=f"Could not parse the inference config from the model package.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error
    if implicit_resize_mode_substitutions is None:
        implicit_resize_mode_substitutions = {}
    if parsed_config.network_input.resize_mode in implicit_resize_mode_substitutions:
        substitution, padding, reason = implicit_resize_mode_substitutions[
            parsed_config.network_input.resize_mode
        ]
        if reason is not None:
            LOGGER.warning(reason)
        parsed_config.network_input.resize_mode = substitution
        parsed_config.network_input.padding_value = padding
    if parsed_config.network_input.resize_mode not in allowed_resize_modes:
        allowed_resize_modes_str = ", ".join([e.value for e in allowed_resize_modes])
        raise CorruptedModelPackageError(
            message=f"Inference configuration shipped with model package defines input resize "
            f"{parsed_config.network_input.resize_mode} which is not supported by the model implementation. "
            f"Config defines: {parsed_config.network_input.resize_mode.value}, but the allowed values are: "
            f"{allowed_resize_modes_str}.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return parsed_config
