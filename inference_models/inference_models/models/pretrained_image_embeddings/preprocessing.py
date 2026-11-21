"""Torch implementation of the Hugging Face image processors that pretrained
embedding packages ship as a verbatim ``preprocessor_config.json``.

Only the subset used by the BiT (DINOv2), SigLIP and DINOv3 processors is
supported. Anything else is rejected while loading the package, so a package
never runs with preprocessing that differs silently from its source model.
"""

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    model_validator,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

from inference_models.entities import ColorFormat
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.clip.preprocessing import inputs_to_tensor
from inference_models.utils.file_system import read_json

CORRUPTED_PACKAGE_HELP_URL = "https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror"

RESAMPLE_INTERPOLATIONS = {
    2: InterpolationMode.BILINEAR,
    3: InterpolationMode.BICUBIC,
}
# Supported processors, and whether each rescales before resizing: transformers
# resizes the uint8 image and rescales afterwards, except for the DINOv3
# processor, which rescales first and resizes the float image.
RESCALE_BEFORE_RESIZE_BY_PROCESSOR = {
    "BitImageProcessor": False,
    "SiglipImageProcessor": False,
    "DINOv3ViTImageProcessor": True,
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShortestEdgeSize(_StrictModel):
    shortest_edge: PositiveInt


class HeightWidthSize(_StrictModel):
    height: PositiveInt
    width: PositiveInt


class ImageProcessorConfig(_StrictModel):
    image_processor_type: str
    do_convert_rgb: Optional[bool] = None
    do_resize: bool
    size: Optional[Union[ShortestEdgeSize, HeightWidthSize]] = None
    resample: Optional[int] = None
    do_center_crop: Optional[bool] = None
    crop_size: Optional[HeightWidthSize] = None
    do_rescale: bool
    rescale_factor: Optional[PositiveFloat] = None
    do_normalize: bool
    image_mean: Optional[Tuple[float, float, float]] = None
    image_std: Optional[Tuple[PositiveFloat, PositiveFloat, PositiveFloat]] = None
    # Settings of transformers processors that do not change the pixel values
    # produced for 3-channel images, accepted only with their neutral values.
    processor_class: Optional[str] = None
    data_format: Optional[Literal["channels_first"]] = None
    input_data_format: None = None
    return_tensors: None = None
    device: None = None
    disable_grouping: Optional[bool] = None
    default_to_square: Optional[bool] = None
    do_pad: Optional[Literal[False]] = None
    pad_size: None = None

    @model_validator(mode="after")
    def _validate_supported_subset(self) -> "ImageProcessorConfig":
        if self.processor_type not in RESCALE_BEFORE_RESIZE_BY_PROCESSOR:
            raise ValueError(
                f"image_processor_type `{self.image_processor_type}` is not supported, "
                f"use one of {sorted(RESCALE_BEFORE_RESIZE_BY_PROCESSOR)}"
            )
        if self.do_resize:
            if self.size is None:
                raise ValueError("do_resize requires size")
            if self.resample not in RESAMPLE_INTERPOLATIONS:
                raise ValueError(
                    f"resample must be one of {sorted(RESAMPLE_INTERPOLATIONS)} "
                    f"(2 = bilinear, 3 = bicubic), got {self.resample}"
                )
        if self.do_center_crop and self.crop_size is None:
            raise ValueError("do_center_crop requires crop_size")
        if self.do_rescale and self.rescale_factor is None:
            raise ValueError("do_rescale requires rescale_factor")
        if self.do_normalize and (self.image_mean is None or self.image_std is None):
            raise ValueError("do_normalize requires image_mean and image_std")
        if self.output_size is None:
            raise ValueError(
                "preprocessing must produce a fixed image size: enable do_center_crop "
                "or resize to an explicit height and width"
            )
        return self

    @property
    def processor_type(self) -> str:
        return self.image_processor_type.removesuffix("Fast")

    @property
    def rescale_before_resize(self) -> bool:
        return RESCALE_BEFORE_RESIZE_BY_PROCESSOR[self.processor_type]

    @property
    def output_size(self) -> Optional[Tuple[int, int]]:
        if self.do_center_crop:
            return self.crop_size.height, self.crop_size.width
        if self.do_resize and isinstance(self.size, HeightWidthSize):
            return self.size.height, self.size.width
        return None

    @property
    def interpolation(self) -> InterpolationMode:
        return RESAMPLE_INTERPOLATIONS[self.resample]

    def describe(self) -> dict:
        """Effective preprocessing as JSON; parameters of disabled steps are None."""
        return {
            "image_processor_type": self.processor_type,
            "do_convert_rgb": bool(self.do_convert_rgb),
            "do_resize": self.do_resize,
            "size": self.size.model_dump() if self.do_resize else None,
            "resample": self.resample if self.do_resize else None,
            "do_center_crop": bool(self.do_center_crop),
            "crop_size": (self.crop_size.model_dump() if self.do_center_crop else None),
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor if self.do_rescale else None,
            "do_normalize": self.do_normalize,
            "image_mean": list(self.image_mean) if self.do_normalize else None,
            "image_std": list(self.image_std) if self.do_normalize else None,
        }


def parse_image_processor_config(config_path: str) -> ImageProcessorConfig:
    try:
        raw_config = read_json(path=config_path)
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Could not read `preprocessor_config.json` of the model package: {error}",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        ) from error
    return image_processor_config_from_dict(raw_config=raw_config)


def image_processor_config_from_dict(raw_config: dict) -> ImageProcessorConfig:
    if not isinstance(raw_config, dict):
        raise CorruptedModelPackageError(
            message="`preprocessor_config.json` of the model package must be a JSON object.",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        )
    try:
        return ImageProcessorConfig.model_validate(raw_config)
    except ValidationError as error:
        raise CorruptedModelPackageError(
            message=f"`preprocessor_config.json` of the model package declares unsupported "
            f"preprocessing: {error}",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        ) from error


def pre_process_images(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    config: ImageProcessorConfig,
    device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
) -> torch.Tensor:
    """Turn images into the float32 NCHW batch the model package expects.

    Numpy images default to BGR and tensors to RGB, as elsewhere in the library.
    Tensors hold pixel values in the 0-255 range, in channels-first layout.
    """
    tensors = inputs_to_tensor(
        images=images, device=device, input_color_format=input_color_format
    )
    if isinstance(tensors, torch.Tensor):
        return _transform(batch=tensors, config=config)
    return torch.cat(
        [_transform(batch=tensor, config=config) for tensor in tensors], dim=0
    ).contiguous()


def _transform(batch: torch.Tensor, config: ImageProcessorConfig) -> torch.Tensor:
    rescale_first = config.do_rescale and config.rescale_before_resize
    if rescale_first:
        batch = batch.to(torch.float32) * config.rescale_factor
    if config.do_resize:
        batch = F.resize(
            batch,
            _resized_size(height=batch.shape[-2], width=batch.shape[-1], config=config),
            interpolation=config.interpolation,
            antialias=True,
        )
    if config.do_center_crop:
        batch = _center_crop(batch=batch, crop_size=config.crop_size)
    batch = batch.to(torch.float32)
    if config.do_rescale and not rescale_first:
        batch = batch * config.rescale_factor
    if config.do_normalize:
        batch = F.normalize(
            batch, mean=list(config.image_mean), std=list(config.image_std)
        )
    return batch.contiguous()


def _resized_size(height: int, width: int, config: ImageProcessorConfig) -> List[int]:
    if isinstance(config.size, HeightWidthSize):
        return [config.size.height, config.size.width]
    # Same rounding as transformers' `get_resize_output_image_size`.
    short, long = (width, height) if width <= height else (height, width)
    new_short = config.size.shortest_edge
    new_long = int(new_short * long / short)
    return [new_long, new_short] if width <= height else [new_short, new_long]


def _center_crop(batch: torch.Tensor, crop_size: HeightWidthSize) -> torch.Tensor:
    # transformers crops at the floor of the offset and pads smaller images with
    # zeros, the extra pixel of odd padding going to the right and bottom.
    height, width = batch.shape[-2:]
    pad_height = max(crop_size.height - height, 0)
    pad_width = max(crop_size.width - width, 0)
    if pad_height or pad_width:
        batch = torch.nn.functional.pad(
            batch,
            (
                pad_width // 2,
                (pad_width + 1) // 2,
                pad_height // 2,
                (pad_height + 1) // 2,
            ),
        )
        height, width = batch.shape[-2:]
    top = (height - crop_size.height) // 2
    left = (width - crop_size.width) // 2
    return batch[..., top : top + crop_size.height, left : left + crop_size.width]
