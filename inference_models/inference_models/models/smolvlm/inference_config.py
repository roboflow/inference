"""SmolVLM packages may delegate all image resizing to their HF processor.

A null training size is not a synthetic fixed resolution. The trainer emits an
identity platform transform for this case; retain original image dimensions and
let the packaged processor perform the same preprocessing as during training.
Other configurations continue through the existing fixed-size parser.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
    read_json,
)


class _AnyImageSize(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["any-size"]


class _ProcessorOwnedNetworkInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_input_size: None
    dynamic_spatial_size_supported: Literal[True]
    dynamic_spatial_size_mode: _AnyImageSize
    color_mode: Literal["rgb"]
    resize_mode: Literal["stretch"]
    input_channels: Literal[3]
    padding_value: None = None
    scaling_factor: None = None
    normalization: None = None


class _ProcessorOwnedInputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_pre_processing: None = None
    network_input: _ProcessorOwnedNetworkInput


def parse_smolvlm_inference_config(config_path: str) -> Optional[InferenceConfig]:
    decoded = read_json(path=config_path)
    network = decoded.get("network_input") if isinstance(decoded, dict) else None
    if (
        isinstance(network, dict)
        and "training_input_size" in network
        and network["training_input_size"] is None
    ):
        try:
            _ProcessorOwnedInputConfig.model_validate(decoded)
        except ValidationError as error:
            raise CorruptedModelPackageError(
                message="SmolVLM without a training resize must use an identity "
                "platform transform and a dynamic processor-owned input size.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from error
        return None
    return parse_inference_config(
        config_path=config_path,
        allowed_resize_modes={
            ResizeMode.STRETCH_TO,
            ResizeMode.LETTERBOX,
            ResizeMode.CENTER_CROP,
            ResizeMode.LETTERBOX_REFLECT_EDGES,
            ResizeMode.FIT_LONGER_EDGE,
        },
    )
