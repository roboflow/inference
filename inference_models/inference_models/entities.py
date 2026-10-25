from collections import namedtuple
from dataclasses import dataclass
from typing import Literal, Union

ImageDimensions = namedtuple("ImageDimensions", ["height", "width"])
ColorFormat = Literal["rgb", "bgr"]
Confidence = Union[float, Literal["best", "default"]]


@dataclass(frozen=True)
class ResolvedModelMetadata:
    model_id: str
    model_package_id: str
    backend: str
    quantization: str
