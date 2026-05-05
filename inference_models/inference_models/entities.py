from collections import namedtuple
from typing import Literal, Union

ImageDimensions = namedtuple("ImageDimensions", ["height", "width"])
ColorFormat = Literal["rgb", "bgr"]
Confidence = Union[float, Literal["best", "default"]]
