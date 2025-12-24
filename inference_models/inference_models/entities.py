from collections import namedtuple
from typing import Literal

ImageDimensions = namedtuple("ImageDimensions", ["height", "width"])
ColorFormat = Literal["rgb", "bgr"]
