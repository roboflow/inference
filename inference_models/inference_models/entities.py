from collections import namedtuple
from typing import Annotated, Literal, Union

from annotated_types import Ge, Le

ImageDimensions = namedtuple("ImageDimensions", ["height", "width"])
ColorFormat = Literal["rgb", "bgr"]
Confidence = Union[Annotated[float, Ge(0), Le(1)], Literal["best", "default"]]
