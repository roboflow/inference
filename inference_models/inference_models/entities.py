from collections import namedtuple
from typing import Annotated, Literal, Union

from pydantic import Field

ImageDimensions = namedtuple("ImageDimensions", ["height", "width"])
ColorFormat = Literal["rgb", "bgr"]
Confidence = Union[Annotated[float, Field(ge=0, le=1)], Literal["best", "default"]]
