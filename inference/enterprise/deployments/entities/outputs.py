from typing import Literal

from pydantic import BaseModel


class JsonField(BaseModel):
    type: Literal["JsonField"]
    name: str
    selector: str
