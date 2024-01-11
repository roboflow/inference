from typing import Literal

from pydantic import BaseModel

from inference.enterprise.deployments.entities.base import GraphNone


class JsonField(BaseModel, GraphNone):
    type: Literal["JsonField"]
    name: str
    selector: str

    def get_type(self) -> str:
        return self.type
