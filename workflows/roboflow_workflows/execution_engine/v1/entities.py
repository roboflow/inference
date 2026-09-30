from typing import List, Optional, Union

from pydantic import BaseModel, Field


class FlowControl(BaseModel):
    context: Optional[Union[str, List[str]]] = Field(default=None)
