from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelDescription:
    model_id: str
    task_type: str
    batch_size: Optional[int]
    input_height: Optional[int]
    input_width: Optional[int]
