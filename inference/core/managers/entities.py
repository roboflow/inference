from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ModelDescription:
    model_id: str
    task_type: str
    batch_size: Optional[int]
    input_height: Optional[int]
    input_width: Optional[int]
    vram_bytes: Optional[int] = None
    request_aliases: List[str] = field(default_factory=list)
