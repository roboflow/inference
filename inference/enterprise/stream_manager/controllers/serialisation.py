from datetime import datetime, date
from enum import Enum
from typing import Any


def serialise_to_json(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if issubclass(type(obj), Enum):
        return obj.value
    raise TypeError(f"Type {type(obj)} not serializable")
