from typing import Any


def safe_stringify(value: Any, max_characters: int = 128) -> str:
    try:
        str_value = str(value)
        if len(str_value) > max_characters:
            str_value = f"{str_value[:max_characters]} [...]"
        return str_value
    except Exception:
        return "could not get string representation of value"
