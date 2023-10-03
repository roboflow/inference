def remove_empty_values(dictionary: dict) -> dict:
    return {k: v for k, v in dictionary.items() if v is not None}
