import hashlib


def get_string_list_hash(text: list) -> str:
    """Get the hash of a list of strings.

    Args:
        text (list): The list of strings.

    Returns:
        str: The hash of the list of strings.
    """
    text_string = ", ".join([f"{idx}:{t}" for idx, t in enumerate(text)])
    text_hash = hashlib.md5(text_string.encode("utf-8")).hexdigest()
    return text_hash
